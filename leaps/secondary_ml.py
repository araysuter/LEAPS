"""Conservative ML validation for LEAPS secondary-eclipse analyses.

This module deliberately does *not* turn a classifier into an eclipse finder.
It uses the same fixed-phase eclipse model and red-noise treatment as the
normal :class:`~leaps.science.SecondaryEclipseService` to make a small,
held-out injection/recovery experiment.  Its only job is to answer a narrow
question: on this imported TESS data set, does a classifier recover injected
eclipses more efficiently than LEAPS' existing, physics-first candidate rule?

The answer is written as a validation result, never used to change the normal
Secondary Eclipse outcome.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import PlanetParameters
from .models import JobStatus, LEAPSError, StageID
from .project import ProjectWorkspace
from .science import CancellationToken, Emitter, SecondaryEclipseService, _emit

FEATURE_NAMES = (
    "Fitted depth (ppm)",
    "Depth uncertainty (ppm)",
    "Red-noise S/N",
    "Red-noise beta",
    "Residual RMS (ppm)",
    "Eclipse improvement (delta chi-squared)",
    "Strongest nearby-control S/N",
    "Positive-depth sector fraction",
    "Inter-sector depth scatter",
)
FEATURE_KEYS = (
    "depth_ppm",
    "depth_uncertainty_ppm",
    "significance",
    "red_noise_beta",
    "residual_rms_ppm",
    "delta_chi_squared",
    "control_significance",
    "positive_sector_fraction",
    "sector_depth_scatter",
)
DEFAULT_DEPTHS_PPM = (25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 400.0)


@dataclass(slots=True)
class MLValidationResult:
    """Portable summary of one secondary-eclipse ML validation run."""

    output_path: Path
    preview_path: Path
    summary_path: Path
    message: str
    recommendation: str
    test_auc: float
    test_false_alarm_rate: float
    calibration_false_alarm_target: float
    ml_recovery_50_ppm: float | None
    rule_recovery_50_ppm: float | None
    train_segments: list[str]
    test_segments: list[str]
    trial_count: int
    raw: dict[str, Any]


class _FixedPhaseEvaluator:
    """Fast, numerically equivalent evaluator for LEAPS' local eclipse fit.

    ``SecondaryEclipseService._fit_window`` is intentionally easy to read and
    writes a rich diagnostic.  An injection experiment calls that model many
    times, though, so this small evaluator caches the fixed design matrices and
    solves the same weighted least-squares equations with compact normal
    matrices.  It keeps LEAPS' clipping and red-noise beta calculation exactly
    the same.  Tests compare the two implementations on synthetic data.
    """

    def __init__(
        self,
        phase: np.ndarray,
        times: np.ndarray,
        uncertainty: np.ndarray,
        *,
        duration_phase: float,
        window_phase: float,
        baseline: str,
    ) -> None:
        self.phase = np.asarray(phase, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.uncertainty = np.asarray(uncertainty, dtype=float)
        self.duration_phase = float(duration_phase)
        self.window_phase = float(window_phase)
        self.baseline = baseline
        self.local_mask = np.abs(self.phase) <= self.window_phase
        self.in_eclipse_mask = self.local_mask & (
            np.abs(self.phase) <= self.duration_phase / 2.0
        )
        before_mask = self.local_mask & (self.phase < -self.duration_phase / 2.0)
        after_mask = self.local_mask & (self.phase > self.duration_phase / 2.0)
        self.coverage = {
            "available": bool(
                self.local_mask.sum() >= 12
                and self.in_eclipse_mask.sum() >= 3
                and before_mask.sum() >= 3
                and after_mask.sum() >= 3
            ),
            "local_points": int(self.local_mask.sum()),
            "in_eclipse_points": int(self.in_eclipse_mask.sum()),
            "before_points": int(before_mask.sum()),
            "after_points": int(after_mask.sum()),
        }
        self.local_phase = self.phase[self.local_mask]
        self.local_times = self.times[self.local_mask]
        self.local_uncertainty = self.uncertainty[self.local_mask]
        self.template = SecondaryEclipseService._eclipse_template(
            self.local_phase, self.duration_phase
        )
        x = self.local_phase / max(self.window_phase, 1e-8)
        design_baseline = [np.ones_like(x)]
        if baseline in {"linear", "quadratic"}:
            design_baseline.append(x)
        if baseline == "quadratic":
            design_baseline.append(x**2)
        self.baseline_matrix = np.column_stack(design_baseline)
        self.design = np.column_stack((self.baseline_matrix, -self.template))
        if self.coverage["available"] and np.linalg.matrix_rank(self.design) < self.design.shape[1]:
            self.coverage["available"] = False
            self.coverage["reason"] = "The observed phase range cannot separate an eclipse from the baseline."

    @staticmethod
    def _weighted_fit(
        design_used: np.ndarray,
        flux_used: np.ndarray,
        uncertainty_used: np.ndarray,
        design_all: np.ndarray,
        flux_all: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        weights = 1.0 / np.square(uncertainty_used)
        normal = design_used.T @ (weights[:, None] * design_used)
        right_hand = design_used.T @ (weights * flux_used)
        try:
            coefficients = np.linalg.solve(normal, right_hand)
        except np.linalg.LinAlgError:
            coefficients = np.linalg.pinv(normal) @ right_hand
        model_used = design_used @ coefficients
        residuals_used = flux_used - model_used
        degrees_of_freedom = max(1, flux_used.size - design_used.shape[1])
        chi_squared = float(np.sum(np.square(residuals_used / uncertainty_used)))
        scale = max(1.0, chi_squared / degrees_of_freedom)
        covariance = np.linalg.pinv(normal) * scale
        model_all = design_all @ coefficients
        return coefficients, covariance, model_all, np.asarray(flux_all) - model_all, chi_squared

    def evaluate(self, flux: np.ndarray) -> dict[str, Any]:
        flux = np.asarray(flux, dtype=float)
        result: dict[str, Any] = {
            "local_mask": self.local_mask,
            "in_eclipse_mask": self.in_eclipse_mask,
            "coverage": dict(self.coverage),
            "model": np.full(int(self.local_mask.sum()), np.nan),
            "baseline_model": np.full(int(self.local_mask.sum()), np.nan),
            "residuals": np.full(int(self.local_mask.sum()), np.nan),
            "template": self.template,
            "depth": None,
            "depth_uncertainty": None,
            "significance": None,
            "red_noise_beta": None,
            "residual_rms": None,
            "delta_chi_squared": None,
            "points_used": 0,
        }
        if not self.coverage["available"]:
            return result

        local_flux = flux[self.local_mask]
        keep = np.ones(local_flux.size, dtype=bool)
        for _ in range(2):
            _, _, _, residuals, _ = self._weighted_fit(
                self.design[keep],
                local_flux[keep],
                self.local_uncertainty[keep],
                self.design,
                local_flux,
            )
            scatter = max(
                SecondaryEclipseService._robust_scatter(residuals[keep]),
                float(np.nanmedian(self.local_uncertainty[keep])),
            )
            updated_keep = np.abs(residuals) <= 5.0 * max(scatter, 1e-12)
            if updated_keep.sum() < self.design.shape[1] + 3 or np.array_equal(updated_keep, keep):
                break
            keep = updated_keep

        coefficients, covariance, model, residuals, chi_squared = self._weighted_fit(
            self.design[keep],
            local_flux[keep],
            self.local_uncertainty[keep],
            self.design,
            local_flux,
        )
        _, _, _, _, no_eclipse_chi_squared = self._weighted_fit(
            self.baseline_matrix[keep],
            local_flux[keep],
            self.local_uncertainty[keep],
            self.baseline_matrix,
            local_flux,
        )
        formal_uncertainty = float(math.sqrt(max(float(covariance[-1, -1]), 0.0)))
        beta = SecondaryEclipseService._red_noise_beta(residuals[keep], self.local_times[keep])
        depth_uncertainty = formal_uncertainty * beta
        depth = float(coefficients[-1])
        result.update(
            {
                "model": model,
                "baseline_model": model + depth * self.template,
                "residuals": residuals,
                "depth": depth,
                "depth_uncertainty": depth_uncertainty,
                "significance": depth / depth_uncertainty if depth_uncertainty > 0 else None,
                "red_noise_beta": beta,
                "residual_rms": float(np.std(residuals[keep], ddof=1)) if keep.sum() > 1 else None,
                "delta_chi_squared": max(0.0, no_eclipse_chi_squared - chi_squared),
                "points_used": int(keep.sum()),
                "kept_mask": keep,
            }
        )
        return result


@dataclass(slots=True)
class _SectorNoise:
    label: str
    indices: np.ndarray
    local_indices: np.ndarray
    baseline: np.ndarray
    residuals: np.ndarray
    template: np.ndarray


class _InjectionGroup:
    """A disjoint group of TESS sectors that can produce synthetic trials."""

    def __init__(
        self,
        sectors: list[_SectorNoise],
        time: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
        parameters: PlanetParameters,
        *,
        expected_phase: float,
        duration_phase: float,
        window_phase: float,
        baseline: str,
    ) -> None:
        combined = np.concatenate([sector.indices for sector in sectors])
        order = np.argsort(time[combined])
        self.indices = combined[order]
        self.time = time[self.indices]
        self.base_flux = flux[self.indices].copy()
        self.uncertainty = uncertainty[self.indices]
        original_to_group = np.full(time.size, -1, dtype=int)
        original_to_group[self.indices] = np.arange(self.indices.size)
        self.local_payload = [
            (
                original_to_group[sector.local_indices],
                sector.baseline,
                sector.residuals,
                sector.template,
            )
            for sector in sectors
        ]
        expected = SecondaryEclipseService._relative_phase(
            self.time, float(parameters.mid_time), float(parameters.period), expected_phase
        )
        self.evaluator = _FixedPhaseEvaluator(
            expected,
            self.time,
            self.uncertainty,
            duration_phase=duration_phase,
            window_phase=window_phase,
            baseline=baseline,
        )
        self.control_evaluators = [
            _FixedPhaseEvaluator(
                SecondaryEclipseService._relative_phase(
                    self.time,
                    float(parameters.mid_time),
                    float(parameters.period),
                    (expected_phase + offset) % 1.0,
                ),
                self.time,
                self.uncertainty,
                duration_phase=duration_phase,
                window_phase=window_phase,
                baseline=baseline,
            )
            for offset in (-0.15, 0.15)
        ]
        # A real occultation must recur at the same predicted phase in
        # independent sectors.  The standard aggregate fit cannot encode that
        # distinction by itself, so retain sector-level evaluators for two
        # deliberately simple, physical repeatability features below.
        self.sector_evaluators: list[tuple[np.ndarray, _FixedPhaseEvaluator]] = []
        for sector in sectors:
            sector_indices = original_to_group[sector.indices]
            sector_indices = sector_indices[np.argsort(self.time[sector_indices])]
            self.sector_evaluators.append(
                (
                    sector_indices,
                    _FixedPhaseEvaluator(
                        expected[sector_indices],
                        self.time[sector_indices],
                        self.uncertainty[sector_indices],
                        duration_phase=duration_phase,
                        window_phase=window_phase,
                        baseline=baseline,
                    ),
                )
            )

    def generate(
        self,
        injected_depth_ppm: float,
        rng: np.random.Generator,
        *,
        decoy_depth_ppm: float = 0.0,
        decoy_control_index: int | None = None,
    ) -> np.ndarray:
        """Return a noise-preserving null curve with an optional fake eclipse.

        Each sector's LEAPS baseline and residuals are separated first, so the
        real eclipse is removed.  A circular residual shift preserves local
        correlated-noise structure but makes the injected event independent of
        the original phase pattern.  No source data are edited on disk.
        """
        flux = self.base_flux.copy()
        for local, baseline, residuals, template in self.local_payload:
            shift = int(rng.integers(0, residuals.size))
            flux[local] = baseline + np.roll(residuals, shift) - injected_depth_ppm * 1e-6 * template
        if decoy_depth_ppm > 0.0:
            control_index = (
                int(rng.integers(0, len(self.control_evaluators)))
                if decoy_control_index is None
                else decoy_control_index
            )
            control = self.control_evaluators[control_index]
            # An off-phase dip is a deliberately hard negative: it can be a
            # phase-curve feature or structured reduction artifact, but it is
            # not an occultation at the predicted ephemeris.
            flux[control.local_mask] -= decoy_depth_ppm * 1e-6 * control.template
        return flux

    def features(self, flux: np.ndarray) -> dict[str, float]:
        fit = self.evaluator.evaluate(flux)
        if not fit["coverage"]["available"] or fit["depth"] is None:
            raise RuntimeError("The selected TESS segments do not cover the expected eclipse.")
        controls = [evaluator.evaluate(flux) for evaluator in self.control_evaluators]
        control_significance = max(
            abs(float(control["significance"] or 0.0)) for control in controls if control["coverage"]["available"]
        )
        sector_fits = [
            evaluator.evaluate(flux[indices]) for indices, evaluator in self.sector_evaluators
        ]
        usable_sector_fits = [
            (float(sector_fit["depth"]), float(sector_fit["depth_uncertainty"]))
            for sector_fit in sector_fits
            if sector_fit["coverage"]["available"]
            and sector_fit["depth"] is not None
            and sector_fit["depth_uncertainty"] is not None
            and float(sector_fit["depth_uncertainty"]) > 0.0
        ]
        sector_depths = np.asarray([pair[0] for pair in usable_sector_fits], dtype=float)
        sector_uncertainties = np.asarray([pair[1] for pair in usable_sector_fits], dtype=float)
        # The sector list is assembled only from usable fixed-phase windows,
        # but keep the fallback defensive so a malformed source cannot create
        # a non-finite feature matrix.
        if sector_depths.size == 0:
            positive_sector_fraction = 0.0
            sector_depth_scatter = 0.0
        else:
            positive_sector_fraction = float(np.mean(sector_depths > 0.0))
            if sector_depths.size == 1:
                sector_depth_scatter = 0.0
            else:
                weights = 1.0 / np.square(sector_uncertainties)
                weighted_depth = float(np.sum(weights * sector_depths) / np.sum(weights))
                sector_depth_scatter = float(
                    np.sum(weights * np.square(sector_depths - weighted_depth))
                    / max(1, sector_depths.size - 1)
                )
        return {
            "depth_ppm": float(fit["depth"]) * 1_000_000.0,
            "depth_uncertainty_ppm": float(fit["depth_uncertainty"]) * 1_000_000.0,
            "significance": float(fit["significance"]),
            "red_noise_beta": float(fit["red_noise_beta"]),
            "residual_rms_ppm": float(fit["residual_rms"]) * 1_000_000.0,
            "delta_chi_squared": float(fit["delta_chi_squared"]),
            "control_significance": float(control_significance),
            "positive_sector_fraction": positive_sector_fraction,
            "sector_depth_scatter": sector_depth_scatter,
        }


class SecondaryEclipseMLService:
    """Run an optional, held-out ML injection/recovery validation for TESS data."""

    OUTPUT_NAME = "secondary_eclipse_ml"
    _EXPLORATORY_FALSE_ALARM = 0.05

    @staticmethod
    def sklearn_available() -> bool:
        return importlib.util.find_spec("sklearn") is not None

    @classmethod
    def availability(cls, project: ProjectWorkspace) -> tuple[bool, str]:
        if not cls.sklearn_available():
            return False, "Install the optional ML dependency with: pip install -e '.[ml]'"
        tess = project.manifest.settings.get("tess_import")
        if not isinstance(tess, dict):
            return False, "ML validation is available only for imported TESS PDCSAP light curves."
        sources = [Path(path) for path in tess.get("source_files", [])]
        readable = sum(path.is_file() for path in sources)
        if readable < 4:
            return (
                False,
                "Choose at least four readable TESS light-curve sectors to make a held-out validation split.",
            )
        return True, "Uses held-out TESS sectors; it never changes the LEAPS eclipse decision."

    def run(
        self,
        project: ProjectWorkspace,
        parameters: PlanetParameters,
        *,
        expected_phase: float = 0.5,
        duration_hours: float | None = None,
        light_curve: str = "aperture",
        baseline: str = "linear",
        trials_per_split: int = 240,
        random_seed: int = 20260714,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> MLValidationResult:
        """Build and test a small classifier from LEAPS injection trials.

        ``trials_per_split`` includes equal-ish null and injected examples in
        both the train and held-out test sectors.  240 is intentionally modest:
        this is a reproducible poster-quality validation rather than a large
        black-box search.
        """
        available, message = self.availability(project)
        if not available:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_ML_UNAVAILABLE",
                "ML validation is not ready for this project",
                message,
                ["Import four or more TESS light-curve sectors", "Install the optional ML dependency"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        if trials_per_split < 80:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_ML_TRIALS_TOO_LOW",
                "Use more ML validation trials",
                "At least 80 trials per split are needed to estimate a held-out false-alarm rate.",
                ["Use the recommended trial count"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        token = token or CancellationToken()
        duration_hours = duration_hours or SecondaryEclipseService.estimate_duration_hours(parameters)
        SecondaryEclipseService._validate_inputs(
            parameters, expected_phase, duration_hours, light_curve, baseline
        )

        def check_cancelled() -> None:
            if token.cancelled:
                raise LEAPSError(
                    "JOB_CANCELLED",
                    "ML validation cancelled",
                    "The incomplete validation was discarded. Your normal LEAPS eclipse result was not changed.",
                    ["Run the validation again when ready"],
                    stage=StageID.SECONDARY_ECLIPSE,
                )

        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.RUNNING,
            "Preparing held-out TESS sector split",
            0,
            4,
            checkpoint="ml_prepare",
        )
        time_utc, flux, uncertainty = SecondaryEclipseService()._load_curve(project, light_curve)
        from astropy.time import Time

        time = np.asarray(Time(time_utc, format="jd", scale="utc").tdb.jd, dtype=float)
        duration_phase = float(duration_hours) / (float(parameters.period) * 24.0)
        window_phase = min(0.24, max(0.035, duration_phase * 3.0))
        sector_masks = self._tess_sector_masks(project, time)
        sectors = self._prepare_sector_noise(
            sector_masks,
            time,
            flux,
            uncertainty,
            parameters,
            expected_phase=expected_phase,
            duration_phase=duration_phase,
            window_phase=window_phase,
            baseline=baseline,
        )
        if len(sectors) < 4:
            raise LEAPSError(
                "SECONDARY_ECLIPSE_ML_SEGMENTS_INSUFFICIENT",
                "Not enough usable TESS sectors",
                "LEAPS could not make independent train and test segments with usable eclipse coverage.",
                ["Import more TESS sectors", "Check the primary-transit ephemeris"],
                stage=StageID.SECONDARY_ECLIPSE,
            )
        check_cancelled()
        train_sectors, calibration_sectors, test_sectors = self._split_sectors(sectors)
        train_group = _InjectionGroup(
            train_sectors,
            time,
            flux,
            uncertainty,
            parameters,
            expected_phase=expected_phase,
            duration_phase=duration_phase,
            window_phase=window_phase,
            baseline=baseline,
        )
        calibration_group = _InjectionGroup(
            calibration_sectors,
            time,
            flux,
            uncertainty,
            parameters,
            expected_phase=expected_phase,
            duration_phase=duration_phase,
            window_phase=window_phase,
            baseline=baseline,
        )
        test_group = _InjectionGroup(
            test_sectors,
            time,
            flux,
            uncertainty,
            parameters,
            expected_phase=expected_phase,
            duration_phase=duration_phase,
            window_phase=window_phase,
            baseline=baseline,
        )
        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.RUNNING,
            "Generating LEAPS injection-recovery trials",
            1,
            5,
            checkpoint="ml_injections",
            details={"trials_per_split": trials_per_split},
        )
        rng = np.random.default_rng(random_seed)
        train_rows = self._make_trials(
            train_group,
            "train",
            trials_per_split,
            rng,
            token=token,
            emit=emit,
            stage_progress=(1, 5),
        )
        calibration_rows = self._make_trials(
            calibration_group,
            "calibration",
            max(80, trials_per_split // 2),
            rng,
            token=token,
            emit=emit,
            stage_progress=(2, 5),
        )
        test_rows = self._make_trials(
            test_group,
            "test",
            trials_per_split,
            rng,
            token=token,
            emit=emit,
            stage_progress=(3, 5),
        )
        check_cancelled()
        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.RUNNING,
            "Training and testing the validation classifier",
            4,
            5,
            checkpoint="ml_evaluate",
        )
        result_payload = self._evaluate(
            train_rows,
            calibration_rows,
            test_rows,
            parameters,
            expected_phase=expected_phase,
            duration_hours=duration_hours,
            duration_phase=duration_phase,
            window_phase=window_phase,
            light_curve=light_curve,
            baseline=baseline,
            train_segments=[sector.label for sector in train_sectors],
            calibration_segments=[sector.label for sector in calibration_sectors],
            test_segments=[sector.label for sector in test_sectors],
            random_seed=random_seed,
        )
        check_cancelled()
        pending = project.temporary_dir / "secondary-eclipse-ml-pending"
        target = project.outputs_dir / self.OUTPUT_NAME
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        try:
            self._write_outputs(pending, result_payload, train_rows + calibration_rows + test_rows)
            check_cancelled()
            if target.exists():
                previous = target.with_name(target.name + "-previous")
                if previous.exists():
                    shutil.rmtree(previous)
                target.replace(previous)
                try:
                    pending.replace(target)
                finally:
                    if previous.exists():
                        shutil.rmtree(previous)
            else:
                pending.replace(target)
        except BaseException:
            if pending.exists():
                shutil.rmtree(pending)
            raise
        _emit(
            emit,
            StageID.SECONDARY_ECLIPSE,
            JobStatus.SUCCEEDED,
            "ML validation complete",
            5,
            5,
            checkpoint="ml_complete",
        )
        summary_path = target / "ml-summary.json"
        preview_path = target / "ml-validation.png"
        return MLValidationResult(
            output_path=target,
            preview_path=preview_path,
            summary_path=summary_path,
            message=str(result_payload["message"]),
            recommendation=str(result_payload["recommendation"]),
            test_auc=float(result_payload["metrics"]["test_roc_auc"]),
            test_false_alarm_rate=float(result_payload["metrics"]["test_ml_false_alarm_rate"]),
            calibration_false_alarm_target=float(result_payload["metrics"]["calibration_false_alarm_target"]),
            ml_recovery_50_ppm=self._optional_float(result_payload["metrics"].get("ml_recovery_50_ppm")),
            rule_recovery_50_ppm=self._optional_float(result_payload["metrics"].get("rule_recovery_50_ppm")),
            train_segments=list(result_payload["sector_split"]["train"]),
            test_segments=list(result_payload["sector_split"]["test"]),
            trial_count=len(train_rows) + len(calibration_rows) + len(test_rows),
            raw=result_payload,
        )

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) else None

    @staticmethod
    def _source_bounds(path: Path) -> tuple[str, float, float]:
        from astropy.io import fits

        with fits.open(path, memmap=False) as hdus:
            sector = str(hdus[0].header.get("SECTOR", path.stem))
            data = hdus[1].data
            raw_time = np.asarray(data["TIME"], dtype=float)
            quality = np.asarray(data["QUALITY"], dtype=int) if "QUALITY" in data.names else np.zeros(raw_time.size)
        finite = np.isfinite(raw_time) & (quality == 0)
        if finite.sum() < 10:
            raise ValueError("too few quality-approved timestamps")
        # SPOC light curves use BTJD, i.e. BJD_TDB - 2457000.
        return sector, float(np.min(raw_time[finite]) + 2457000.0), float(np.max(raw_time[finite]) + 2457000.0)

    def _tess_sector_masks(
        self, project: ProjectWorkspace, time: np.ndarray
    ) -> list[tuple[str, np.ndarray]]:
        tess = project.manifest.settings.get("tess_import", {})
        source_files = [Path(path) for path in tess.get("source_files", [])]
        masks: list[tuple[str, np.ndarray]] = []
        for source in source_files:
            if not source.is_file():
                continue
            try:
                label, start, end = self._source_bounds(source)
            except (OSError, KeyError, TypeError, ValueError):
                continue
            mask = (time >= start - 1e-7) & (time <= end + 1e-7)
            if mask.sum() >= 10:
                masks.append((f"Sector {label}", mask))
        if len(masks) >= 4:
            return masks
        # A move of the original FITS files should not make the imported data
        # unusable.  Large gaps in TESS time series are a conservative fallback
        # for independent segments, though the labels make that fact explicit.
        order = np.argsort(time)
        gaps = np.flatnonzero(np.diff(time[order]) > 4.0) + 1
        chunks = np.split(order, gaps)
        fallback = []
        for index, chunk in enumerate(chunks, start=1):
            if chunk.size < 10:
                continue
            mask = np.zeros(time.size, dtype=bool)
            mask[chunk] = True
            fallback.append((f"Time block {index}", mask))
        return fallback

    @staticmethod
    def _prepare_sector_noise(
        masks: list[tuple[str, np.ndarray]],
        time: np.ndarray,
        flux: np.ndarray,
        uncertainty: np.ndarray,
        parameters: PlanetParameters,
        *,
        expected_phase: float,
        duration_phase: float,
        window_phase: float,
        baseline: str,
    ) -> list[_SectorNoise]:
        phase = SecondaryEclipseService._relative_phase(
            time, float(parameters.mid_time), float(parameters.period), expected_phase
        )
        sectors: list[_SectorNoise] = []
        for label, mask in masks:
            indices = np.flatnonzero(mask)
            evaluator = _FixedPhaseEvaluator(
                phase[indices],
                time[indices],
                uncertainty[indices],
                duration_phase=duration_phase,
                window_phase=window_phase,
                baseline=baseline,
            )
            fit = evaluator.evaluate(flux[indices])
            if not fit["coverage"]["available"] or fit["depth"] is None:
                continue
            local_indices = indices[evaluator.local_mask]
            sectors.append(
                _SectorNoise(
                    label=label,
                    indices=indices,
                    local_indices=local_indices,
                    baseline=np.asarray(fit["baseline_model"], dtype=float),
                    residuals=np.asarray(fit["residuals"], dtype=float),
                    template=np.asarray(fit["template"], dtype=float),
                )
            )
        return sectors

    @staticmethod
    def _split_sectors(
        sectors: list[_SectorNoise],
    ) -> tuple[list[_SectorNoise], list[_SectorNoise], list[_SectorNoise]]:
        """Return disjoint model-training, threshold-calibration, and test sectors.

        The score threshold must not be tuned on the same sector residuals that
        trained the forest.  Alternating sectors also distribute early and late
        TESS visits across the splits instead of making a convenient but weak
        chronological split.
        """
        alternating = sectors[::2]
        held_out = sectors[1::2]
        if len(sectors) >= 6 and len(alternating) >= 3 and len(held_out) >= 2:
            return alternating[:-1], alternating[-1:], held_out
        # Four- and five-sector projects can still run a small proof-of-concept
        # validation, but deliberately reserve one full sector for calibration
        # and one for the final test.
        training = sectors[: max(1, len(sectors) - 2)]
        calibration = sectors[max(1, len(sectors) - 2) : -1]
        testing = sectors[-1:]
        return training, calibration, testing

    def _make_trials(
        self,
        group: _InjectionGroup,
        split: str,
        count: int,
        rng: np.random.Generator,
        *,
        token: CancellationToken,
        emit: Emitter | None,
        stage_progress: tuple[int, int],
    ) -> list[dict[str, Any]]:
        # Half the rows are non-eclipse examples.  Some are clean nulls, while
        # the rest contain a deliberately off-phase dip.  Those hard negatives
        # make the experiment answer more than "is a large depth detectable?":
        # they test whether a predicted-phase eclipse can be separated from a
        # structured dip elsewhere in the orbit.
        # Balancing the labels matters: a classifier trained on mostly injected
        # rows would look deceptively good by simply predicting "eclipse".
        null_count = count // 2
        positive = np.resize(np.asarray(DEFAULT_DEPTHS_PPM, dtype=float), count - null_count)
        clean_null_count = null_count // 2
        decoy_count = null_count - clean_null_count
        specifications: list[tuple[float, str, float]] = [
            (0.0, "clean_null", 0.0) for _ in range(clean_null_count)
        ]
        specifications.extend(
            (0.0, "off_phase_decoy", float(depth))
            for depth in np.resize(np.asarray((75.0, 150.0, 300.0), dtype=float), decoy_count)
        )
        specifications.extend((float(depth), "expected_eclipse", 0.0) for depth in positive)
        rng.shuffle(specifications)
        rows: list[dict[str, Any]] = []
        start, total = stage_progress
        for index, (depth, example_type, decoy_depth) in enumerate(specifications, start=1):
            if token.cancelled:
                raise LEAPSError(
                    "JOB_CANCELLED",
                    "ML validation cancelled",
                    "The incomplete validation was discarded. Your normal LEAPS eclipse result was not changed.",
                    ["Run the validation again when ready"],
                    stage=StageID.SECONDARY_ECLIPSE,
                )
            features = group.features(
                group.generate(float(depth), rng, decoy_depth_ppm=decoy_depth)
            )
            rule_candidate = self._leaps_candidate_rule(features)
            rows.append(
                {
                    "split": split,
                    "injected_depth_ppm": float(depth),
                    "label_injected_eclipse": int(depth > 0.0),
                    "example_type": example_type,
                    "off_phase_decoy_depth_ppm": decoy_depth,
                    "leaps_candidate_rule": int(rule_candidate),
                    **features,
                }
            )
            if index == count or index % max(10, count // 8) == 0:
                fraction = index / count
                _emit(
                    emit,
                    StageID.SECONDARY_ECLIPSE,
                    JobStatus.RUNNING,
                    f"Running {split} injection trial {index} of {count}",
                    start + fraction,
                    total,
                    checkpoint="ml_injections",
                )
        return rows

    @staticmethod
    def _leaps_candidate_rule(features: dict[str, Any]) -> bool:
        """Mirror ``SecondaryEclipseService._classify`` for a usable fit."""
        significance = float(features["significance"])
        return bool(significance >= 5.0 and SecondaryEclipseMLService._leaps_control_guard(features))

    @staticmethod
    def _leaps_control_guard(features: dict[str, Any]) -> bool:
        """Retain LEAPS' positive-depth and nearby-control safety condition."""
        depth = float(features["depth_ppm"])
        significance = float(features["significance"])
        control = abs(float(features["control_significance"]))
        return bool(depth > 0.0 and control < max(3.0, significance - 1.5))

    def _evaluate(
        self,
        train_rows: list[dict[str, Any]],
        calibration_rows: list[dict[str, Any]],
        test_rows: list[dict[str, Any]],
        parameters: PlanetParameters,
        *,
        expected_phase: float,
        duration_hours: float,
        duration_phase: float,
        window_phase: float,
        light_curve: str,
        baseline: str,
        train_segments: list[str],
        calibration_segments: list[str],
        test_segments: list[str],
        random_seed: int,
    ) -> dict[str, Any]:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score

        def matrix(rows: list[dict[str, Any]]) -> np.ndarray:
            return np.asarray([[float(row[key]) for key in FEATURE_KEYS] for row in rows], dtype=float)

        train_x = matrix(train_rows)
        calibration_x = matrix(calibration_rows)
        test_x = matrix(test_rows)
        train_y = np.asarray([int(row["label_injected_eclipse"]) for row in train_rows], dtype=int)
        calibration_y = np.asarray(
            [int(row["label_injected_eclipse"]) for row in calibration_rows], dtype=int
        )
        test_y = np.asarray([int(row["label_injected_eclipse"]) for row in test_rows], dtype=int)
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features=0.85,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )
        classifier.fit(train_x, train_y)
        calibration_scores = classifier.predict_proba(calibration_x)[:, 1]
        calibration_null = calibration_scores[calibration_y == 0]
        if calibration_null.size < 10:
            raise RuntimeError("The calibration split did not contain enough null trials.")
        exploratory_threshold = float(
            np.quantile(calibration_null, 1.0 - self._EXPLORATORY_FALSE_ALARM, method="higher")
        )
        # This is the operating point used for the headline comparison to the
        # fixed LEAPS candidate rule.  It has no calibration false positives,
        # so it cannot claim a lower recovery floor merely by tolerating more
        # false alarms than the physics-first rule.
        conservative_threshold = float(np.max(calibration_null))
        train_scores = classifier.predict_proba(train_x)[:, 1]
        test_scores = classifier.predict_proba(test_x)[:, 1]
        for row, score in zip(train_rows, train_scores, strict=True):
            row["ml_score"] = float(score)
            row["ml_candidate_at_5pct_threshold"] = int(score >= exploratory_threshold)
            row["ml_safety_guard_passed"] = int(self._leaps_control_guard(row))
            row["ml_candidate_at_conservative_threshold"] = int(
                score > conservative_threshold and self._leaps_control_guard(row)
            )
        for row, score in zip(calibration_rows, calibration_scores, strict=True):
            row["ml_score"] = float(score)
            row["ml_candidate_at_5pct_threshold"] = int(score >= exploratory_threshold)
            row["ml_safety_guard_passed"] = int(self._leaps_control_guard(row))
            row["ml_candidate_at_conservative_threshold"] = int(
                score > conservative_threshold and self._leaps_control_guard(row)
            )
        for row, score in zip(test_rows, test_scores, strict=True):
            row["ml_score"] = float(score)
            row["ml_candidate_at_5pct_threshold"] = int(score >= exploratory_threshold)
            row["ml_safety_guard_passed"] = int(self._leaps_control_guard(row))
            row["ml_candidate_at_conservative_threshold"] = int(
                score > conservative_threshold and self._leaps_control_guard(row)
            )
        test_null = test_y == 0
        test_positive = test_y == 1
        test_exploratory_fpr = (
            float(np.mean(test_scores[test_null] >= exploratory_threshold)) if test_null.any() else math.nan
        )
        test_raw_conservative_fpr = (
            float(np.mean(test_scores[test_null] > conservative_threshold)) if test_null.any() else math.nan
        )
        test_fpr = (
            float(
                np.mean(
                    np.asarray(
                        [row["ml_candidate_at_conservative_threshold"] for row in test_rows], dtype=bool
                    )[test_null]
                )
            )
            if test_null.any()
            else math.nan
        )
        test_rule_fpr = float(
            np.mean(
                np.asarray([row["leaps_candidate_rule"] for row in test_rows], dtype=bool)[test_null]
            )
        )
        test_auc = float(roc_auc_score(test_y, test_scores))
        curve = self._recovery_curve(test_rows)
        ml_50 = self._recovery_floor(curve, "ml_recovery")
        rule_50 = self._recovery_floor(curve, "rule_recovery")
        improvement = (
            ml_50 is not None
            and rule_50 is not None
            and ml_50 + 10.0 < rule_50
            and test_fpr <= test_rule_fpr
            and test_auc >= 0.75
        )
        if improvement:
            recommendation = (
                "Promising as a LEAPS triage aid: its score plus the non-negotiable LEAPS nearby-control guard "
                "recovered held-out injected eclipses at a lower 50% depth than the fixed rule with no held-out "
                "false alarms. Keep the normal LEAPS decision and all physics checks as the authority."
            )
        else:
            recommendation = (
                "This validation did not show a robust enough independent advantage over LEAPS' fixed-phase rule. "
                "Keep the ML result as a negative-method result, not as a detection tool."
            )
        message = (
            f"Held-out-sector ROC-AUC {test_auc:.2f}; conservative ML false-alarm rate {test_fpr:.1%} "
            f"after the LEAPS nearby-control guard. {recommendation}"
        )
        return {
            "analysis": "LEAPS secondary-eclipse ML injection/recovery validation",
            "version": 1,
            "planet": parameters.name,
            "parameters": asdict(parameters),
            "configuration": {
                "expected_phase": expected_phase,
                "duration_hours": duration_hours,
                "duration_phase": duration_phase,
                "window_phase": window_phase,
                "light_curve": light_curve,
                "baseline": baseline,
                "random_seed": random_seed,
                "features": list(FEATURE_NAMES),
                "feature_design": (
                    "Aggregate LEAPS fit metrics plus sector-repeatability features: an astrophysical "
                    "eclipse should have positive, mutually consistent fixed-phase depths in independent sectors."
                ),
                "classifier": "RandomForestClassifier (300 trees, min_samples_leaf=3)",
            },
            "sector_split": {
                "train": train_segments,
                "calibration": calibration_segments,
                "test": test_segments,
            },
            "metrics": {
                "test_roc_auc": test_auc,
                "calibration_false_alarm_target": 0.0,
                "exploratory_calibration_false_alarm_target": self._EXPLORATORY_FALSE_ALARM,
                "test_ml_false_alarm_rate": test_fpr,
                "test_ml_at_5pct_false_alarm_rate": test_exploratory_fpr,
                "test_ml_raw_score_false_alarm_rate": test_raw_conservative_fpr,
                "test_leaps_rule_false_alarm_rate": test_rule_fpr,
                "ml_probability_threshold": conservative_threshold,
                "ml_5pct_probability_threshold": exploratory_threshold,
                "ml_recovery_50_ppm": ml_50,
                "rule_recovery_50_ppm": rule_50,
                "test_positive_trials": int(test_positive.sum()),
                "test_null_trials": int(test_null.sum()),
                "training_trials": int(train_y.size),
                "calibration_trials": int(calibration_y.size),
                "feature_importance": {
                    name: float(value)
                    for name, value in zip(FEATURE_NAMES, classifier.feature_importances_, strict=True)
                },
            },
            "recovery_curve": curve,
            "message": message,
            "recommendation": recommendation,
            "scientific_scope": [
                "Training labels are synthetic eclipses injected into LEAPS-cleaned real TESS residuals; no hand labels are used.",
                "Negative examples include clean nulls and deliberately off-phase dips, so a strong nearby control phase is represented as structured noise rather than a real occultation.",
                "Training, threshold-calibration, and test sectors are disjoint. This is a held-out sector test, not a claim of universal performance across planets.",
                "The two sector-repeatability features are physical consistency checks: the fraction of independent sectors with a positive fitted depth and the uncertainty-weighted scatter of their fitted depths. They are not measurements of an eclipse by themselves.",
                "A classifier score is acted on only after the positive-depth and nearby-control safety guard already used by LEAPS; it never changes the normal LEAPS secondary-eclipse outcome or turns a marginal signal into a confirmation.",
                "Use the fixed-phase fit, nearby controls, independent sectors, and injection-recovery curve as the scientific evidence.",
            ],
        }

    @staticmethod
    def _recovery_curve(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
        output: list[dict[str, float | int]] = []
        depths = sorted({float(row["injected_depth_ppm"]) for row in rows})
        for depth in depths:
            selected = [row for row in rows if float(row["injected_depth_ppm"]) == depth]
            output.append(
                {
                    "injected_depth_ppm": depth,
                    "trials": len(selected),
                    "ml_recovery": float(
                        np.mean([int(row["ml_candidate_at_conservative_threshold"]) for row in selected])
                    ),
                    "rule_recovery": float(
                        np.mean([int(row["leaps_candidate_rule"]) for row in selected])
                    ),
                }
            )
        return output

    @staticmethod
    def _recovery_floor(curve: list[dict[str, float | int]], key: str) -> float | None:
        positive = [entry for entry in curve if float(entry["injected_depth_ppm"]) > 0.0]
        positive.sort(key=lambda entry: float(entry["injected_depth_ppm"]))
        previous: dict[str, float | int] | None = None
        for entry in positive:
            if float(entry[key]) >= 0.5:
                if previous is None or float(previous[key]) >= 0.5:
                    return float(entry["injected_depth_ppm"])
                x0, y0 = float(previous["injected_depth_ppm"]), float(previous[key])
                x1, y1 = float(entry["injected_depth_ppm"]), float(entry[key])
                if y1 <= y0:
                    return x1
                return x0 + (0.5 - y0) * (x1 - x0) / (y1 - y0)
            previous = entry
        return None

    @staticmethod
    def _write_outputs(destination: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "ml-summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        fieldnames = [
            "split",
            "injected_depth_ppm",
            "label_injected_eclipse",
            "example_type",
            "off_phase_decoy_depth_ppm",
            "leaps_candidate_rule",
            *FEATURE_KEYS,
            "ml_score",
            "ml_candidate_at_5pct_threshold",
            "ml_safety_guard_passed",
            "ml_candidate_at_conservative_threshold",
        ]
        with (destination / "ml-trials.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        SecondaryEclipseMLService._write_figure(destination / "ml-validation.png", summary)
        (destination / "README.md").write_text(
            "# LEAPS secondary-eclipse ML validation\n\n"
            "This folder is an **optional injection/recovery validation**, not a new eclipse detection. "
            "LEAPS removed the fitted real eclipse from each sector's local window, circularly shifted the "
            "real residuals, injected known fake eclipses (plus off-phase decoy dips for structured-noise negatives), "
            "and ran the same fixed-phase/red-noise-aware model. "
            "The classifier was trained on one set of sectors, its threshold was calibrated on a different sector, "
            "and it was evaluated only on held-out sectors.\n\n"
            "`ml-summary.json` records the configuration, held-out metrics, recovery curve, and scientific caveats. "
            "`ml-trials.csv` is the row-level, reproducible trial table. `ml-validation.png` is the poster-ready summary.\n\n"
            "A score is only acted on after LEAPS' positive-depth and nearby-control safety guard. "
            "The classifier never changes the standard LEAPS candidate/marginal/inconclusive outcome. "
            "Do not use an ML score alone as evidence for a secondary eclipse.\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_figure(destination: Path, summary: dict[str, Any]) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        curve = summary["recovery_curve"]
        depths = np.asarray([float(entry["injected_depth_ppm"]) for entry in curve], dtype=float)
        ml = np.asarray([float(entry["ml_recovery"]) for entry in curve], dtype=float)
        rule = np.asarray([float(entry["rule_recovery"]) for entry in curve], dtype=float)
        importance = summary["metrics"]["feature_importance"]
        labels = list(importance)
        values = np.asarray([float(importance[label]) for label in labels], dtype=float)
        order = np.argsort(values)
        figure = Figure(figsize=(12.0, 6.9), facecolor="#0b2638", constrained_layout=True)
        FigureCanvasAgg(figure)
        grid = figure.add_gridspec(2, 2, height_ratios=(0.28, 1.0), width_ratios=(1.28, 1.0))
        header = figure.add_subplot(grid[0, :])
        header.axis("off")
        header.text(
            0.0,
            0.85,
            f"{summary['planet']}  ·  LEAPS held-out ML validation",
            color="#f7fbff",
            fontsize=20,
            fontweight="bold",
            va="top",
        )
        metrics = summary["metrics"]
        def compact(labels: list[str]) -> str:
            return ", ".join(label.replace("Sector ", "S") for label in labels)

        header.text(
            0.0,
            0.26,
            "Train: "
            + compact(summary["sector_split"]["train"])
            + "    |    Calibration: "
            + compact(summary["sector_split"].get("calibration", []))
            + "    |    Held out: "
            + compact(summary["sector_split"]["test"])
            + f"\nROC-AUC {metrics['test_roc_auc']:.2f}  ·  conservative ML false alarms {metrics['test_ml_false_alarm_rate']:.1%}"
            + "  ·  zero calibration false alarms",
            color="#b8c8d6",
            fontsize=10.5,
            va="top",
        )
        recovery_axis = figure.add_subplot(grid[1, 0])
        recovery_axis.set_facecolor("#102f43")
        recovery_axis.plot(depths, rule, "o-", color="#f1bd50", lw=2.0, label="LEAPS fixed rule")
        recovery_axis.plot(
            depths,
            ml,
            "o-",
            color="#25c2c7",
            lw=2.6,
            label="ML + LEAPS safety guard",
        )
        recovery_axis.axhline(0.5, color="#93a6b4", ls="--", lw=1.0)
        for key, color, label in (
            ("rule_recovery_50_ppm", "#f1bd50", "Rule 50%"),
            ("ml_recovery_50_ppm", "#25c2c7", "ML 50%"),
        ):
            value = SecondaryEclipseMLService._optional_float(metrics.get(key))
            if value is not None:
                recovery_axis.axvline(value, color=color, ls=":", lw=1.3)
                recovery_axis.text(value, 0.08, f"{label}: {value:.0f} ppm", color=color, rotation=90, va="bottom", ha="right", fontsize=8.5)
        recovery_axis.set_title("Injected eclipse recovery", color="#f7fbff", loc="left", fontweight="bold")
        recovery_axis.set_xlabel("Injected depth (ppm)", color="#d6e2ea")
        recovery_axis.set_ylabel("Recovered as candidate", color="#d6e2ea")
        recovery_axis.set_ylim(-0.04, 1.05)
        recovery_axis.tick_params(colors="#c7d5df")
        for spine in recovery_axis.spines.values():
            spine.set_color("#49667a")
        legend = recovery_axis.legend(frameon=False, loc="lower right")
        for text in legend.get_texts():
            text.set_color("#e6f0f6")

        feature_axis = figure.add_subplot(grid[1, 1])
        feature_axis.set_facecolor("#102f43")
        feature_axis.barh(np.arange(values.size), values[order], color="#4d9dcc")
        feature_axis.set_yticks(np.arange(values.size), [labels[index] for index in order])
        feature_axis.set_xlabel("Random-forest importance", color="#d6e2ea")
        feature_axis.set_title("What the classifier used", color="#f7fbff", loc="left", fontweight="bold")
        feature_axis.tick_params(colors="#c7d5df", labelsize=8.5)
        for spine in feature_axis.spines.values():
            spine.set_color("#49667a")
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=180, facecolor=figure.get_facecolor())
