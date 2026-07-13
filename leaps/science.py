from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import PlanetParameters
from .models import JobStatus, LEAPSError, StageEvent, StageID
from .project import ProjectWorkspace

Emitter = Callable[[StageEvent], None]


class CancellationToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise LEAPSError(
                "JOB_CANCELLED",
                "Processing was safely cancelled",
                "Verified checkpoints were kept. Resume or restart this stage when ready.",
                ["Resume", "Restart stage"],
            )


@dataclass(slots=True)
class ReductionConfig:
    exposure_key: str = "EXPTIME"
    date_key: str = "DATE-OBS"
    time_key: str = "TIME-OBS"
    filter_name: str = "R"
    combine_method: str = "median"
    binning: int = 1
    crop: tuple[int, int, int, int] | None = None


@dataclass(slots=True)
class InspectionResult:
    frames: list[dict[str, Any]]
    median_sky: float
    median_psf: float


@dataclass(slots=True)
class PlateSolveAttempt:
    index: int
    pixel_scale: float
    status: str
    detail: str


@dataclass(slots=True)
class PlateSolveResult:
    solved: bool
    attempts: list[PlateSolveAttempt]
    target_xy: tuple[float, float] | None = None
    identified_stars: int = 0
    wcs_header: dict[str, Any] = field(default_factory=dict)
    unverified: bool = False


def _emit(
    emit: Emitter | None,
    stage: StageID,
    status: JobStatus,
    message: str,
    current: int = 0,
    total: int = 0,
    checkpoint: str | None = None,
) -> None:
    if emit:
        emit(StageEvent(stage, status, message, current, total, checkpoint))


class ReductionService:
    def run(
        self,
        project: ProjectWorkspace,
        config: ReductionConfig,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Path:
        from astropy.io import fits

        token = token or CancellationToken()
        files = project.manifest.raw_files
        science = [project.resolve(path) for path in files.get("science", [])]
        if not science:
            raise LEAPSError(
                "NO_SCIENCE_FRAMES",
                "No science frames are assigned",
                "Return to Data & Target and confirm the FITS frame assignments.",
                ["Review frame assignments"],
                stage=StageID.REDUCTION,
            )
        pending, target = project.begin_transaction(StageID.REDUCTION)
        _emit(emit, StageID.REDUCTION, JobStatus.RUNNING, "Building calibration frames", 0, len(science))
        master_bias, bias_exposure = self._master_bias(project, files.get("bias", []), config)
        master_dark = self._master_dark(project, files.get("dark", []), config, master_bias, bias_exposure)
        master_dark_flat = self._master_dark(
            project, files.get("dark_flat", []), config, master_bias, bias_exposure, fallback=master_dark
        )
        master_flat = self._master_flat(
            project, files.get("flat", []), config, master_bias, master_dark_flat, bias_exposure
        )
        metadata: list[dict[str, Any]] = []
        for index, path in enumerate(science, start=1):
            token.raise_if_cancelled()
            try:
                with fits.open(path, memmap=True, ignore_missing_end=True) as hdus:
                    hdu = next(
                        candidate for candidate in hdus if getattr(candidate, "data", None) is not None
                    )
                    data = np.asarray(hdu.data, dtype=np.float32)
                    header = hdu.header.copy()
                exposure = float(header.get(config.exposure_key, 0.0))
                reduced = (
                    data - master_bias - max(0.0, exposure - bias_exposure) * master_dark
                ) / master_flat
                reduced[~np.isfinite(reduced)] = 0
                if config.crop:
                    x1, x2, y1, y2 = config.crop
                    reduced = reduced[y1 : y2 or None, x1 : x2 or None]
                if config.binning > 1:
                    from hops.hops_tools.image_analysis import bin_frame

                    reduced = bin_frame(reduced, config.binning)
                mean, std, psf = self._statistics(reduced, header)
                output_name = f"r_{index:05d}_{path.name}"
                output = pending / output_name
                output_header = header.copy()
                output_header["LEAPSVER"] = "0.1.0"
                output_header["HOPSJD"] = _julian_date(output_header, config)
                output_header["HOPSMEAN"] = mean
                output_header["HOPSSTD"] = std
                output_header["HOPSPSF"] = psf
                output_header["HOPSSKIP"] = bool(not np.isfinite(psf))
                output_header["HOPSFLT"] = config.filter_name
                fits.PrimaryHDU(reduced.astype(np.float32), header=output_header).writeto(
                    output, overwrite=True
                )
                metadata.append(
                    {
                        "file": output_name,
                        "source": project.relative(path),
                        "mean": mean,
                        "std": std,
                        "psf": psf,
                        "exposure": exposure,
                        "skip": bool(not np.isfinite(psf)),
                    }
                )
                checkpoint = project.checkpoints_dir / "reduction.json"
                checkpoint.write_text(
                    json.dumps({"completed": index, "files": metadata}, indent=2), encoding="utf-8"
                )
                _emit(
                    emit,
                    StageID.REDUCTION,
                    JobStatus.RUNNING,
                    f"Reduced {path.name}",
                    index,
                    len(science),
                    project.relative(checkpoint),
                )
            except LEAPSError:
                raise
            except Exception as exc:
                raise LEAPSError(
                    "REDUCTION_FRAME_FAILED",
                    f"{path.name} could not be reduced",
                    "The last successful reduction remains available and the source FITS file was not modified.",
                    ["Inspect the FITS header", "Exclude this frame", "Export diagnostics"],
                    stage=StageID.REDUCTION,
                    technical_details=str(exc),
                ) from exc
        (pending / "frames.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        _emit(emit, StageID.REDUCTION, JobStatus.SUCCEEDED, "Reduction complete", len(science), len(science))
        return target

    @staticmethod
    def _load(project: ProjectWorkspace, paths: Iterable[str]) -> list[tuple[np.ndarray, dict[str, Any]]]:
        from astropy.io import fits

        result: list[tuple[np.ndarray, dict[str, Any]]] = []
        for relative in paths:
            with fits.open(project.resolve(relative), memmap=True, ignore_missing_end=True) as hdus:
                hdu = next(candidate for candidate in hdus if getattr(candidate, "data", None) is not None)
                result.append((np.asarray(hdu.data, dtype=np.float32), dict(hdu.header)))
        return result

    def _master_bias(
        self, project: ProjectWorkspace, paths: list[str], config: ReductionConfig
    ) -> tuple[np.ndarray | float, float]:
        frames = self._load(project, paths)
        if not frames:
            return 0.0, 0.0
        exposures = np.array([float(header.get(config.exposure_key, 0.0)) for _, header in frames])
        median_exposure = float(np.median(exposures))
        arrays = [array for (array, _), use in zip(frames, np.isclose(exposures, median_exposure)) if use]
        return _combine(arrays, config.combine_method), median_exposure

    def _master_dark(
        self,
        project: ProjectWorkspace,
        paths: list[str],
        config: ReductionConfig,
        master_bias: np.ndarray | float,
        bias_exposure: float,
        fallback: np.ndarray | float = 0.0,
    ) -> np.ndarray | float:
        frames = self._load(project, paths)
        if not frames:
            return fallback
        corrected = [
            (array - master_bias) / max(float(header.get(config.exposure_key, 0.0)) - bias_exposure, 1e-9)
            for array, header in frames
        ]
        return _combine(corrected, config.combine_method)

    def _master_flat(
        self,
        project: ProjectWorkspace,
        paths: list[str],
        config: ReductionConfig,
        master_bias: np.ndarray | float,
        master_dark_flat: np.ndarray | float,
        bias_exposure: float,
    ) -> np.ndarray | float:
        frames = self._load(project, paths)
        if not frames:
            return 1.0
        corrected = []
        for array, header in frames:
            exposure = max(float(header.get(config.exposure_key, 0.0)) - bias_exposure, 0.0)
            flat = array - master_bias - exposure * master_dark_flat
            median = float(np.nanmedian(flat))
            if not math.isfinite(median) or median == 0:
                continue
            corrected.append(flat / median)
        if not corrected:
            return 1.0
        master = _combine(corrected, config.combine_method)
        master = np.where(np.isfinite(master) & (master > 0), master, 1.0)
        return master / np.nanmedian(master)

    @staticmethod
    def _statistics(data: np.ndarray, header: Any) -> tuple[float, float, float]:
        from hops.hops_tools.image_analysis import image_mean_std, image_psf

        mean, std = image_mean_std(data)
        saturation = float(header.get("SATURATE", np.nanmax(data)))
        psf = image_psf(data, header, mean, std, saturation)
        return float(mean), float(std), float(psf)


class InspectionService:
    def run(
        self,
        project: ProjectWorkspace,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> InspectionResult:
        token = token or CancellationToken()
        reduction = project.outputs_dir / StageID.REDUCTION.value
        frames = sorted(reduction.glob("*.fit*"))
        if not frames:
            raise LEAPSError(
                "NO_REDUCED_FRAMES",
                "No reduced frames are available",
                "Run Reduction before Inspection.",
                ["Open Reduction"],
                stage=StageID.INSPECTION,
            )
        from astropy.io import fits

        values: list[dict[str, Any]] = []
        for index, path in enumerate(frames, start=1):
            token.raise_if_cancelled()
            header = fits.getheader(path)
            values.append(
                {
                    "file": path.name,
                    "sky": float(header.get("HOPSMEAN", 0.0)),
                    "sky_std": float(header.get("HOPSSTD", 0.0)),
                    "psf": float(header.get("HOPSPSF", float("nan"))),
                    "excluded": bool(header.get("HOPSSKIP", False)),
                }
            )
            _emit(emit, StageID.INSPECTION, JobStatus.RUNNING, f"Checked {path.name}", index, len(frames))
        skies = np.array([record["sky"] for record in values], dtype=float)
        psfs = np.array([record["psf"] for record in values], dtype=float)
        sky_median, psf_median = float(np.nanmedian(skies)), float(np.nanmedian(psfs))
        sky_mad = max(float(np.nanmedian(np.abs(skies - sky_median))), 1e-9)
        psf_mad = max(float(np.nanmedian(np.abs(psfs - psf_median))), 1e-9)
        for record in values:
            if abs(record["sky"] - sky_median) > 5 * sky_mad or abs(record["psf"] - psf_median) > 5 * psf_mad:
                record["suggest_exclude"] = True
            else:
                record["suggest_exclude"] = False
        pending, target = project.begin_transaction(StageID.INSPECTION)
        result = InspectionResult(values, sky_median, psf_median)
        (pending / "inspection.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        _emit(emit, StageID.INSPECTION, JobStatus.SUCCEEDED, "Inspection complete", len(frames), len(frames))
        return result


class AlignmentService:
    def run(
        self,
        project: ProjectWorkspace,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Path:
        token = token or CancellationToken()
        from astropy.io import fits

        from hops.hops_tools.image_analysis import image_find_stars
        from hops.thirdparty import twirl

        frames = sorted((project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        if len(frames) < 2:
            raise LEAPSError(
                "ALIGNMENT_INPUT_MISSING",
                "Alignment needs at least two reduced frames",
                "Run Reduction first.",
                ["Open Reduction"],
                stage=StageID.ALIGNMENT,
            )
        reference_data, reference_header = fits.getdata(frames[0], header=True)
        detected_reference = image_find_stars(reference_data, reference_header, star_limit=60) or []
        reference_stars = np.asarray(detected_reference, dtype=float)
        if reference_stars.size:
            reference_stars = reference_stars[:, :2]
        if len(reference_stars) < 5:
            raise LEAPSError(
                "TOO_FEW_ALIGNMENT_STARS",
                "Too few stars were found for alignment",
                "Try a lower star-detection threshold or inspect the first frame.",
                ["Review the first frame", "Adjust advanced alignment settings"],
                stage=StageID.ALIGNMENT,
            )
        records = []
        for index, path in enumerate(frames, start=1):
            token.raise_if_cancelled()
            try:
                data, header = fits.getdata(path, header=True)
                detected = image_find_stars(data, header, star_limit=60) or []
                stars = np.asarray(detected, dtype=float)
                if not stars.size:
                    raise ValueError("No alignment stars were detected")
                stars = stars[:, :2]
                count = min(20, len(reference_stars), len(stars))
                transform = twirl.utils.find_transform(
                    reference_stars[:count], stars[:count], n=count, tolerance=12
                )
                matrix = np.asarray(transform)
                rotation = float(math.atan2(matrix[1, 0], matrix[0, 0]))
                x0, y0 = float(matrix[0, 2]), float(matrix[1, 2])
                header["HOPSX0"] = x0
                header["HOPSY0"] = y0
                header["HOPSU0"] = rotation
                fits.writeto(path, data, header, overwrite=True)
                records.append(
                    {"file": path.name, "x0": x0, "y0": y0, "rotation": rotation, "matched": count}
                )
            except Exception as exc:
                records.append({"file": path.name, "failed": True, "reason": str(exc)})
            _emit(emit, StageID.ALIGNMENT, JobStatus.RUNNING, f"Aligned {path.name}", index, len(frames))
        pending, target = project.begin_transaction(StageID.ALIGNMENT)
        (pending / "alignment.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        _emit(emit, StageID.ALIGNMENT, JobStatus.SUCCEEDED, "Alignment complete", len(frames), len(frames))
        return target


class PlateSolveService:
    def solve(
        self,
        frame: str | Path,
        ra: str,
        dec: str,
        pixel_scale: float,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> PlateSolveResult:
        token = token or CancellationToken()
        import astropy.units as units
        from astropy.coordinates import SkyCoord
        from astropy.io import fits
        from astropy.time import Time

        from hops.hops_tools.image_analysis import image_find_stars, image_plate_solve

        coordinate = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
        _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, "Coordinates validated", 0, 3)
        data, header = fits.getdata(frame, header=True)
        stars = image_find_stars(data, header, star_limit=100) or []
        if len(stars) < 5:
            raise LEAPSError(
                "TOO_FEW_PLATE_STARS",
                "Too few stars were detected",
                "Plate solving needs at least five usable stars.",
                ["Adjust contrast and detection threshold", "Choose another frame"],
                stage=StageID.PHOTOMETRY,
            )
        _emit(
            emit,
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            f"{len(stars)} stars detected",
            0,
            3,
        )
        attempts: list[PlateSolveAttempt] = []
        for index, scale in enumerate((pixel_scale, pixel_scale * 0.5, pixel_scale * 2.0), start=1):
            token.raise_if_cancelled()
            try:
                timestamp = Time(header.get("DATE-OBS", Time.now().isot))
                solution = image_plate_solve(
                    data,
                    header,
                    coordinate.ra.deg,
                    coordinate.dec.deg,
                    timestamp,
                    stars=stars,
                    pixel=scale,
                    verbose=False,
                )
                identified = len(solution["identified_stars"])
                if identified < max(5, int(0.25 * len(stars))):
                    raise ValueError(f"Only {identified} of {len(stars)} detected stars matched")
                x, y = coordinate.to_pixel(solution["plate_solution"])
                attempts.append(PlateSolveAttempt(index, scale, "complete", f"{identified} stars matched"))
                _emit(emit, StageID.PHOTOMETRY, JobStatus.SUCCEEDED, "Plate solution found", index, 3)
                return PlateSolveResult(
                    True,
                    attempts,
                    (float(x), float(y)),
                    identified,
                    dict(solution["plate_solution"].to_header()),
                )
            except Exception as exc:
                attempts.append(PlateSolveAttempt(index, scale, "failed", str(exc)))
                _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, f"Solve attempt {index} failed", index, 3)
        details = "\n".join(f"Attempt {item.index}: {item.detail}" for item in attempts)
        raise LEAPSError(
            "PLATE_SOLVE_FAILED",
            "Plate solve needs attention",
            "The image and detected stars are safe. LEAPS stopped after three bounded attempts.",
            ["Retry plate solve", "Place the target manually and continue with an unverified WCS"],
            stage=StageID.PHOTOMETRY,
            technical_details=details,
        )

    @staticmethod
    def manual(target_xy: tuple[float, float]) -> PlateSolveResult:
        return PlateSolveResult(False, [], target_xy=target_xy, unverified=True)


class PhotometryService:
    def rank_comparisons(
        self, frame: str | Path, target_xy: tuple[float, float], limit: int = 10
    ) -> list[dict[str, float]]:
        from astropy.io import fits

        from hops.hops_tools.image_analysis import image_find_stars

        data, header = fits.getdata(frame, header=True)
        stars = np.asarray(image_find_stars(data, header, star_limit=150) or [])
        if stars.size == 0:
            return []
        tx, ty = target_xy
        ranked = []
        for star in stars:
            x, y, peak = map(float, star[:3])
            distance = math.hypot(x - tx, y - ty)
            if distance < max(10.0, float(header.get("HOPSPSF", 3)) * 5):
                continue
            saturation = float(header.get("HOPSSAT", np.nanmax(data)))
            score = min(peak / max(saturation, 1), 0.95) - 0.05 * abs(math.log10(max(peak, 1)))
            ranked.append({"x": x, "y": y, "peak": peak, "distance": distance, "score": score})
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

    def run(
        self,
        project: ProjectWorkspace,
        target_xy: tuple[float, float],
        comparisons: list[tuple[float, float]],
        aperture_radius: float,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> Path:
        token = token or CancellationToken()
        from astropy.io import fits
        from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry

        frames = sorted((project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        positions = [target_xy, *comparisons]
        if len(positions) < 2:
            raise LEAPSError(
                "COMPARISON_STARS_REQUIRED",
                "Choose at least one comparison star",
                "Differential photometry needs the target and one or more comparison stars.",
                ["Review suggested comparison stars"],
                stage=StageID.PHOTOMETRY,
            )
        rows = []
        for index, path in enumerate(frames, start=1):
            token.raise_if_cancelled()
            data, header = fits.getdata(path, header=True)
            apertures = CircularAperture(positions, r=aperture_radius)
            annuli = CircularAnnulus(positions, r_in=aperture_radius * 1.7, r_out=aperture_radius * 2.4)
            sums = np.asarray(aperture_photometry(data, apertures)["aperture_sum"], dtype=float)
            backgrounds = []
            for mask in annuli.to_mask(method="center"):
                values = mask.multiply(data)
                valid = values[mask.data > 0]
                backgrounds.append(float(np.nanmedian(valid)) * apertures.area)
            fluxes = sums - np.asarray(backgrounds)
            comparison_flux = float(np.nansum(fluxes[1:]))
            relative = float(fluxes[0] / comparison_flux) if comparison_flux > 0 else float("nan")
            time = float(header.get("HOPSJD", index))
            uncertainty = abs(relative) * math.sqrt(1 / max(fluxes[0], 1) + 1 / max(comparison_flux, 1))
            rows.append((time, relative, uncertainty))
            _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, f"Measured {path.name}", index, len(frames))
        array = np.asarray(rows)
        finite = np.isfinite(array[:, 1])
        normalization = np.nanmedian(array[finite, 1]) if np.any(finite) else 1.0
        array[:, 1:] /= normalization
        pending, target = project.begin_transaction(StageID.PHOTOMETRY)
        output = pending / "light_curve_aperture.txt"
        np.savetxt(output, array, header="JD_UTC relative_flux relative_flux_uncertainty")
        (pending / "photometry.json").write_text(
            json.dumps(
                {"target": target_xy, "comparisons": comparisons, "aperture_radius": aperture_radius},
                indent=2,
            ),
            encoding="utf-8",
        )
        project.commit_transaction(pending, target)
        _emit(emit, StageID.PHOTOMETRY, JobStatus.SUCCEEDED, "Photometry complete", len(frames), len(frames))
        return target / output.name


class FittingService:
    def run(
        self,
        project: ProjectWorkspace,
        parameters: PlanetParameters,
        *,
        full: bool,
        exposure_time: float,
        filter_name: str,
        latitude: float,
        longitude: float,
        iterations: int = 5000,
        burn_in: int = 1000,
    ) -> dict[str, Any]:
        import exoclock

        import hops.pylightcurve41 as plc

        light_curve_path = project.outputs_dir / StageID.PHOTOMETRY.value / "light_curve_aperture.txt"
        light_curve = np.loadtxt(light_curve_path, unpack=True)
        planet = plc.Planet(
            parameters.name,
            exoclock.Hours(parameters.ra).deg(),
            exoclock.Degrees(parameters.dec).deg_coord(),
            parameters.logg,
            parameters.temperature,
            parameters.metallicity,
            parameters.rp_over_rs,
            parameters.period,
            parameters.sma_over_rs,
            parameters.eccentricity,
            parameters.inclination,
            parameters.periastron,
            parameters.mid_time,
        )
        planet.add_observation(
            time=light_curve[0],
            time_format="JD_UTC",
            exp_time=exposure_time,
            time_stamp="start",
            flux=light_curve[1],
            flux_unc=light_curve[2],
            flux_format="flux",
            filter_name=filter_name,
            observatory_latitude=latitude,
            observatory_longitude=longitude,
            detrending_series="airmass",
            detrending_order=1,
        )
        pending, target = project.begin_transaction(StageID.FITTING)
        result = planet.transit_fitting(
            output_folder=str(pending) if full else None,
            scale_uncertainties=True,
            filter_outliers=True,
            fit_sma_over_rs=False,
            fit_inclination=False,
            counter=None,
            window_counter=False,
            iterations=iterations,
            burn_in=burn_in,
            optimiser="emcee" if full else "curve_fit",
        )
        summary = {"planet": parameters.name, "source": parameters.source, "complete": bool(result)}
        (pending / "fit-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        project.commit_transaction(pending, target)
        return result


def _combine(arrays: list[np.ndarray], method: str) -> np.ndarray:
    if not arrays:
        raise ValueError("At least one array is required")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise LEAPSError(
            "CALIBRATION_SHAPE_MISMATCH",
            "Calibration frames have different sizes",
            "All calibration frames must match the science frame dimensions.",
            ["Review frame assignments", "Exclude the mismatched frame"],
            stage=StageID.REDUCTION,
        )
    stack = np.stack([np.asarray(array, dtype=np.float32) for array in arrays])
    return np.nanmean(stack, axis=0) if method == "mean" else np.nanmedian(stack, axis=0)


def _julian_date(header: Any, config: ReductionConfig) -> float:
    from astropy.time import Time

    date = str(header.get(config.date_key, ""))
    if "T" not in date and header.get(config.time_key):
        date = f"{date}T{header[config.time_key]}"
    try:
        return float(Time(date, format="isot", scale="utc").jd)
    except Exception:
        return float(header.get("JD", header.get("MJD-OBS", 0.0)))
