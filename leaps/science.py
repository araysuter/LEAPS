from __future__ import annotations

import hashlib
import json
import math
import threading
import warnings
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import PlanetParameters
from .models import JobStatus, LEAPSError, StageEvent, StageID
from .project import ProjectWorkspace

Emitter = Callable[[StageEvent], None]


def _read_fits_image(path: Path) -> tuple[np.ndarray, Any]:
    """Read a scaled FITS image without modifying or memory-mapping scaled pixels.

    Astropy cannot expose FITS images containing BZERO, BSCALE, or BLANK through
    its usual scaled memmap path. Reading the stored values and applying the
    standard FITS scaling ourselves keeps raw files read-only and avoids loading
    an additional, implicitly scaled copy.
    """
    from astropy.io import fits
    from astropy.utils.exceptions import AstropyUserWarning

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Header block contains null bytes instead of spaces for padding.*",
            category=AstropyUserWarning,
        )
        with fits.open(
            path,
            memmap=True,
            do_not_scale_image_data=True,
            ignore_missing_end=True,
        ) as hdus:
            hdu = next(candidate for candidate in hdus if getattr(candidate, "data", None) is not None)
            stored = np.asarray(hdu.data)
            data = stored.astype(np.float32, copy=True)
            header = hdu.header.copy()

    blank = header.get("BLANK")
    if blank is not None:
        data[data == float(blank)] = np.nan
    scale = float(header.get("BSCALE", 1.0))
    zero = float(header.get("BZERO", 0.0))
    if scale != 1.0:
        data *= scale
    if zero != 0.0:
        data += zero
    return data, header


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
class PhotometryConfig:
    aperture_radius: float = 8.0
    sky_inner_aperture: float = 1.7
    sky_outer_aperture: float = 2.4
    saturation_fraction: float = 0.95
    camera_gain: float = 1.0
    variable_aperture: bool = True
    geometric_center: bool = False
    centroids_snr: float = 4.0
    stars_snr: float = 4.0


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
                data, header = _read_fits_image(path)
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
    def _load(project: ProjectWorkspace, paths: Iterable[str]) -> list[tuple[np.ndarray, Any]]:
        result: list[tuple[np.ndarray, Any]] = []
        for relative in paths:
            path = project.resolve(relative)
            try:
                result.append(_read_fits_image(path))
            except Exception as exc:
                raise LEAPSError(
                    "CALIBRATION_FRAME_UNREADABLE",
                    f"{path.name} could not be read",
                    "The calibration frames could not be combined. The raw FITS file was not modified.",
                    ["Review the frame assignment", "Open the FITS file", "Export diagnostics"],
                    stage=StageID.REDUCTION,
                    technical_details=f"{type(exc).__name__}: {exc}",
                ) from exc
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
                    {
                        "file": path.name,
                        "x0": x0,
                        "y0": y0,
                        "rotation": rotation,
                        "matched": count,
                        "matrix": matrix.tolist(),
                    }
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
        from astropy.time import Time

        from hops.hops_tools.image_analysis import image_find_stars, image_plate_solve

        coordinate = SkyCoord(ra, dec, unit=(units.hourangle, units.deg))
        _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, "Coordinates validated", 0, 3)
        data, header = _read_fits_image(Path(frame))
        mean = float(header.get("HOPSMEAN", np.nanmedian(data)))
        std = float(header.get("HOPSSTD", 1.4826 * np.nanmedian(np.abs(data - mean))))
        psf = max(float(header.get("HOPSPSF", 2.0)), 1.0)
        burn_limit = float(header.get("HOPSSAT", header.get("SATURATE", np.nanmax(data))))
        stars = image_find_stars(
            data,
            header,
            mean=mean,
            std=std,
            psf=psf,
            burn_limit=burn_limit,
            star_limit=100,
        ) or []
        if len(stars) < 5:
            raise LEAPSError(
                "TOO_FEW_PLATE_STARS",
                "Too few stars were detected",
                "Plate solving needs at least five usable stars.",
                ["Adjust contrast and detection threshold", "Choose another frame"],
                stage=StageID.PHOTOMETRY,
            )
        existing_wcs = None
        try:
            from astropy.wcs import WCS
            from astropy.wcs.utils import proj_plane_pixel_scales
            from astropy.wcs.wcs import FITSFixedWarning

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FITSFixedWarning)
                existing_wcs = WCS(header)
            if existing_wcs.has_celestial:
                x, y = coordinate.to_pixel(existing_wcs)
                x, y = float(np.asarray(x)), float(np.asarray(y))
                nearest_star = min(
                    (
                        math.hypot(float(star[0]) - x, float(star[1]) - y)
                        for star in stars
                    ),
                    default=float("inf"),
                )
                if (
                    0 <= x < data.shape[1]
                    and 0 <= y < data.shape[0]
                    and nearest_star <= max(5.0 * psf, 8.0)
                ):
                    scales = proj_plane_pixel_scales(existing_wcs.celestial) * 3600.0
                    detected_scale = float(np.nanmedian(scales))
                    attempt = PlateSolveAttempt(
                        0,
                        detected_scale,
                        "complete",
                        "Existing FITS WCS validated and contains the target",
                    )
                    _emit(
                        emit,
                        StageID.PHOTOMETRY,
                        JobStatus.SUCCEEDED,
                        "Existing FITS WCS validated",
                        1,
                        1,
                    )
                    return PlateSolveResult(
                        True,
                        [attempt],
                        (x, y),
                        len(stars),
                        dict(existing_wcs.to_header()),
                    )
        except Exception:
            pass
        _emit(
            emit,
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            f"{len(stars)} stars detected",
            0,
            3,
        )
        cache_key = hashlib.sha256(f"{coordinate.ra.deg:.6f},{coordinate.dec.deg:.6f}".encode()).hexdigest()[:16]
        gaia_cache = Path(frame).resolve().parents[2] / "cache" / f"gaia-{cache_key}.ecsv"
        gaia_query = None
        catalog_limit = max(100, 10 * len(stars))
        if gaia_cache.exists():
            try:
                from astropy.table import Table

                gaia_query = Table.read(gaia_cache, format="ascii.ecsv")
                if len(gaia_query) < catalog_limit:
                    gaia_query = None
            except Exception:
                gaia_query = None
        if gaia_query is None:
            try:
                from hops.hops_tools.centroids_and_stars import _get_gaia_stars

                gaia_query = _get_gaia_stars(
                    coordinate.ra.deg,
                    coordinate.dec.deg,
                    0.5,
                    limit=catalog_limit,
                )
                gaia_cache.parent.mkdir(parents=True, exist_ok=True)
                gaia_query.write(gaia_cache, format="ascii.ecsv", overwrite=True)
            except Exception as exc:
                raise LEAPSError(
                    "GAIA_CATALOG_UNAVAILABLE",
                    "Gaia catalogue could not be reached",
                    "Manual target selection is still available. Retry online or install offline data for this target region.",
                    ["Select target manually", "Retry Gaia", "Open Offline Data settings"],
                    stage=StageID.PHOTOMETRY,
                    technical_details=f"{type(exc).__name__}: {exc}",
                ) from exc
        if existing_wcs is not None and existing_wcs.has_celestial:
            corrected = self._correct_existing_wcs(
                existing_wcs,
                data.shape,
                stars,
                gaia_query,
                coordinate,
                psf,
            )
            if corrected is not None:
                _emit(
                    emit,
                    StageID.PHOTOMETRY,
                    JobStatus.SUCCEEDED,
                    "Existing FITS WCS corrected with Gaia",
                    1,
                    1,
                )
                return corrected
        attempts: list[PlateSolveAttempt] = []
        base_scale = pixel_scale if pixel_scale > 0 else 2.0 / psf
        for index, scale in enumerate((base_scale, base_scale * 0.5, base_scale * 2.0), start=1):
            token.raise_if_cancelled()
            try:
                timestamp = (
                    Time(float(header["HOPSJD"]), format="jd")
                    if header.get("HOPSJD") is not None
                    else Time(header.get("DATE-OBS", Time.now().isot))
                )
                solution = image_plate_solve(
                    data,
                    header,
                    coordinate.ra.deg,
                    coordinate.dec.deg,
                    timestamp,
                    stars=stars,
                    pixel=scale,
                    mean=mean,
                    std=std,
                    psf=psf,
                    burn_limit=burn_limit,
                    gaia_query_ext=gaia_query,
                    verbose=False,
                )
                identified = len(solution["identified_stars"])
                if identified < 5:
                    raise ValueError(f"Only {identified} of {len(stars)} detected stars matched")
                x, y = coordinate.to_pixel(solution["plate_solution"])
                nearest_star = min(
                    math.hypot(float(star[0]) - float(x), float(star[1]) - float(y))
                    for star in stars
                )
                if nearest_star > max(5.0 * psf, 8.0):
                    raise ValueError(
                        f"Solved target is {nearest_star:.1f} pixels from the nearest detected star"
                    )
                attempts.append(PlateSolveAttempt(index, scale, "complete", f"{identified} stars matched"))
                _emit(emit, StageID.PHOTOMETRY, JobStatus.SUCCEEDED, "Plate solution found", index, 3)
                return PlateSolveResult(
                    True,
                    attempts,
                    (float(x), float(y)),
                    identified,
                    dict(solution["plate_solution"].to_header(relax=True)),
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

    @staticmethod
    def _correct_existing_wcs(
        existing_wcs: Any,
        image_shape: tuple[int, int],
        stars: list[Any],
        gaia_query: Any,
        coordinate: Any,
        psf: float,
    ) -> PlateSolveResult | None:
        """Correct a plausible header WCS for telescope pointing offset.

        Many acquisition programs write the requested target coordinates as
        CRVAL even when the actual pointing is tens of pixels away. This keeps
        HOPS's Gaia catalogue and WCS fit, but gives it a robust translation
        seed before the bounded blind attempts.
        """
        from astropy.coordinates import SkyCoord
        from astropy.wcs.utils import fit_wcs_from_points, proj_plane_pixel_scales
        from scipy.spatial import cKDTree

        detected = np.asarray([[star[0], star[1]] for star in stars], dtype=float)
        if len(detected) < 5:
            return None
        try:
            world = np.column_stack(
                (
                    np.asarray(gaia_query["ra"], dtype=float),
                    np.asarray(gaia_query["dec"], dtype=float),
                )
            )
            projected = np.asarray(existing_wcs.all_world2pix(world, 0), dtype=float)
        except Exception:
            return None
        height, width = image_shape
        margin = 0.2 * min(width, height)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] > -margin)
            & (projected[:, 0] < width + margin)
            & (projected[:, 1] > -margin)
            & (projected[:, 1] < height + margin)
        )
        projected = projected[valid][:150]
        world = world[valid][:150]
        if len(projected) < 5:
            return None

        tree = cKDTree(detected)
        tolerance = max(3.0 * psf, 8.0)
        max_shift = 0.25 * min(width, height)
        best_score = 0
        best_shift = None
        for catalogue_point in projected[:50]:
            for detected_point in detected[:50]:
                shift = detected_point - catalogue_point
                if np.linalg.norm(shift) > max_shift:
                    continue
                distances, indices = tree.query(projected + shift, k=1)
                score = len(set(indices[distances < tolerance].tolist()))
                if score > best_score:
                    best_score = score
                    best_shift = shift
        if best_shift is None or best_score < 5:
            return None

        distances, indices = tree.query(projected + best_shift, k=1)
        candidate_rows = np.where(distances < tolerance)[0]
        pairs: list[tuple[int, int]] = []
        used_detected: set[int] = set()
        for catalogue_index in sorted(candidate_rows, key=lambda index: distances[index]):
            detected_index = int(indices[catalogue_index])
            if detected_index not in used_detected:
                pairs.append((int(catalogue_index), detected_index))
                used_detected.add(detected_index)
        if len(pairs) < 5:
            return None

        catalogue_indices = np.asarray([pair[0] for pair in pairs], dtype=int)
        detected_indices = np.asarray([pair[1] for pair in pairs], dtype=int)
        try:
            solution = fit_wcs_from_points(
                detected[detected_indices].T,
                SkyCoord(world[catalogue_indices], unit="deg"),
                sip_degree=None,
            )
            refined = np.asarray(solution.all_world2pix(world, 0), dtype=float)
            refined_distances, refined_indices = tree.query(refined, k=1)
            candidate_rows = np.where(refined_distances < tolerance)[0]
            pairs = []
            used_detected = set()
            for catalogue_index in sorted(
                candidate_rows, key=lambda index: refined_distances[index]
            ):
                detected_index = int(refined_indices[catalogue_index])
                if detected_index not in used_detected:
                    pairs.append((int(catalogue_index), detected_index))
                    used_detected.add(detected_index)
            if len(pairs) < 5:
                return None
            catalogue_indices = np.asarray([pair[0] for pair in pairs], dtype=int)
            detected_indices = np.asarray([pair[1] for pair in pairs], dtype=int)
            solution = fit_wcs_from_points(
                detected[detected_indices].T,
                SkyCoord(world[catalogue_indices], unit="deg"),
                sip_degree=2 if len(pairs) >= 10 else None,
            )
            target_x, target_y = map(float, coordinate.to_pixel(solution))
        except Exception:
            return None

        nearest_index = int(
            np.argmin(np.hypot(detected[:, 0] - target_x, detected[:, 1] - target_y))
        )
        nearest_distance = float(
            math.hypot(
                detected[nearest_index, 0] - target_x,
                detected[nearest_index, 1] - target_y,
            )
        )
        if nearest_distance > max(5.0 * psf, 8.0):
            return None
        scales = proj_plane_pixel_scales(solution.celestial) * 3600.0
        pixel_scale = float(np.nanmedian(scales))
        return PlateSolveResult(
            True,
            [
                PlateSolveAttempt(
                    0,
                    pixel_scale,
                    "complete",
                    f"Existing FITS WCS corrected with {len(pairs)} Gaia matches",
                )
            ],
            (
                float(detected[nearest_index, 0]),
                float(detected[nearest_index, 1]),
            ),
            len(pairs),
            dict(solution.to_header(relax=True)),
        )


class PhotometryService:
    def locate_star(
        self,
        frame: str | Path,
        x: float,
        y: float,
        config: PhotometryConfig | None = None,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
    ) -> dict[str, float]:
        if token:
            token.raise_if_cancelled()
        config = config or PhotometryConfig()
        data, header = _read_fits_image(Path(frame))
        return self._locate_star(data, header, x, y, config.aperture_radius, config)

    @staticmethod
    def _locate_star(
        data: np.ndarray,
        header: Any,
        x: float,
        y: float,
        aperture: float,
        config: PhotometryConfig,
    ) -> dict[str, float]:
        from hops.hops_tools.image_analysis import image_find_stars

        mean = float(header.get("HOPSMEAN", np.nanmedian(data)))
        std = float(header.get("HOPSSTD", 1.4826 * np.nanmedian(np.abs(data - mean))))
        psf = max(float(header.get("HOPSPSF", 2.0)), 1.0)
        saturation = float(
            header.get("HOPSSAT", header.get("SATURATE", np.nanmax(data)))
        ) * config.saturation_fraction
        search = max(5.0 * psf, aperture * 2.0)
        stars = image_find_stars(
            data,
            header,
            x_low=x - search,
            x_upper=x + search,
            y_low=y - search,
            y_upper=y + search,
            x_centre=x,
            y_centre=y,
            mean=mean,
            std=std,
            burn_limit=saturation,
            psf=psf,
            centroids_snr=config.centroids_snr,
            stars_snr=config.stars_snr,
            order_by_flux=False,
            absolute_aperture=aperture,
            sky_inner_aperture=config.sky_inner_aperture,
            sky_outer_aperture=config.sky_outer_aperture,
            star_limit=5,
        ) or []
        if not stars:
            raise LEAPSError(
                "PHOTOMETRY_STAR_NOT_FOUND",
                "No acceptable star was found at that position",
                "Click closer to the center of an unsaturated star inside the usable field of view.",
                ["Choose another star", "Adjust advanced detection settings"],
                stage=StageID.PHOTOMETRY,
            )
        star = min(stars, key=lambda value: math.hypot(float(value[0]) - x, float(value[1]) - y))
        gaussian_x, gaussian_y = float(star[0]), float(star[1])
        aperture_x, aperture_y = gaussian_x, gaussian_y
        total_flux = float(star[6])
        sky_flux = float(star[8])
        if config.geometric_center:
            from photutils.aperture import CircularAperture, aperture_photometry

            half_width = max(int(3.0 * psf), 1)
            x1 = max(int(gaussian_x) - half_width, 0)
            x2 = min(int(gaussian_x) + half_width + 1, data.shape[1])
            y1 = max(int(gaussian_y) - half_width, 0)
            y2 = min(int(gaussian_y) + half_width + 1, data.shape[0])
            area = np.asarray(data[y1:y2, x1:x2], dtype=float)
            area_x, area_y = np.meshgrid(
                np.arange(x1, x2) + 0.5,
                np.arange(y1, y2) + 0.5,
            )
            finite = np.isfinite(area)
            weight = float(np.sum(area[finite]))
            if weight != 0 and math.isfinite(weight):
                aperture_x = float(np.sum(area[finite] * area_x[finite]) / weight)
                aperture_y = float(np.sum(area[finite] * area_y[finite]) / weight)
                total_flux = float(
                    aperture_photometry(
                        data,
                        CircularAperture(
                            np.array([aperture_x - 0.5, aperture_y - 0.5]), aperture
                        ),
                    )["aperture_sum"][0]
                )
        gaussian_flux = float(2.0 * math.pi * star[2] * star[4] * star[5])
        aperture_flux = total_flux - sky_flux
        return {
            "x": aperture_x,
            "y": aperture_y,
            "gaussian_x": gaussian_x,
            "gaussian_y": gaussian_y,
            "aperture": float(aperture),
            "peak": float(star[2] + star[3]),
            "total_flux": total_flux,
            "background_flux": sky_flux,
            "background_error": float(star[9]),
            "aperture_flux": aperture_flux,
            "aperture_error": float(
                math.sqrt(abs(aperture_flux) / max(config.camera_gain, 1e-9) + float(star[9]) ** 2)
            ),
            "gaussian_flux": gaussian_flux,
            "gaussian_error": float(math.sqrt(abs(gaussian_flux) / max(config.camera_gain, 1e-9))),
            "hwhm": float(0.5 * 2.355 * max(star[4], star[5])),
        }

    def rank_comparisons(
        self, frame: str | Path, target_xy: tuple[float, float], limit: int = 10
    ) -> list[dict[str, float]]:
        from hops.hops_tools.image_analysis import image_find_stars

        data, header = _read_fits_image(Path(frame))
        stars = np.asarray(image_find_stars(data, header, star_limit=150) or [])
        if stars.size == 0:
            return []
        tx, ty = target_xy
        config = PhotometryConfig()
        try:
            target_flux = self._locate_star(
                data, header, target_xy[0], target_xy[1], config.aperture_radius, config
            )["aperture_flux"]
        except LEAPSError:
            target_flux = float(np.nanmedian(stars[:, 7]))
        ranked = []
        for star in stars:
            x, y = map(float, star[:2])
            peak = float(star[2] + star[3])
            flux = float(star[7])
            distance = math.hypot(x - tx, y - ty)
            if distance < max(10.0, float(header.get("HOPSPSF", 3)) * 5):
                continue
            saturation = float(header.get("HOPSSAT", np.nanmax(data)))
            flux_similarity = abs(math.log10(max(flux, 1) / max(target_flux, 1)))
            score = 1.0 - flux_similarity - 0.02 * math.log10(max(distance, 1))
            if peak >= 0.95 * saturation:
                score -= 2.0
            ranked.append(
                {
                    "x": x,
                    "y": y,
                    "peak": peak,
                    "flux": flux,
                    "distance": distance,
                    "score": score,
                }
            )
        return sorted(ranked, key=lambda item: item["score"], reverse=True)[:limit]

    def run(
        self,
        project: ProjectWorkspace,
        target_xy: tuple[float, float],
        comparisons: list[tuple[float, float]],
        aperture_radius: float,
        emit: Emitter | None = None,
        token: CancellationToken | None = None,
        config: PhotometryConfig | None = None,
    ) -> Path:
        token = token or CancellationToken()
        frames = sorted((project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        if not frames:
            raise LEAPSError(
                "PHOTOMETRY_INPUT_MISSING",
                "No reduced frames are available",
                "Run Reduction before starting photometry.",
                ["Open Reduction"],
                stage=StageID.PHOTOMETRY,
            )
        positions = [target_xy, *comparisons]
        if len(positions) < 2:
            raise LEAPSError(
                "COMPARISON_STARS_REQUIRED",
                "Choose at least one comparison star",
                "Differential photometry needs the target and one or more comparison stars.",
                ["Review suggested comparison stars"],
                stage=StageID.PHOTOMETRY,
            )
        config = config or PhotometryConfig(aperture_radius=aperture_radius)
        config.aperture_radius = aperture_radius
        alignment_path = project.outputs_dir / StageID.ALIGNMENT.value / "alignment.json"
        alignment_records = []
        if alignment_path.exists():
            alignment_records = json.loads(alignment_path.read_text(encoding="utf-8"))
            failed_frames = {
                str(record.get("file"))
                for record in alignment_records
                if record.get("failed")
            }
            frames = [path for path in frames if path.name not in failed_frames]
            if not frames:
                raise LEAPSError(
                    "PHOTOMETRY_ALIGNMENT_MISSING",
                    "No successfully aligned frames are available",
                    "Review the Alignment diagnostics and rerun that stage.",
                    ["Open Alignment", "Review diagnostics"],
                    stage=StageID.PHOTOMETRY,
                )
        transforms = {record.get("file"): self._alignment_matrix(record) for record in alignment_records}
        reference_transform = transforms.get(frames[0].name, np.eye(3))
        try:
            inverse_reference = np.linalg.inv(reference_transform)
        except np.linalg.LinAlgError:
            inverse_reference = np.eye(3)

        fingerprint_payload = {
            "frames": [path.name for path in frames],
            "positions": positions,
            "config": asdict(config),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        checkpoint = project.checkpoints_dir / "photometry.json"
        rows: list[dict[str, Any]] = []
        if checkpoint.exists():
            try:
                saved = json.loads(checkpoint.read_text(encoding="utf-8"))
                if saved.get("fingerprint") == fingerprint:
                    rows = list(saved.get("rows", []))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                rows = []
        start = len(rows)
        reference_psf = max(float(_read_fits_image(frames[0])[1].get("HOPSPSF", 1.0)), 1e-9)
        for index, path in enumerate(frames[start:], start=start + 1):
            token.raise_if_cancelled()
            data, header = _read_fits_image(path)
            transform = transforms.get(path.name, np.eye(3)) @ inverse_reference
            psf = max(float(header.get("HOPSPSF", reference_psf)), 1e-9)
            scale = psf / reference_psf if config.variable_aperture else 1.0
            aperture = aperture_radius * scale
            measurements = []
            for x, y in positions:
                predicted = transform @ np.array([x, y, 1.0])
                try:
                    measurement = self._locate_star(
                        data,
                        header,
                        float(predicted[0]),
                        float(predicted[1]),
                        aperture,
                        config,
                    )
                    measurement["failed"] = False
                except LEAPSError as exc:
                    measurement = {
                        "x": float(predicted[0]),
                        "y": float(predicted[1]),
                        "gaussian_x": float(predicted[0]),
                        "gaussian_y": float(predicted[1]),
                        "aperture": aperture,
                        "aperture_flux": float("nan"),
                        "aperture_error": float("nan"),
                        "gaussian_flux": float("nan"),
                        "gaussian_error": float("nan"),
                        "failed": True,
                        "reason": exc.message,
                    }
                measurements.append(measurement)
            rows.append(
                {
                    "file": path.name,
                    "jd": float(header.get("HOPSJD", index)),
                    "measurements": measurements,
                }
            )
            checkpoint.write_text(
                json.dumps({"fingerprint": fingerprint, "rows": rows}, indent=2, allow_nan=True),
                encoding="utf-8",
            )
            _emit(emit, StageID.PHOTOMETRY, JobStatus.RUNNING, f"Measured {path.name}", index, len(frames))

        aperture_array = self._light_curve(rows, "aperture_flux", "aperture_error")
        gaussian_array = self._light_curve(rows, "gaussian_flux", "gaussian_error")
        pending, target = project.begin_transaction(StageID.PHOTOMETRY)
        output = pending / "light_curve_aperture.txt"
        np.savetxt(output, aperture_array, header="JD_UTC relative_flux relative_flux_uncertainty")
        np.savetxt(
            pending / "light_curve_gauss.txt",
            gaussian_array,
            header="JD_UTC relative_flux relative_flux_uncertainty",
        )
        np.savetxt(pending / "PHOTOMETRY_APERTURE.txt", aperture_array)
        np.savetxt(pending / "PHOTOMETRY_GAUSS.txt", gaussian_array)
        np.savetxt(
            pending / "PHOTOMETRY_a.txt",
            self._measurement_table(rows, "aperture_flux", "aperture_error"),
            fmt="%s",
        )
        np.savetxt(
            pending / "PHOTOMETRY_g.txt",
            self._measurement_table(rows, "gaussian_flux", "gaussian_error"),
            fmt="%s",
        )
        (pending / "measurements.json").write_text(
            json.dumps(rows, indent=2, allow_nan=True), encoding="utf-8"
        )
        (pending / "photometry.json").write_text(
            json.dumps(
                {
                    "engine": "HOPS photometry",
                    "target": target_xy,
                    "comparisons": comparisons,
                    "config": asdict(config),
                    "checkpoint_fingerprint": fingerprint,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (pending / "ExoClock_info.txt").write_text(
            "\n".join(
                (
                    "LEAPS / HOPS-compatible photometry",
                    f"Target: {project.manifest.target_name or 'Unnamed target'}",
                    f"Coordinates: {project.manifest.target_ra} {project.manifest.target_dec}",
                    "Time format: JD_UTC",
                    "Time stamp: exposure start",
                    "Flux format: target flux / summed comparison flux",
                    "Suggested upload: PHOTOMETRY_APERTURE.txt",
                )
            ),
            encoding="utf-8",
        )
        self._write_figures(
            pending,
            frames[0],
            positions,
            aperture_radius,
            aperture_array,
            gaussian_array,
        )
        project.commit_transaction(pending, target)
        checkpoint.unlink(missing_ok=True)
        _emit(emit, StageID.PHOTOMETRY, JobStatus.SUCCEEDED, "Photometry complete", len(frames), len(frames))
        return target / output.name

    @staticmethod
    def _alignment_matrix(record: dict[str, Any]) -> np.ndarray:
        if record.get("matrix"):
            matrix = np.asarray(record["matrix"], dtype=float)
            if matrix.shape == (3, 3):
                return matrix
        rotation = float(record.get("rotation", 0.0) or 0.0)
        cosine, sine = math.cos(rotation), math.sin(rotation)
        return np.array(
            [
                [cosine, -sine, float(record.get("x0", 0.0) or 0.0)],
                [sine, cosine, float(record.get("y0", 0.0) or 0.0)],
                [0.0, 0.0, 1.0],
            ]
        )

    @staticmethod
    def _light_curve(
        rows: list[dict[str, Any]], flux_key: str, error_key: str
    ) -> np.ndarray:
        times = np.asarray([row["jd"] for row in rows], dtype=float)
        fluxes = np.asarray(
            [[star.get(flux_key, float("nan")) for star in row["measurements"]] for row in rows],
            dtype=float,
        )
        errors = np.asarray(
            [[star.get(error_key, float("nan")) for star in row["measurements"]] for row in rows],
            dtype=float,
        )
        comparison_flux = np.nansum(fluxes[:, 1:], axis=1)
        relative = fluxes[:, 0] / comparison_flux
        comparison_error = np.sqrt(np.nansum(errors[:, 1:] ** 2, axis=1))
        relative_error = np.abs(relative) * np.sqrt(
            (errors[:, 0] / fluxes[:, 0]) ** 2
            + (comparison_error / comparison_flux) ** 2
        )
        normalization = float(np.nanmedian(relative))
        if not math.isfinite(normalization) or normalization == 0:
            normalization = 1.0
        return np.column_stack((times, relative / normalization, relative_error / normalization))

    @staticmethod
    def _measurement_table(
        rows: list[dict[str, Any]], flux_key: str, error_key: str
    ) -> np.ndarray:
        table: list[list[Any]] = []
        for row in rows:
            values: list[Any] = [row["file"], row["jd"]]
            for star in row["measurements"]:
                gaussian = flux_key == "gaussian_flux"
                values.extend(
                    (
                        star.get("gaussian_x" if gaussian else "x", float("nan")),
                        star.get("gaussian_y" if gaussian else "y", float("nan")),
                        star.get(flux_key, float("nan")),
                        star.get(error_key, float("nan")),
                    )
                )
            table.append(values)
        return np.asarray(table, dtype=object)

    @staticmethod
    def _write_figures(
        destination: Path,
        reference_frame: Path,
        positions: list[tuple[float, float]],
        aperture: float,
        aperture_curve: np.ndarray,
        gaussian_curve: np.ndarray,
    ) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle

        data = np.asarray(_read_fits_image(reference_frame)[0], dtype=float)
        median = float(np.nanmedian(data))
        std = float(1.4826 * np.nanmedian(np.abs(data - median))) or 1.0
        field = Figure(figsize=(8, 8), facecolor="white")
        FigureCanvasAgg(field)
        axis = field.add_subplot(111)
        axis.imshow(
            data,
            origin="lower",
            cmap="gray_r",
            vmin=median - 3 * std,
            vmax=median + 20 * std,
        )
        for index, (x, y) in enumerate(positions):
            color = "#d99000" if index == 0 else "#00a6d6"
            label = "T" if index == 0 else f"C{index}"
            axis.add_patch(Circle((x, y), aperture, fill=False, color=color, linewidth=1.2))
            axis.text(x + aperture + 3, y + aperture + 3, label, color=color, fontsize=9)
        axis.set_title("Selected photometry field")
        field.savefig(destination / "FOV.png", dpi=160, bbox_inches="tight")
        field.savefig(destination / "FOV.pdf", bbox_inches="tight")

        results = Figure(figsize=(10, 5), facecolor="white")
        FigureCanvasAgg(results)
        axis = results.add_subplot(111)
        start = aperture_curve[0, 0]
        axis.errorbar(
            (aperture_curve[:, 0] - start) * 24,
            aperture_curve[:, 1],
            yerr=aperture_curve[:, 2],
            fmt="ko",
            markersize=3,
            linewidth=0.7,
            label="Aperture",
        )
        axis.plot(
            (gaussian_curve[:, 0] - start) * 24,
            gaussian_curve[:, 1],
            "o",
            color="#d85845",
            markersize=3,
            label="Gaussian",
        )
        axis.set_xlabel("Time from first exposure (hours)")
        axis.set_ylabel("Normalized relative flux")
        axis.legend()
        axis.grid(alpha=0.2)
        results.savefig(destination / "RESULTS.png", dpi=160, bbox_inches="tight")
        results.savefig(destination / "RESULTS.pdf", bbox_inches="tight")


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
