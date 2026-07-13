from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from leaps.models import LEAPSError
from leaps.project import ProjectWorkspace
from leaps.science import (
    CancellationToken,
    PhotometryConfig,
    PhotometryService,
    PlateSolveService,
    ReductionConfig,
    ReductionService,
)


def test_reduction_analysis_import_does_not_initialize_online_catalogues() -> None:
    sys.modules.pop("exoclock", None)
    from hops.hops_tools.image_analysis import image_mean_std

    mean, std = image_mean_std(np.array([[1.0, 1.0], [1.0, 2.0]]))
    assert mean == 1.0
    assert std >= 0
    assert "exoclock" not in sys.modules


def test_reduction_keeps_raw_fits_immutable_and_commits_output(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "light_001.fits"
    header = fits.Header()
    header["EXPTIME"] = 30.0
    header["DATE-OBS"] = "2026-07-12T02:00:00"
    fits.writeto(raw, np.arange(256, dtype=np.float32).reshape(16, 16), header)
    before = hashlib.sha256(raw.read_bytes()).hexdigest()
    project = ProjectWorkspace.create(tmp_path)
    project.manifest.raw_files["science"] = ["light_001.fits"]
    project.save()
    monkeypatch.setattr(ReductionService, "_statistics", staticmethod(lambda data, header: (100.0, 5.0, 2.5)))
    output = ReductionService().run(project, ReductionConfig())
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == before
    assert len(list(output.glob("r_*.fits"))) == 1
    assert (output / "frames.json").exists()


def test_reduction_reads_unsigned_scaled_fits_without_changing_raw_files(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    header = fits.Header({"EXPTIME": 0.0, "DATE-OBS": "2026-07-12T02:00:00"})
    raw_files: dict[str, list[str]] = {
        "science": ["light.fits"],
        "bias": ["bias.fits"],
        "dark": [],
        "dark_flat": [],
        "flat": [],
        "unknown": [],
    }
    for name, value in (("light.fits", 40000), ("bias.fits", 1000)):
        fits.writeto(
            tmp_path / name,
            np.full((8, 8), value, dtype=np.uint16),
            header=header,
        )
    checksums = {
        name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
        for name in ("light.fits", "bias.fits")
    }
    project.manifest.raw_files = raw_files
    project.save()
    monkeypatch.setattr(
        ReductionService,
        "_statistics",
        staticmethod(lambda data, header: (float(np.mean(data)), 0.0, 2.5)),
    )

    output = ReductionService().run(project, ReductionConfig())

    reduced = fits.getdata(next(output.glob("r_*.fits")))
    assert np.allclose(reduced, 39000.0)
    assert all(
        hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum
        for name, checksum in checksums.items()
    )


def test_cancellation_is_typed_and_recoverable() -> None:
    token = CancellationToken()
    token.cancel()
    try:
        token.raise_if_cancelled()
    except LEAPSError as failure:
        assert failure.code == "JOB_CANCELLED"
        assert "Resume" in failure.recovery
    else:
        raise AssertionError("cancelled token did not raise")


def test_hops_photometry_writes_aperture_gaussian_and_legacy_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    reduction = project.outputs_dir / "reduction"
    reduction.mkdir()
    for index in range(2):
        header = fits.Header(
            {
                "HOPSJD": 2460000.0 + index / 100,
                "HOPSMEAN": 100.0,
                "HOPSSTD": 5.0,
                "HOPSPSF": 2.0,
            }
        )
        fits.writeto(
            reduction / f"r_{index + 1:05d}.fits",
            np.full((32, 32), 100.0, dtype=np.float32),
            header,
        )
    alignment = project.outputs_dir / "alignment"
    alignment.mkdir()
    (alignment / "alignment.json").write_text(
        '[{"file":"r_00001.fits","x0":0,"y0":0,"rotation":0},'
        '{"file":"r_00002.fits","x0":1,"y0":0,"rotation":0}]',
        encoding="utf-8",
    )

    def located(data, header, x, y, aperture, config):
        target = x < 15
        flux = 1000.0 if target else 500.0
        return {
            "x": x,
            "y": y,
            "aperture": aperture,
            "aperture_flux": flux,
            "aperture_error": 10.0,
            "gaussian_flux": flux * 0.98,
            "gaussian_error": 11.0,
            "failed": False,
        }

    monkeypatch.setattr(PhotometryService, "_locate_star", staticmethod(located))
    output = PhotometryService().run(
        project,
        (10.0, 10.0),
        [(20.0, 20.0), (25.0, 20.0)],
        8.0,
        config=PhotometryConfig(),
    )
    names = {path.name for path in output.parent.iterdir()}
    assert {
        "light_curve_aperture.txt",
        "light_curve_gauss.txt",
        "PHOTOMETRY_APERTURE.txt",
        "PHOTOMETRY_GAUSS.txt",
        "PHOTOMETRY_a.txt",
        "PHOTOMETRY_g.txt",
        "FOV.png",
        "RESULTS.png",
    } <= names
    curve = np.loadtxt(output)
    assert np.allclose(curve[:, 1], 1.0)


def test_hops_geometric_center_moves_only_the_aperture_measurement(monkeypatch) -> None:
    import hops.hops_tools.image_analysis as image_analysis

    data = np.zeros((24, 24), dtype=float)
    data[10, 12] = 100.0
    header = fits.Header({"HOPSMEAN": 0.0, "HOPSSTD": 1.0, "HOPSPSF": 1.0, "HOPSSAT": 1000.0})
    monkeypatch.setattr(
        image_analysis,
        "image_find_stars",
        lambda *args, **kwargs: [[10.0, 10.0, 100.0, 0.0, 1.0, 1.0, 100.0, 100.0, 0.0, 1.0]],
    )

    measurement = PhotometryService._locate_star(
        data,
        header,
        10.0,
        10.0,
        3.0,
        PhotometryConfig(aperture_radius=3.0, geometric_center=True),
    )

    assert measurement["gaussian_x"] == 10.0
    assert measurement["gaussian_y"] == 10.0
    assert measurement["x"] == 12.5
    assert measurement["y"] == 10.5


def test_plate_solver_uses_valid_existing_fits_wcs_without_gaia(
    tmp_path: Path, monkeypatch
) -> None:
    from astropy.wcs import WCS

    import hops.hops_tools.image_analysis as image_analysis

    frame = tmp_path / "wcs.fits"
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [32.0, 32.0]
    wcs.wcs.crval = [268.029125, 37.546153]
    wcs.wcs.cdelt = [-0.34 / 3600.0, 0.34 / 3600.0]
    header = wcs.to_header()
    header["HOPSMEAN"] = 100.0
    header["HOPSSTD"] = 5.0
    header["HOPSPSF"] = 2.0
    fits.writeto(frame, np.full((64, 64), 100.0, dtype=np.float32), header)
    monkeypatch.setattr(
        image_analysis,
        "image_find_stars",
        lambda *args, **kwargs: [[31.0 + index * 0.2, 31.0, 100.0] for index in range(6)],
    )

    result = PlateSolveService().solve(frame, "17:52:06.99", "+37:32:46.15", 0.0)

    assert result.solved
    assert result.attempts[0].detail.startswith("Existing FITS WCS validated")
    assert 0 <= result.target_xy[0] < 64
    assert 0 <= result.target_xy[1] < 64


def test_plate_solver_corrects_header_pointing_offset_with_gaia() -> None:
    import astropy.units as units
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crpix = [128.0, 128.0]
    wcs.wcs.crval = [100.0, 20.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]
    projected = np.vstack(
        (
            np.asarray([[127.0, 127.0]]),
            np.asarray(
                [
                    (x, y)
                    for y in (40, 80, 120, 160, 200)
                    for x in (40, 80, 120, 160, 200)
                ],
                dtype=float,
            ),
        )
    )
    world = np.asarray(wcs.all_pix2world(projected, 0), dtype=float)
    shift = np.asarray([12.0, -18.0])
    detected = projected + shift
    stars = [[x, y] for x, y in detected]
    catalogue = Table({"ra": world[:, 0], "dec": world[:, 1]})
    coordinate = SkyCoord(100.0 * units.deg, 20.0 * units.deg)

    result = PlateSolveService._correct_existing_wcs(
        wcs,
        (256, 256),
        stars,
        catalogue,
        coordinate,
        2.0,
    )

    assert result is not None
    assert result.identified_stars >= 20
    assert np.allclose(result.target_xy, (139.0, 109.0), atol=1.0)
