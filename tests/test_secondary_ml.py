from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from leaps.catalog import PlanetParameters
from leaps.models import StageID
from leaps.project import ProjectWorkspace
from leaps.science import SecondaryEclipseService
from leaps.secondary_ml import SecondaryEclipseMLService, _FixedPhaseEvaluator


def _parameters() -> PlanetParameters:
    return PlanetParameters(
        name="Synthetic ML b",
        ra="12:00:00",
        dec="+20:00:00",
        period=2.0,
        mid_time=2460000.0,
        rp_over_rs=0.1,
        sma_over_rs=8.0,
        inclination=87.0,
        eccentricity=0.0,
        periastron=90.0,
        metallicity=0.0,
        temperature=5700.0,
        logg=4.4,
        source="Synthetic catalog",
    )


def test_fast_ml_evaluator_matches_the_leaps_secondary_fit() -> None:
    parameters = _parameters()
    time = parameters.mid_time + np.linspace(-8.0, 8.0, 8_000) / 24.0
    phase = SecondaryEclipseService._relative_phase(time, parameters.mid_time, parameters.period, 0.5)
    duration_phase = 2.0 / (parameters.period * 24.0)
    template = SecondaryEclipseService._eclipse_template(phase, duration_phase)
    rng = np.random.default_rng(44)
    uncertainty = np.full(time.size, 0.00025)
    flux = 1.0 - 0.00045 * template + rng.normal(0.0, uncertainty[0], time.size)

    expected = SecondaryEclipseService._fit_window(
        phase,
        time,
        flux,
        uncertainty,
        duration_phase=duration_phase,
        window_phase=0.18,
        baseline="linear",
    )
    actual = _FixedPhaseEvaluator(
        phase,
        time,
        uncertainty,
        duration_phase=duration_phase,
        window_phase=0.18,
        baseline="linear",
    ).evaluate(flux)

    for key in (
        "depth",
        "depth_uncertainty",
        "significance",
        "red_noise_beta",
        "residual_rms",
        "delta_chi_squared",
    ):
        assert actual[key] == pytest.approx(expected[key], rel=1e-8, abs=1e-12)


def _write_tess_bounds_file(path: Path, sector: int, time_bjd_tdb: np.ndarray) -> None:
    columns = [
        fits.Column(name="TIME", array=time_bjd_tdb - 2457000.0, format="D"),
        fits.Column(name="QUALITY", array=np.zeros(time_bjd_tdb.size, dtype=np.int32), format="J"),
    ]
    primary = fits.PrimaryHDU()
    primary.header["SECTOR"] = sector
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)


@pytest.mark.filterwarnings("ignore:.*invalid value encountered.*")
def test_ml_validation_uses_disjoint_tess_segments_and_writes_outputs(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    parameters = _parameters()
    rng = np.random.default_rng(3)
    pieces = []
    source_files = []
    duration_phase = 2.0 / (parameters.period * 24.0)
    # Each block brackets an eclipse, rather than starting exactly at one, so
    # the LEAPS local model has before/in/after-event coverage.
    for sector, start in enumerate((2459992.3, 2460012.3, 2460032.3, 2460052.3), start=1):
        time = start + np.arange(1_000) * (2.0 / 1440.0)
        phase = SecondaryEclipseService._relative_phase(time, parameters.mid_time, parameters.period, 0.5)
        template = SecondaryEclipseService._eclipse_template(phase, duration_phase)
        error = np.full(time.size, 0.00035)
        flux = 1.0 - 0.00035 * template + rng.normal(0.0, error[0], time.size)
        pieces.append((time, flux, error))
        source = tmp_path / f"synthetic-s{sector:04d}_lc.fits"
        _write_tess_bounds_file(source, sector, time)
        source_files.append(str(source))

    time = np.concatenate([piece[0] for piece in pieces])
    flux = np.concatenate([piece[1] for piece in pieces])
    error = np.concatenate([piece[2] for piece in pieces])
    project = ProjectWorkspace.create(tmp_path / "synthetic-tess", "Synthetic TESS")
    light_curve = project.outputs_dir / StageID.LIGHT_CURVE.value
    light_curve.mkdir(parents=True)
    time_utc = Time(time, format="jd", scale="tdb").utc.jd
    np.savetxt(light_curve / "light_curve_aperture.txt", np.column_stack((time_utc, flux, error)))
    np.savetxt(light_curve / "light_curve_gauss.txt", np.column_stack((time_utc, flux, error)))
    project.manifest.settings["tess_import"] = {"source_files": source_files}
    project.save()

    result = SecondaryEclipseMLService().run(
        project,
        parameters,
        duration_hours=2.0,
        trials_per_split=80,
        random_seed=12,
    )

    assert result.preview_path.exists()
    assert result.summary_path.exists()
    assert (result.output_path / "ml-trials.csv").exists()
    assert result.train_segments
    assert result.test_segments
    assert set(result.train_segments).isdisjoint(result.test_segments)
    assert 0.0 <= result.test_auc <= 1.0
    assert result.raw["analysis"].startswith("LEAPS secondary-eclipse")
    assert "Positive-depth sector fraction" in result.raw["configuration"]["features"]
    assert "Inter-sector depth scatter" in result.raw["configuration"]["features"]

    import csv

    with (result.output_path / "ml-trials.csv").open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))
    assert 0.0 <= float(first_row["positive_sector_fraction"]) <= 1.0
    assert float(first_row["sector_depth_scatter"]) >= 0.0
