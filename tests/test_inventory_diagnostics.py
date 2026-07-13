from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from leaps.diagnostics import DiagnosticLogger
from leaps.fits_inventory import FITSInventory, target_from_header, validate_coordinates
from leaps.models import LEAPSError, StageID
from leaps.project import ProjectWorkspace


def _write_fits(path: Path, image_type: str, exposure: float = 30.0) -> None:
    header = fits.Header()
    header["IMAGETYP"] = image_type
    header["EXPTIME"] = exposure
    header["OBSERVER"] = "Private Name"
    fits.writeto(path, np.arange(64, dtype=np.uint16).reshape(8, 8), header=header)


def test_inventory_reads_headers_only_and_groups_frames(tmp_path: Path) -> None:
    _write_fits(tmp_path / "light_001.fits", "Light Frame")
    _write_fits(tmp_path / "master_dark.fit", "Dark Frame")
    _write_fits(tmp_path / "flat_001.fts", "Flat Field")
    records = FITSInventory(tmp_path).discover()
    grouped = FITSInventory.group(records)
    assert len(grouped["science"]) == 1
    assert len(grouped["dark"]) == 1
    assert len(grouped["flat"]) == 1
    assert all(record.shape == (8, 8) for record in records)


def test_coordinates_are_validated_without_requiring_a_name() -> None:
    ra, dec = validate_coordinates("19:34:55.87", "+36:48:55.79")
    assert ra.startswith("19:34:55")
    assert dec.startswith("+36:48:55")
    with pytest.raises(LEAPSError) as error:
        validate_coordinates("not-ra", "not-dec")
    assert error.value.code == "INVALID_COORDINATES"


def test_target_coordinates_are_normalized_from_common_fits_headers() -> None:
    name, ra, dec = target_from_header({"OBJECT": "TrES-3 b", "RA": "17:52:06.998", "DEC": "+37:32:46.195"})
    assert name == "TrES-3 b"
    assert ra.startswith("17:52:07")
    assert dec.startswith("+37:32:46")

    _, wcs_ra, wcs_dec = target_from_header({"CRVAL1": 268.029161, "CRVAL2": 37.546166})
    assert wcs_ra.startswith("17:52:07")
    assert wcs_dec.startswith("+37:32:46")


def test_redacted_diagnostics_contains_headers_but_never_pixels(tmp_path: Path) -> None:
    raw = tmp_path / "light.fits"
    _write_fits(raw, "Light Frame")
    project = ProjectWorkspace.create(tmp_path)
    logger = DiagnosticLogger(project)
    logger.record("test_event")
    output = logger.export_bundle(tmp_path / "diagnostics.zip", [raw])
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert "project.json" in names
        assert "logs/leaps.jsonl" in names
        assert "headers/header-1.json" in names
        assert not any(name.endswith((".fits", ".fit", ".fts")) for name in names)
        header = json.loads(archive.read("headers/header-1.json"))
        assert "OBSERVER" not in header
        assert header["IMAGETYP"] == "Light Frame"


def test_typed_failure_is_logged_once_with_its_stage(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path)
    logger = DiagnosticLogger(project)
    failure = LEAPSError(
        "CALIBRATION_FRAME_UNREADABLE",
        "A bias frame could not be read",
        "Review the calibration assignment.",
        ["Review frame assignments"],
        stage=StageID.REDUCTION,
    )

    assert logger.failure(failure, StageID.REDUCTION) is failure
    event = json.loads(logger.path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["event"] == "failure"
    assert event["stage"] == "reduction"
    assert event["code"] == "CALIBRATION_FRAME_UNREADABLE"
    assert event["diagnostic_id"] == failure.diagnostic_id
