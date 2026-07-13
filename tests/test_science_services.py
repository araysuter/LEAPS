from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

from leaps.models import LEAPSError
from leaps.project import ProjectWorkspace
from leaps.science import CancellationToken, ReductionConfig, ReductionService


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
