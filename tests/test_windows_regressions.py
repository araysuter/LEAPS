from __future__ import annotations

import json
import mmap
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from PySide6.QtGui import QImage, QPalette

from leaps.diagnostics import DiagnosticLogger
from leaps.models import JobStatus, LEAPSError, StageEvent, StageID, StageState, StageStatus
from leaps.project import ProjectWorkspace
from leaps.science import AlignmentService, InspectionService, LightCurveReviewService
from leaps.ui.main_window import MainWindow
from leaps.ui.pages import DataTargetPage, ProcessingPage


def _alignment_project(root: Path, frame_count: int = 4) -> ProjectWorkspace:
    project = ProjectWorkspace.create(root, "Windows regression")
    reduction = project.outputs_dir / StageID.REDUCTION.value
    reduction.mkdir()
    for index in range(frame_count):
        header = fits.Header(
            {
                "FRAMEIDX": index,
                "HOPSMEAN": 100.0,
                "HOPSSTD": 5.0,
                "HOPSPSF": 2.0,
            }
        )
        fits.writeto(
            reduction / f"r_{index:05d}.fits",
            np.full((32, 32), 100.0 + index, dtype=np.float32),
            header,
        )
    inspection = InspectionService().run(project)
    InspectionService.confirm(
        project,
        {str(record["file"]): False for record in inspection.frames},
    )
    project.set_stage(StageID.INSPECTION, StageStatus.COMPLETE, "Confirmed")
    return project


def _light_curve_project(root: Path) -> ProjectWorkspace:
    project = ProjectWorkspace.create(root, "Windows light-curve review")
    photometry = project.outputs_dir / StageID.PHOTOMETRY.value
    photometry.mkdir()
    target = [100.0, 98.0, 101.0, 99.0]
    comparison_fluxes = (
        [50.0, 50.0, 50.0, 50.0],
        [50.0, 100.0, 50.0, 50.0],
        [50.0, 50.0, 50.0, 50.0],
    )
    rows = []
    for index, measurements in enumerate(
        zip(target, *comparison_fluxes, strict=True)
    ):
        rows.append(
            {
                "file": f"light_{index:03d}.fits",
                "jd": 2461000.0 + index / 1440,
                "measurements": [
                    {
                        "aperture_flux": flux,
                        "aperture_error": 1.0,
                        "gaussian_flux": flux * 0.98,
                        "gaussian_error": 1.1,
                    }
                    for flux in measurements
                ],
            }
        )
    (photometry / "measurements.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    project.manifest.stages[StageID.PHOTOMETRY.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Complete",
    )
    project.manifest.stages[StageID.LIGHT_CURVE.value] = StageState(
        status=StageStatus.READY,
        summary="Review comparison stars",
    )
    project.save()
    return project


def _alignment_stars(index: int) -> list[list[float]]:
    return [
        [10.0 + index + offset * 3.0, 12.0 + index * 0.5 + offset * 2.0, 1000.0]
        for offset in range(8)
    ]


def _alignment_transform(reference: np.ndarray, stars: np.ndarray, **_kwargs) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, float(stars[0, 0] - reference[0, 0])],
            [0.0, 1.0, float(stars[0, 1] - reference[0, 1])],
            [0.0, 0.0, 1.0],
        ]
    )


def _patch_successful_alignment(monkeypatch) -> None:
    import hops.hops_tools.image_analysis as image_analysis
    from hops.thirdparty import twirl

    monkeypatch.setattr(
        image_analysis,
        "image_find_stars",
        lambda _data, header, **_kwargs: _alignment_stars(int(header["FRAMEIDX"])),
    )
    monkeypatch.setattr(twirl.utils, "find_transform", _alignment_transform)


def _is_mmap_backed(data: np.ndarray) -> bool:
    current: object | None = data
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        if isinstance(current, mmap.mmap):
            return True
        visited.add(id(current))
        current = getattr(current, "base", None)
    return False


def test_alignment_releases_fits_mapping_before_updating_headers(
    tmp_path: Path, monkeypatch
) -> None:
    project = _alignment_project(tmp_path)
    reduction = project.outputs_dir / StageID.REDUCTION.value
    before = {path.name: fits.getdata(path, memmap=False).copy() for path in reduction.glob("*.fits")}
    _patch_successful_alignment(monkeypatch)
    original_writeto = fits.writeto

    def reject_windows_style_overwrite(path, data, *args, **kwargs):
        if _is_mmap_backed(data):
            raise PermissionError(13, "Windows cannot replace a memory-mapped FITS file", str(path))
        return original_writeto(path, data, *args, **kwargs)

    monkeypatch.setattr(fits, "writeto", reject_windows_style_overwrite)

    output = AlignmentService().run(project)
    records = json.loads((output / "alignment.json").read_text(encoding="utf-8"))

    assert all(not record.get("failed", False) for record in records)
    for name, expected in before.items():
        path = reduction / name
        actual, header = fits.getdata(path, header=True, memmap=False)
        assert np.array_equal(actual, expected)
        assert all(key in header for key in ("HOPSX0", "HOPSY0", "HOPSU0"))


def test_alignment_requires_two_successes_and_preserves_previous_summary(
    tmp_path: Path, monkeypatch
) -> None:
    import hops.hops_tools.image_analysis as image_analysis
    from hops.thirdparty import twirl

    project = _alignment_project(tmp_path)
    previous = project.outputs_dir / StageID.ALIGNMENT.value
    previous.mkdir()
    previous_records = [
        {"file": "r_00000.fits", "x0": 0.0, "y0": 0.0, "rotation": 0.0},
        {"file": "r_00001.fits", "x0": 1.0, "y0": 0.0, "rotation": 0.0},
    ]
    previous_text = json.dumps(previous_records, indent=2)
    (previous / "alignment.json").write_text(previous_text, encoding="utf-8")

    def only_reference_stars(_data, header, **_kwargs):
        index = int(header["FRAMEIDX"])
        return _alignment_stars(index) if index == 0 else []

    monkeypatch.setattr(image_analysis, "image_find_stars", only_reference_stars)
    monkeypatch.setattr(twirl.utils, "find_transform", _alignment_transform)

    with pytest.raises(LEAPSError) as caught:
        AlignmentService().run(project)

    assert caught.value.code == "ALIGNMENT_INSUFFICIENT_SUCCESSES"
    assert caught.value.stage is StageID.ALIGNMENT
    assert "Successful frames: 1 of 4" in caught.value.technical_details
    assert "ValueError" in caught.value.technical_details
    assert (previous / "alignment.json").read_text(encoding="utf-8") == previous_text
    assert not (project.temporary_dir / "alignment-pending").exists()


def test_partial_alignment_reports_counts_eta_and_failed_reason(
    tmp_path: Path, monkeypatch
) -> None:
    import hops.hops_tools.image_analysis as image_analysis
    from hops.thirdparty import twirl

    project = _alignment_project(tmp_path)

    def one_failed_frame(_data, header, **_kwargs):
        index = int(header["FRAMEIDX"])
        return [] if index == 2 else _alignment_stars(index)

    monkeypatch.setattr(image_analysis, "image_find_stars", one_failed_frame)
    monkeypatch.setattr(twirl.utils, "find_transform", _alignment_transform)
    events: list[StageEvent] = []

    output = AlignmentService().run(project, emit=events.append)
    records = json.loads((output / "alignment.json").read_text(encoding="utf-8"))

    assert len(AlignmentService.successful_frames(project)) == 3
    assert records[2]["failed"] is True
    assert records[2]["error_type"] == "ValueError"
    running = [event for event in events if event.status == JobStatus.RUNNING]
    assert [event.current for event in running] == [1, 2, 3, 4]
    assert all("workers" in event.details for event in running)
    assert all("success_count" in event.details for event in running)
    assert all("failure_count" in event.details for event in running)
    assert all("eta_seconds" in event.details for event in running)
    completed = events[-1]
    assert completed.status == JobStatus.SUCCEEDED
    assert completed.details["success_count"] == 3
    assert completed.details["failure_count"] == 1
    assert completed.details["eta_seconds"] == 0.0
    assert "3 aligned" in completed.message
    assert "1 skipped" in completed.message


def test_invalid_completed_alignment_recovers_without_deleting_outputs(qapp, tmp_path: Path) -> None:
    project = _alignment_project(tmp_path, frame_count=3)
    alignment = project.outputs_dir / StageID.ALIGNMENT.value
    alignment.mkdir()
    invalid_records = [
        {
            "file": f"r_{index:05d}.fits",
            "failed": True,
            "error_type": "PermissionError",
            "reason": "The process cannot access the file because it is being used by another process",
        }
        for index in range(3)
    ]
    (alignment / "alignment.json").write_text(json.dumps(invalid_records), encoding="utf-8")
    project.manifest.stages[StageID.ALIGNMENT.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Complete",
    )
    for stage in (
        StageID.PHOTOMETRY,
        StageID.LIGHT_CURVE,
        StageID.FITTING,
        StageID.SECONDARY_ECLIPSE,
    ):
        project.manifest.stages[stage.value] = StageState(
            status=StageStatus.COMPLETE,
            summary="Complete",
        )
    for key in (
        "plate_solution",
        "photometry",
        "light_curve_review",
        "fitting_setup",
        "secondary_eclipse_setup",
    ):
        project.manifest.settings[key] = {"stale": True}
    project.save()
    window = MainWindow(demo=True)

    window.set_project(project)

    state = project.manifest.stages[StageID.ALIGNMENT.value]
    assert state.status == StageStatus.NEEDS_ATTENTION
    assert state.summary == "Rerun Alignment"
    assert "ALIGNMENT_RESULT_INVALID" in state.warning_codes
    for stage in (
        StageID.PHOTOMETRY,
        StageID.LIGHT_CURVE,
        StageID.FITTING,
        StageID.SECONDARY_ECLIPSE,
    ):
        assert project.manifest.stages[stage.value].status == StageStatus.LOCKED
    assert not any(key in project.manifest.settings for key in ("plate_solution", "photometry"))
    assert (alignment / "alignment.json").is_file()
    assert len(list((project.outputs_dir / StageID.REDUCTION.value).glob("*.fits"))) == 3
    window.close()


def test_tess_import_is_not_treated_as_invalid_ground_alignment(qapp, tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TESS import")
    project.manifest.settings["tess_import"] = {"product": "test"}
    project.manifest.stages[StageID.ALIGNMENT.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Not applicable",
    )
    project.manifest.stages[StageID.PHOTOMETRY.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Imported",
    )
    project.save()
    window = MainWindow(demo=True)

    window.set_project(project)

    state = project.manifest.stages[StageID.ALIGNMENT.value]
    assert state.status == StageStatus.COMPLETE
    assert state.summary == "Not applicable"
    assert "ALIGNMENT_RESULT_INVALID" not in state.warning_codes
    assert MainWindow._photometry_reference_frame(project) is None
    window.close()


def test_backtracked_light_curve_reapproval_reopens_fitting_with_locked_old_output(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    project = _light_curve_project(tmp_path)
    service = LightCurveReviewService()
    first_output = service.commit(project, [True, True, True])
    first_curve = np.loadtxt(first_output)
    project.set_stage(
        StageID.LIGHT_CURVE,
        StageStatus.COMPLETE,
        "3 comparisons approved",
        output_path=first_output,
    )
    fitting_output = project.outputs_dir / StageID.FITTING.value
    fitting_output.mkdir()
    (fitting_output / "fit-summary.json").write_text("{}", encoding="utf-8")
    project.manifest.stages[StageID.FITTING.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Complete",
    )
    project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Complete",
    )
    project.save()

    window = MainWindow(demo=True)
    window.set_project(project)
    window.open_stage(StageID.LIGHT_CURVE)
    failures: list[LEAPSError] = []
    monkeypatch.setattr(window, "_show_failure", failures.append)
    monkeypatch.setattr(window, "prepare_fitting_setup", lambda **_kwargs: None)
    original_rmtree = shutil.rmtree

    def hold_windows_output_lock(path, *args, **kwargs):
        if Path(path).name.startswith("light_curve-previous"):
            raise PermissionError(13, "The process cannot access the file", str(path))
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("leaps.project.shutil.rmtree", hold_windows_output_lock)

    window.confirm_light_curve_review([False, False, True])

    updated_curve = np.loadtxt(
        project.outputs_dir / StageID.LIGHT_CURVE.value / "light_curve_aperture.txt"
    )
    assert not failures
    assert not np.allclose(updated_curve[:, 1], first_curve[:, 1])
    assert project.manifest.settings["light_curve_review"]["active_comparisons"] == [
        False,
        False,
        True,
    ]
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.READY
    assert project.manifest.stages[StageID.FITTING.value].summary == (
        "Ready · previous result preserved"
    )
    assert project.manifest.stages[StageID.SECONDARY_ECLIPSE.value].status == (
        StageStatus.LOCKED
    )
    assert window.stack.currentWidget() is window.fitting_page
    assert list(project.outputs_dir.glob("light_curve-previous*"))
    window.close()


@pytest.mark.parametrize("cancelled", [False, True])
def test_failed_alignment_rerun_restores_previous_completed_state(
    qapp, tmp_path: Path, monkeypatch, cancelled: bool
) -> None:
    project = _alignment_project(tmp_path, frame_count=3)
    _patch_successful_alignment(monkeypatch)
    output = AlignmentService().run(project)
    project.set_stage(
        StageID.ALIGNMENT,
        StageStatus.COMPLETE,
        "3 aligned",
        progress=1.0,
        output_path=output,
    )
    previous_text = (output / "alignment.json").read_text(encoding="utf-8")
    window = MainWindow(demo=True)
    window.set_project(project)
    previous = project.manifest.stages[StageID.ALIGNMENT.value]
    window._alignment_previous_state = StageState(
        status=previous.status,
        summary=previous.summary,
        progress=previous.progress,
        output_path=previous.output_path,
        warning_codes=list(previous.warning_codes),
    )
    project.set_stage(StageID.ALIGNMENT, StageStatus.RUNNING, "Processing", progress=0.0)
    monkeypatch.setattr(window, "_show_failure", lambda _failure: None)
    failure = LEAPSError(
        "JOB_CANCELLED" if cancelled else "ALIGNMENT_INSUFFICIENT_SUCCESSES",
        "Alignment cancelled" if cancelled else "Alignment failed",
        "The previous successful result should remain active.",
        ["Rerun Alignment"],
        stage=StageID.ALIGNMENT,
    )

    window._stage_failed(StageID.ALIGNMENT, failure)

    restored = project.manifest.stages[StageID.ALIGNMENT.value]
    assert restored.status == StageStatus.COMPLETE
    assert restored.summary == "3 aligned"
    assert restored.progress == 1.0
    assert (output / "alignment.json").read_text(encoding="utf-8") == previous_text
    window.close()


def test_valid_alignment_keeps_manual_target_fallback_available(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    project = _alignment_project(tmp_path, frame_count=3)
    _patch_successful_alignment(monkeypatch)
    AlignmentService().run(project)
    project.set_stage(StageID.ALIGNMENT, StageStatus.COMPLETE, "3 aligned")
    window = MainWindow(demo=True)
    window.set_project(project)
    failure = LEAPSError(
        "PLATE_SOLVE_FAILED",
        "Plate solving was unavailable",
        "Place the target manually.",
        ["Select target manually"],
        stage=StageID.PHOTOMETRY,
    )

    window._plate_failed(failure)
    window.plate_page.inspector.manual.click()
    qapp.processEvents()

    frame, target = window._photometry_inputs(require_target=False)
    assert frame.name == "r_00000.fits"
    assert target is None
    assert window.plate_page.workspace.image.mode == "select"
    assert window.plate_page.workspace.image.selection_role == "target"
    assert project.manifest.stages[StageID.PHOTOMETRY.value].status == StageStatus.READY
    window.close()


def test_filter_popup_has_explicit_dark_rendering_and_visible_selection(qapp) -> None:
    assert "QComboBox QAbstractItemView" in qapp.styleSheet()
    page = DataTargetPage()
    page.resize(900, 800)
    page.show()
    page.filter.setCurrentIndex(1)
    page.filter.showPopup()
    qapp.processEvents()
    view = page.filter.view()
    image = view.grab().toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    pixels = np.frombuffer(image.bits(), dtype=np.uint8, count=image.sizeInBytes()).reshape(
        image.height(), image.width(), 4
    )
    palette = view.palette()

    assert page.filter.currentText() == "Clear"
    assert all(page.filter.itemText(index) for index in range(page.filter.count()))
    assert np.median(pixels[:, :, :3]) < 180
    assert palette.color(QPalette.ColorRole.Base).lightnessF() < 0.35
    assert palette.color(QPalette.ColorRole.Text).lightnessF() > 0.65
    assert palette.color(QPalette.ColorRole.Highlight).isValid()
    assert palette.color(QPalette.ColorRole.HighlightedText).isValid()
    page.filter.hidePopup()
    page.close()


def test_alignment_processing_page_shows_counts_and_eta(qapp) -> None:
    page = ProcessingPage(StageID.ALIGNMENT, "Alignment", "Register frames", [])
    page.update_event(
        StageEvent(
            StageID.ALIGNMENT,
            JobStatus.RUNNING,
            "Aligned r_00003.fits",
            3,
            12,
            details={
                "workers": 4,
                "success_count": 2,
                "failure_count": 1,
                "eta_seconds": 125,
            },
        )
    )

    assert "3 of 12" in page.counter.text()
    assert "2 aligned · 1 skipped" in page.counter.text()
    assert "about 2m 05s remaining" in page.counter.text()
    page.close()


def test_diagnostic_bundle_includes_alignment_state_but_not_fits_pixels(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Diagnostic alignment")
    alignment = project.outputs_dir / StageID.ALIGNMENT.value
    alignment.mkdir()
    records = [
        {
            "file": "r_00000.fits",
            "failed": True,
            "error_type": "PermissionError",
            "reason": "file is locked",
        }
    ]
    (alignment / "alignment.json").write_text(json.dumps(records), encoding="utf-8")
    logger = DiagnosticLogger(project)
    logger.record("test_event")

    output = logger.export_bundle(tmp_path / "diagnostics.zip")

    with zipfile.ZipFile(output) as archive:
        assert "state/alignment.json" in archive.namelist()
        assert json.loads(archive.read("state/alignment.json")) == records
        assert not any(name.endswith((".fits", ".fit", ".fts")) for name in archive.namelist())
