from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import numpy as np
import pytest

from leaps.models import LEAPSError, StageID, StageState, StageStatus
from leaps.project import ProjectWorkspace
from leaps.science import LightCurveReviewService
from leaps.ui.main_window import MainWindow


def _completed_project(root: Path) -> ProjectWorkspace:
    project = ProjectWorkspace.create(root, "Comparison-star reapproval")
    photometry = project.outputs_dir / StageID.PHOTOMETRY.value
    photometry.mkdir()
    target = [100.0, 98.0, 101.0, 99.0]
    comparison_fluxes = (
        [50.0, 50.0, 50.0, 50.0],
        [50.0, 100.0, 50.0, 50.0],
        [50.0, 50.0, 50.0, 50.0],
    )
    rows = []
    for index, measurements in enumerate(zip(target, *comparison_fluxes, strict=True)):
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
    (photometry / "measurements.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    project.manifest.stages[StageID.PHOTOMETRY.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Complete",
        progress=1.0,
    )
    project.manifest.stages[StageID.LIGHT_CURVE.value] = StageState(
        status=StageStatus.READY,
        summary="Review comparison stars",
    )
    project.save()

    # Do not depend on commit's return type: the approved output location is
    # part of the workspace contract used by Fitting.
    LightCurveReviewService().commit(project, [True, True, True])
    approved_curve = project.outputs_dir / StageID.LIGHT_CURVE.value / "light_curve_aperture.txt"
    project.manifest.stages[StageID.LIGHT_CURVE.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="3 comparisons approved",
        progress=1.0,
        output_path=project.relative(approved_curve),
    )

    fitting = project.outputs_dir / StageID.FITTING.value
    fitting.mkdir()
    (fitting / "last-success.txt").write_text("previous fitting result", encoding="utf-8")
    project.manifest.stages[StageID.FITTING.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Full fit complete",
        progress=1.0,
        checkpoint="complete",
        output_path=project.relative(fitting),
    )
    project.manifest.settings["fitting_setup"] = {
        "light_curve": "gaussian",
        "detrending": "quadratic",
        "sentinel": "preserve these choices",
    }

    eclipse = project.outputs_dir / StageID.SECONDARY_ECLIPSE.value
    eclipse.mkdir()
    (eclipse / "last-success.txt").write_text("previous secondary-eclipse result", encoding="utf-8")
    project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Eclipse analysis complete",
        progress=1.0,
        checkpoint="complete",
        output_path=project.relative(eclipse),
    )
    project.manifest.settings["secondary_eclipse_setup"] = {"sentinel": "stale after the light curve changes"}
    project.save()
    return project


def _file_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _open_light_curve(
    project: ProjectWorkspace, monkeypatch: pytest.MonkeyPatch
) -> tuple[MainWindow, list[LEAPSError]]:
    window = MainWindow(demo=True)
    window.set_project(project)
    window.open_stage(StageID.LIGHT_CURVE)
    failures: list[LEAPSError] = []
    monkeypatch.setattr(window, "_show_failure", failures.append)
    # Reapproval should open Fitting without starting catalog/FITS work. This
    # also keeps the assertion focused on the coordinated confirmation save.
    monkeypatch.setattr(window, "prepare_fitting_setup", lambda **_kwargs: None)
    return window, failures


def _windows_lock(winerror: int, path: Path) -> OSError:
    error = OSError(errno.EACCES, "The process cannot access the file", str(path))
    error.winerror = winerror
    return error


def test_changed_selection_retries_onedrive_lock_and_persists_invalidated_states(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _completed_project(tmp_path)
    old_curve = np.loadtxt(project.outputs_dir / StageID.LIGHT_CURVE.value / "light_curve_aperture.txt")
    fitting_setup = dict(project.manifest.settings["fitting_setup"])
    window, failures = _open_light_curve(project, monkeypatch)
    stale_preview = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-mark.png"
    window.fitting_page.show_preview(
        stale_preview,
        planet="Previous fit",
        passband="COUSINS_R",
        residual_std=0.002,
    )
    assert window.fitting_page.view_in_files.isEnabled()

    original_replace = os.replace
    attempts: list[Path] = []
    delays: list[float] = []

    def replace_after_transient_lock(source: str | Path, destination: str | Path) -> None:
        destination_path = Path(destination)
        if destination_path == project.manifest_path:
            attempts.append(Path(source))
            if len(attempts) <= 2:
                raise _windows_lock(32, project.manifest_path)
        original_replace(source, destination)

    monkeypatch.setattr("leaps.models.os.replace", replace_after_transient_lock)
    monkeypatch.setattr("leaps.models.time.sleep", delays.append)

    window.confirm_light_curve_review([False, False, True])

    new_curve = np.loadtxt(project.outputs_dir / StageID.LIGHT_CURVE.value / "light_curve_aperture.txt")
    assert failures == []
    assert len(attempts) == 3
    assert len(set(attempts)) == 1
    assert delays == [0.05, 0.1]
    assert not np.allclose(new_curve[:, 1], old_curve[:, 1])
    assert project.manifest.settings["light_curve_review"]["active_comparisons"] == [False, False, True]
    assert project.manifest.settings["fitting_setup"] == fitting_setup
    assert "secondary_eclipse_setup" not in project.manifest.settings

    light_curve = project.manifest.stages[StageID.LIGHT_CURVE.value]
    fitting = project.manifest.stages[StageID.FITTING.value]
    eclipse = project.manifest.stages[StageID.SECONDARY_ECLIPSE.value]
    assert light_curve.status == StageStatus.COMPLETE
    assert light_curve.summary == "1 comparisons approved"
    assert fitting == StageState(
        status=StageStatus.READY,
        summary="Light curve changed · run Preview Fit",
        checkpoint="light_curve_changed",
        updated_at=fitting.updated_at,
    )
    assert eclipse.status == StageStatus.LOCKED
    assert eclipse.summary == "Locked"
    assert (project.outputs_dir / StageID.FITTING.value / "last-success.txt").is_file()
    assert (project.outputs_dir / StageID.SECONDARY_ECLIPSE.value / "last-success.txt").is_file()
    assert window.stack.currentWidget() is window.fitting_page
    assert window.fitting_page._preview_path is None
    assert not window.fitting_page.view_in_files.isEnabled()
    assert not window.fitting_page.full.isEnabled()

    reopened = ProjectWorkspace.open(project.root)
    assert reopened.manifest.settings["light_curve_review"]["active_comparisons"] == [False, False, True]
    assert reopened.manifest.settings["fitting_setup"] == fitting_setup
    assert "secondary_eclipse_setup" not in reopened.manifest.settings
    assert reopened.manifest.stages[StageID.FITTING.value].status == StageStatus.READY
    assert reopened.manifest.stages[StageID.FITTING.value].output_path is None
    assert reopened.manifest.stages[StageID.SECONDARY_ECLIPSE.value].status == StageStatus.LOCKED
    assert not list(project.workspace.glob("project.json.*.tmp"))
    window.close()


def test_unchanged_selection_is_a_zero_write_noop(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _completed_project(tmp_path)
    manifest_before = project.manifest_path.read_bytes()
    curve_dir = project.outputs_dir / StageID.LIGHT_CURVE.value
    curves_before = _file_snapshot(curve_dir)
    window, failures = _open_light_curve(project, monkeypatch)
    stale_preview = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-mark.png"
    window.fitting_page.show_preview(
        stale_preview,
        planet="Previous fit",
        passband="COUSINS_R",
        residual_std=None,
    )
    commits: list[list[bool]] = []
    saves: list[None] = []

    def unexpected_commit(
        _service: LightCurveReviewService,
        _project: ProjectWorkspace,
        active_comparisons: list[bool],
        **_kwargs,
    ) -> None:
        commits.append(active_comparisons)
        raise AssertionError("unchanged selection regenerated the light curve")

    monkeypatch.setattr(LightCurveReviewService, "commit", unexpected_commit)
    monkeypatch.setattr(project, "save", lambda: saves.append(None))

    window.confirm_light_curve_review([True, True, True])

    assert commits == []
    assert saves == []
    assert failures == []
    assert project.manifest_path.read_bytes() == manifest_before
    assert _file_snapshot(curve_dir) == curves_before
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.COMPLETE
    assert project.manifest.stages[StageID.SECONDARY_ECLIPSE.value].status == StageStatus.COMPLETE
    assert "secondary_eclipse_setup" in project.manifest.settings
    assert window.fitting_page._preview_path == stale_preview
    assert window.fitting_page.view_in_files.isEnabled()
    window.close()


def test_unchanged_selection_regenerates_when_approved_output_is_invalid(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _completed_project(tmp_path)
    output = project.outputs_dir / StageID.LIGHT_CURVE.value
    (output / "review.json").write_text(
        json.dumps({"active_comparisons": [False, False, True]}),
        encoding="utf-8",
    )
    window, failures = _open_light_curve(project, monkeypatch)

    window.confirm_light_curve_review([True, True, True])

    assert failures == []
    review = json.loads((output / "review.json").read_text(encoding="utf-8"))
    assert review["active_comparisons"] == [True, True, True]
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.READY
    assert (
        project.manifest.stages[StageID.FITTING.value].checkpoint
        == "light_curve_changed"
    )
    assert window.stack.currentWidget() is window.fitting_page
    window.close()


def test_permanent_manifest_lock_restores_previous_curve_and_manifest(
    qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _completed_project(tmp_path)
    manifest_before = project.manifest_path.read_bytes()
    manifest_values_before = json.loads(manifest_before)
    curve_dir = project.outputs_dir / StageID.LIGHT_CURVE.value
    curves_before = _file_snapshot(curve_dir)
    window, failures = _open_light_curve(project, monkeypatch)
    attempts: list[Path] = []
    delays: list[float] = []
    original_replace = os.replace

    def keep_manifest_locked(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == project.manifest_path:
            attempts.append(Path(source))
            raise _windows_lock(5, project.manifest_path)
        original_replace(source, destination)

    monkeypatch.setattr("leaps.models.os.replace", keep_manifest_locked)
    monkeypatch.setattr("leaps.models.time.sleep", delays.append)

    window.confirm_light_curve_review([False, False, True])

    assert len(failures) == 1
    assert failures[0].code == "PROJECT_MANIFEST_SAVE_BLOCKED"
    assert "OneDrive" in failures[0].message
    assert len(attempts) == 6
    assert len(set(attempts)) == 1
    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8]
    assert project.manifest_path.read_bytes() == manifest_before
    assert project.manifest.to_dict() == manifest_values_before
    assert _file_snapshot(curve_dir) == curves_before
    assert project.manifest.stages[StageID.FITTING.value].status == StageStatus.COMPLETE
    assert project.manifest.stages[StageID.SECONDARY_ECLIPSE.value].status == StageStatus.COMPLETE
    assert project.manifest.settings["light_curve_review"]["active_comparisons"] == [True, True, True]
    assert "secondary_eclipse_setup" in project.manifest.settings
    assert not (project.temporary_dir / "light_curve-pending").exists()
    assert not list(project.outputs_dir.glob("light_curve-previous*"))
    assert not list(project.workspace.glob("project.json.*.tmp"))
    assert window.stack.currentWidget() is window.light_curve_page
    monkeypatch.setattr("leaps.models.os.replace", original_replace)
    window.close()


@pytest.mark.parametrize("secondary_status", [StageStatus.READY, StageStatus.LOCKED])
def test_stale_secondary_result_is_not_loaded_until_its_stage_is_complete(
    qapp,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    secondary_status: StageStatus,
) -> None:
    project = ProjectWorkspace.create(tmp_path, "Stale eclipse result")
    project.manifest.stages[StageID.FITTING.value] = StageState(
        status=StageStatus.COMPLETE,
        summary="Full fit complete",
    )
    project.manifest.stages[StageID.SECONDARY_ECLIPSE.value] = StageState(
        status=secondary_status,
        summary="Ready" if secondary_status == StageStatus.READY else "Locked",
    )
    fitting = project.outputs_dir / StageID.FITTING.value
    fitting.mkdir()
    (fitting / "fit-summary.json").write_text(
        json.dumps(
            {
                "passband": "COUSINS_R",
                "light_curve": "gaussian",
                "parameters": {
                    "name": "WASP-18 b",
                    "ra": "01:37:25.07",
                    "dec": "-45:40:40.10",
                    "period": 0.941452,
                    "mid_time": 2458354.458,
                    "rp_over_rs": 0.1018,
                    "sma_over_rs": 3.48,
                    "inclination": 83.5,
                    "eccentricity": 0.0,
                    "periastron": 0.0,
                    "metallicity": 0.0,
                    "temperature": 6432.0,
                    "logg": 4.31,
                    "source": "Test",
                },
                "fitted_ephemeris": {
                    "period": 0.941452,
                    "mid_time": 2458354.458,
                },
            }
        ),
        encoding="utf-8",
    )
    eclipse = project.outputs_dir / StageID.SECONDARY_ECLIPSE.value
    eclipse.mkdir()
    (eclipse / "secondary-eclipse.json").write_text(
        json.dumps({"message": "stale result must remain hidden"}),
        encoding="utf-8",
    )
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)
    shown: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        window.secondary_eclipse_page,
        "show_saved_result",
        lambda *args: shown.append(args),
    )

    window.prepare_secondary_eclipse_setup()

    assert shown == []
    assert not window.secondary_eclipse_page._result_valid
    assert not window.secondary_eclipse_page.view_in_files.isEnabled()
    assert (eclipse / "secondary-eclipse.json").is_file()
    window.close()
