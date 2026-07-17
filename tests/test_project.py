from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from leaps.models import LEAPSError, ProjectManifest, StageID, StageStatus
from leaps.project import ProjectWorkspace


def test_manifest_round_trip_uses_versioned_plain_values(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="Test transit")
    manifest.stages[StageID.DATA_TARGET.value].status = StageStatus.COMPLETE
    manifest.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 2
    assert payload["stages"]["data_target"]["status"] == "complete"
    restored = ProjectManifest.load(path)
    assert restored.stages["data_target"].status is StageStatus.COMPLETE


def test_project_paths_remain_relative_and_unlock_next_stage(tmp_path: Path) -> None:
    raw = tmp_path / "science" / "light_001.fits"
    raw.parent.mkdir()
    raw.touch()
    project = ProjectWorkspace.create(tmp_path, "Portable run")
    project.manifest.raw_files["science"] = [project.relative(raw)]
    project.set_stage(StageID.DATA_TARGET, StageStatus.COMPLETE, "Target selected")
    assert project.manifest.raw_files["science"] == ["science/light_001.fits"]
    assert project.resolve("science/light_001.fits") == raw
    assert project.manifest.stages[StageID.REDUCTION.value].status is StageStatus.READY
    assert project.workspace == tmp_path / "LEAPS"
    assert project.manifest_path == tmp_path / "LEAPS" / "project.json"


def test_open_project_discards_stale_appledouble_raw_references(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "External drive")
    project.manifest.raw_files["bias"] = ["._bias_001.fits", "bias_001.fits"]
    project.save()

    reopened = ProjectWorkspace.open(tmp_path)

    assert reopened.manifest.raw_files["bias"] == ["bias_001.fits"]
    assert reopened.manifest.warnings[-1]["code"] == "APPLEDOUBLE_FILES_IGNORED"


def test_open_project_accepts_appledouble_companions_for_generated_entries(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External drive")
    for name in (
        "._.DS_Store",
        "._cache",
        "._checkpoints",
        "._logs",
        "._outputs",
        "._project.json",
        "._tmp",
    ):
        (project.workspace / name).write_bytes(b"macOS metadata")

    reopened = ProjectWorkspace.open(tmp_path)

    assert reopened.manifest.project_id == project.manifest.project_id


def test_open_project_accepts_strict_generated_manifest_temporary_names(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path, "Interrupted save")
    token = "0123456789abcdef0123456789abcdef"
    (project.workspace / f"project.json.{token}.tmp").write_text(
        "staged manifest", encoding="utf-8"
    )
    (project.workspace / f"._project.json.{token}.tmp").write_bytes(
        b"macOS metadata"
    )

    reopened = ProjectWorkspace.open(tmp_path)

    assert reopened.manifest.project_id == project.manifest.project_id


def test_open_project_rejects_similarly_named_user_manifest_temporary(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path, "Unrelated file")
    unrelated = project.workspace / "project.json.observer-notes.tmp"
    unrelated.write_text("do not delete", encoding="utf-8")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert unrelated.read_text(encoding="utf-8") == "do not delete"


def test_open_project_rejects_appledouble_companion_for_unrelated_entry(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External drive")
    unrelated = project.workspace / "._observer-notes.txt"
    unrelated.write_bytes(b"macOS metadata for unrelated data")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert unrelated.is_file()


def test_process_access_preflight_reads_raw_files_and_cleans_write_probe(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "science.fits"
    raw.write_bytes(b"SIMPLE  =")
    project = ProjectWorkspace.create(tmp_path, "Accessible run")
    project.manifest.raw_files["science"] = [raw.name]

    project.verify_process_access(StageID.REDUCTION)

    assert not list(project.temporary_dir.glob(".access-check-*"))


def test_process_access_preflight_reports_permission_denied_for_raw_read(
    tmp_path: Path, monkeypatch
) -> None:
    raw = tmp_path / "science.fits"
    raw.write_bytes(b"SIMPLE  =")
    project = ProjectWorkspace.create(tmp_path, "Denied run")
    project.manifest.raw_files["science"] = [raw.name]
    original_open = Path.open

    def deny_raw(path: Path, *args, **kwargs):
        if path == raw and args and args[0] == "rb":
            raise PermissionError(13, "Permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_raw)

    with pytest.raises(LEAPSError) as error:
        project.verify_process_access(StageID.REDUCTION)

    assert error.value.code == "PROJECT_STORAGE_ACCESS_DENIED"
    assert error.value.stage is StageID.REDUCTION
    assert "read" in error.value.technical_details.casefold()


def test_process_access_preflight_reports_permission_denied_for_workspace_write(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "Read-only run")
    original_open = Path.open

    def deny_probe(path: Path, *args, **kwargs):
        if path.name.startswith(".access-check-"):
            raise PermissionError(13, "Permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_probe)

    with pytest.raises(LEAPSError) as error:
        project.verify_process_access(StageID.ALIGNMENT)

    assert error.value.code == "PROJECT_STORAGE_ACCESS_DENIED"
    assert error.value.stage is StageID.ALIGNMENT
    assert "write" in error.value.technical_details.casefold()


def test_reduced_fits_files_ignore_external_drive_appledouble_sidecars(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External run")
    reduction = project.outputs_dir / StageID.REDUCTION.value
    reduction.mkdir()
    reduced = reduction / "r_00001.fits"
    reduced.write_bytes(b"SIMPLE  =")
    (reduction / "._r_00001.fits").write_bytes(b"\x00\x05\x16\x07Mac OS X")

    assert project.reduced_fits_files(StageID.INSPECTION) == [reduced]
    project.verify_process_access(StageID.INSPECTION)


def test_process_access_preflight_reports_permission_denied_for_reduced_frame(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External run")
    reduction = project.outputs_dir / StageID.REDUCTION.value
    reduction.mkdir()
    reduced = reduction / "r_00001.fits"
    reduced.write_bytes(b"SIMPLE  =")
    original_open = Path.open

    def deny_reduced(path: Path, *args, **kwargs):
        if path == reduced and args and args[0] == "rb":
            raise PermissionError(1, "Operation not permitted")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_reduced)

    with pytest.raises(LEAPSError) as error:
        project.verify_process_access(StageID.INSPECTION)

    assert error.value.code == "PROJECT_STORAGE_ACCESS_DENIED"
    assert error.value.stage is StageID.INSPECTION
    assert str(reduced) in error.value.technical_details


def test_transaction_replaces_output_only_after_commit(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.REDUCTION.value
    target.mkdir()
    (target / "old.txt").write_text("last success")
    pending, resolved_target = project.begin_transaction(StageID.REDUCTION)
    (pending / "new.txt").write_text("new success")
    assert (target / "old.txt").read_text() == "last success"
    project.commit_transaction(pending, resolved_target)
    assert not (target / "old.txt").exists()
    assert (target / "new.txt").read_text() == "new success"


def test_transaction_finalize_failure_restores_previous_output(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.LIGHT_CURVE.value
    target.mkdir()
    (target / "curve.txt").write_text("last successful", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.LIGHT_CURVE)
    (pending / "curve.txt").write_text("new approval", encoding="utf-8")

    def reject_finalize() -> None:
        raise LEAPSError(
            "PROJECT_MANIFEST_SAVE_BLOCKED",
            "The project could not be saved",
            "OneDrive retained the manifest lock.",
            ["Retry"],
        )

    with pytest.raises(LEAPSError) as error:
        project.commit_transaction(
            pending,
            resolved_target,
            finalize=reject_finalize,
        )

    assert error.value.code == "PROJECT_MANIFEST_SAVE_BLOCKED"
    assert (target / "curve.txt").read_text(encoding="utf-8") == "last successful"
    assert (pending / "curve.txt").read_text(encoding="utf-8") == "new approval"
    assert not list(project.outputs_dir.glob("light_curve-previous*"))


def test_transaction_finalize_failure_without_previous_output_removes_target(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    pending, target = project.begin_transaction(StageID.LIGHT_CURVE)
    (pending / "curve.txt").write_text("uncommitted approval", encoding="utf-8")

    def reject_finalize() -> None:
        raise RuntimeError("manifest save failed")

    with pytest.raises(RuntimeError, match="manifest save failed"):
        project.commit_transaction(
            pending,
            target,
            finalize=reject_finalize,
        )

    assert not target.exists()
    assert (pending / "curve.txt").read_text(encoding="utf-8") == "uncommitted approval"
    assert not list(project.outputs_dir.glob("light_curve-previous*"))


def test_transaction_discards_previous_only_after_finalize_succeeds(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.LIGHT_CURVE.value
    target.mkdir()
    (target / "curve.txt").write_text("last successful", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.LIGHT_CURVE)
    (pending / "curve.txt").write_text("new approval", encoding="utf-8")
    observed: list[str] = []

    def finalize() -> None:
        observed.append((target / "curve.txt").read_text(encoding="utf-8"))
        assert list(project.outputs_dir.glob("light_curve-previous*"))

    project.commit_transaction(pending, resolved_target, finalize=finalize)

    assert observed == ["new approval"]
    assert (target / "curve.txt").read_text(encoding="utf-8") == "new approval"
    assert not list(project.outputs_dir.glob("light_curve-previous*"))


def test_transaction_rollback_failure_is_retained_and_recovered_on_reopen(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.LIGHT_CURVE.value
    target.mkdir()
    (target / "curve.txt").write_text("last successful", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.LIGHT_CURVE)
    (pending / "curve.txt").write_text("new approval", encoding="utf-8")
    previous = target.with_name("light_curve-previous")
    original_replace = Path.replace

    def block_previous_restore(path: Path, destination: Path):
        if path == previous and destination == target:
            raise PermissionError(13, "OneDrive still holds the output", str(path))
        return original_replace(path, destination)

    def reject_finalize() -> None:
        raise RuntimeError("manifest save failed")

    monkeypatch.setattr(Path, "replace", block_previous_restore)

    with pytest.raises(LEAPSError) as error:
        project.commit_transaction(
            pending,
            resolved_target,
            finalize=reject_finalize,
        )

    assert error.value.code == "PROJECT_TRANSACTION_ROLLBACK_FAILED"
    assert not target.exists()
    assert (previous / "curve.txt").read_text(encoding="utf-8") == "last successful"
    assert (pending / "curve.txt").read_text(encoding="utf-8") == "new approval"
    marker = project.temporary_dir / ".light_curve-transaction-rollback.json"
    assert marker.is_file()

    monkeypatch.setattr(Path, "replace", original_replace)
    reopened = ProjectWorkspace.open(tmp_path)

    assert (reopened.outputs_dir / StageID.LIGHT_CURVE.value / "curve.txt").read_text(
        encoding="utf-8"
    ) == "last successful"
    assert not pending.exists()
    assert not previous.exists()
    assert not marker.exists()


def test_transaction_rotation_failure_preserves_current_output(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.LIGHT_CURVE.value
    target.mkdir()
    (target / "curve.txt").write_text("last successful", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.LIGHT_CURVE)
    (pending / "curve.txt").write_text("new approval", encoding="utf-8")
    original_replace = Path.replace

    def reject_target_rotation(path: Path, destination: Path):
        if path == target:
            raise PermissionError(13, "The process cannot access the file", str(path))
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", reject_target_rotation)

    with pytest.raises(PermissionError):
        project.commit_transaction(pending, resolved_target)

    assert (target / "curve.txt").read_text(encoding="utf-8") == "last successful"
    assert (pending / "curve.txt").read_text(encoding="utf-8") == "new approval"
    assert not list(project.outputs_dir.glob("light_curve-previous*"))


def test_transaction_retries_locked_backup_cleanup_without_rejecting_new_output(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.LIGHT_CURVE.value
    target.mkdir()
    (target / "curve.txt").write_text("first approval", encoding="utf-8")
    original_rmtree = shutil.rmtree

    def hold_windows_output_lock(path, *args, **kwargs):
        if Path(path).name.startswith("light_curve-previous"):
            raise PermissionError(13, "The process cannot access the file", str(path))
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("leaps.project.shutil.rmtree", hold_windows_output_lock)

    for value in ("second approval", "third approval"):
        pending, resolved_target = project.begin_transaction(StageID.LIGHT_CURVE)
        (pending / "curve.txt").write_text(value, encoding="utf-8")
        project.commit_transaction(pending, resolved_target)
        assert (target / "curve.txt").read_text(encoding="utf-8") == value

    retained = list(project.outputs_dir.glob("light_curve-previous*"))
    assert len(retained) == 2
    assert not (project.temporary_dir / "light_curve-pending").exists()

    monkeypatch.setattr("leaps.project.shutil.rmtree", original_rmtree)
    pending, resolved_target = project.begin_transaction(StageID.LIGHT_CURVE)
    (pending / "curve.txt").write_text("fourth approval", encoding="utf-8")
    project.commit_transaction(pending, resolved_target)

    assert (target / "curve.txt").read_text(encoding="utf-8") == "fourth approval"
    assert not list(project.outputs_dir.glob("light_curve-previous*"))


def test_transaction_missing_backup_during_final_cleanup_keeps_new_output(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.INSPECTION.value
    target.mkdir()
    (target / "inspection.json").write_text("old", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.INSPECTION)
    (pending / "inspection.json").write_text("confirmed", encoding="utf-8")
    original_rmtree = shutil.rmtree

    def lose_backup_entry(path, *args, **kwargs):
        if Path(path).name == "inspection-previous":
            raise FileNotFoundError(2, "No such file or directory", str(path))
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("leaps.project.shutil.rmtree", lose_backup_entry)

    project.commit_transaction(pending, resolved_target)

    assert (target / "inspection.json").read_text(encoding="utf-8") == "confirmed"
    assert (project.outputs_dir / "inspection-previous" / "inspection.json").read_text(
        encoding="utf-8"
    ) == "old"


def test_transaction_restores_previous_output_when_pending_install_fails(
    tmp_path: Path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.INSPECTION.value
    target.mkdir()
    (target / "inspection.json").write_text("last confirmed", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.INSPECTION)
    (pending / "inspection.json").write_text("new confirmation", encoding="utf-8")
    original_replace = Path.replace

    def reject_pending_install(path: Path, destination: Path):
        if path == pending:
            raise PermissionError(13, "The process cannot access the file", str(path))
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", reject_pending_install)

    with pytest.raises(PermissionError):
        project.commit_transaction(pending, resolved_target)

    assert (target / "inspection.json").read_text(encoding="utf-8") == "last confirmed"
    assert (pending / "inspection.json").read_text(encoding="utf-8") == "new confirmation"
    assert not list(project.outputs_dir.glob("inspection-previous*"))


def test_transaction_cleanup_does_not_remove_similarly_named_user_output(
    tmp_path: Path,
) -> None:
    project = ProjectWorkspace.create(tmp_path)
    target = project.outputs_dir / StageID.INSPECTION.value
    target.mkdir()
    (target / "inspection.json").write_text("old", encoding="utf-8")
    notes = project.outputs_dir / "inspection-previous-notes"
    notes.mkdir()
    (notes / "observer.txt").write_text("keep", encoding="utf-8")
    pending, resolved_target = project.begin_transaction(StageID.INSPECTION)
    (pending / "inspection.json").write_text("new", encoding="utf-8")

    project.commit_transaction(pending, resolved_target)

    assert (notes / "observer.txt").read_text(encoding="utf-8") == "keep"


def test_legacy_hidden_workspace_migrates_with_outputs_and_relative_references(tmp_path: Path) -> None:
    legacy = tmp_path / ".leaps"
    manifest = ProjectManifest(name="Legacy transit")
    state = manifest.stages[StageID.REDUCTION.value]
    state.checkpoint = ".leaps/checkpoints/reduction.json"
    state.output_path = ".leaps/outputs/reduction"
    manifest.save(legacy / "project.json")
    output = legacy / "outputs" / "reduction" / "reduced.fits"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"generated")
    log = legacy / "logs" / "leaps.jsonl"
    log.parent.mkdir()
    log.write_text('{"event":"saved"}\n', encoding="utf-8")
    checkpoint = legacy / "checkpoints" / "reduction.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text('{"frame":12}', encoding="utf-8")

    project = ProjectWorkspace.open(tmp_path)

    assert project.workspace == tmp_path / "LEAPS"
    assert not legacy.exists()
    assert (project.outputs_dir / "reduction" / "reduced.fits").read_bytes() == b"generated"
    assert (project.logs_dir / "leaps.jsonl").read_text(encoding="utf-8") == '{"event":"saved"}\n'
    assert (project.checkpoints_dir / "reduction.json").read_text(encoding="utf-8") == '{"frame":12}'
    state = project.manifest.stages[StageID.REDUCTION.value]
    assert state.checkpoint == "LEAPS/checkpoints/reduction.json"
    assert state.output_path == "LEAPS/outputs/reduction"


def test_project_open_stops_when_visible_and_legacy_workspaces_both_exist(tmp_path: Path) -> None:
    ProjectManifest(name="Visible").save(tmp_path / "LEAPS" / "project.json")
    ProjectManifest(name="Legacy").save(tmp_path / ".leaps" / "project.json")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert (tmp_path / "LEAPS" / "project.json").exists()
    assert (tmp_path / ".leaps" / "project.json").exists()


def test_failed_legacy_manifest_update_rolls_folder_back_without_data_loss(
    tmp_path: Path, monkeypatch
) -> None:
    legacy = tmp_path / ".leaps"
    manifest = ProjectManifest(name="Legacy transit")
    manifest.stages[StageID.REDUCTION.value].output_path = ".leaps/outputs/reduction"
    manifest.save(legacy / "project.json")
    output = legacy / "outputs" / "reduction" / "reduced.fits"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"generated")

    def fail_save(_manifest, _path):
        raise OSError("project manifest became unwritable")

    monkeypatch.setattr(ProjectManifest, "save", fail_save)
    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_MIGRATION_FAILED"
    assert legacy.exists()
    assert not (tmp_path / "LEAPS").exists()
    assert output.read_bytes() == b"generated"
    saved = json.loads((legacy / "project.json").read_text(encoding="utf-8"))
    assert saved["stages"]["reduction"]["output_path"] == ".leaps/outputs/reduction"


def test_project_creation_does_not_overwrite_unrelated_visible_folder(tmp_path: Path) -> None:
    unrelated = tmp_path / "LEAPS"
    unrelated.mkdir()
    (unrelated / "notes.txt").write_text("not a project")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.create(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert (unrelated / "notes.txt").read_text() == "not a project"


def test_project_open_stops_when_valid_workspace_contains_unrelated_data(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Visible")
    note = project.workspace / "observer-notes.txt"
    note.write_text("keep me", encoding="utf-8")

    with pytest.raises(LEAPSError) as error:
        ProjectWorkspace.open(tmp_path)

    assert error.value.code == "PROJECT_WORKSPACE_CONFLICT"
    assert note.read_text(encoding="utf-8") == "keep me"


def test_reset_deletes_only_generated_workspace_and_preserves_raw_frames(tmp_path: Path) -> None:
    raw = tmp_path / "light_001.fits"
    raw.write_bytes(b"raw pixels")
    project = ProjectWorkspace.create(tmp_path, "Safe reset")
    generated = project.outputs_dir / "reduction" / "reduced.fits"
    generated.parent.mkdir()
    generated.write_bytes(b"generated pixels")
    project.manifest.raw_files["science"] = [raw.name]
    project.save()

    removed = project.delete_generated_data()

    assert removed >= len(b"generated pixels")
    assert raw.read_bytes() == b"raw pixels"
    assert not project.workspace.exists()
    assert list(tmp_path.iterdir()) == [raw]


def test_reset_rejects_workspace_symlink_without_touching_target(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Linked project")
    real_workspace = tmp_path / "workspace-backup"
    project.workspace.rename(real_workspace)
    project.workspace.symlink_to(real_workspace, target_is_directory=True)

    with pytest.raises(LEAPSError) as error:
        project.delete_generated_data()

    assert error.value.code == "PROJECT_RESET_SYMLINK"
    assert (real_workspace / "project.json").exists()


def test_interrupted_reset_reports_remaining_staging_folder(tmp_path: Path, monkeypatch) -> None:
    project = ProjectWorkspace.create(tmp_path, "Interrupted reset")

    def fail_delete(_path):
        raise OSError("disk became unavailable")

    monkeypatch.setattr("leaps.project.shutil.rmtree", fail_delete)
    with pytest.raises(LEAPSError) as error:
        project.delete_generated_data()

    assert error.value.code == "PROJECT_RESET_INCOMPLETE"
    assert not project.workspace.exists()
    assert len(list(tmp_path.glob(".LEAPS-reset-*"))) == 1


def test_reset_rejects_any_path_outside_observing_run(tmp_path: Path) -> None:
    project = ProjectWorkspace.create(tmp_path, "Unsafe path")
    project.workspace = tmp_path.parent

    with pytest.raises(LEAPSError) as error:
        project.delete_generated_data()

    assert error.value.code == "PROJECT_RESET_UNSAFE_PATH"
    assert (tmp_path / "LEAPS" / "project.json").exists()
