from __future__ import annotations

import json
from pathlib import Path

from leaps.models import ProjectManifest, StageID, StageStatus
from leaps.project import ProjectWorkspace


def test_manifest_round_trip_uses_versioned_plain_values(tmp_path: Path) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="Test transit")
    manifest.stages[StageID.DATA_TARGET.value].status = StageStatus.COMPLETE
    manifest.save(path)
    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
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
