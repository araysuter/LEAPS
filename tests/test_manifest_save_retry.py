from __future__ import annotations

import errno
import json
import os
from pathlib import Path

import pytest

from leaps.models import LEAPSError, ProjectManifest


def _windows_error(winerror: int) -> OSError:
    error = OSError(errno.EIO, "The process cannot access the file")
    error.winerror = winerror
    return error


@pytest.mark.parametrize(
    "failure",
    [
        PermissionError("Access is denied"),
        OSError(errno.EACCES, "Permission denied"),
        OSError(errno.EPERM, "Operation not permitted"),
        OSError(errno.EBUSY, "Resource busy"),
        _windows_error(5),
        _windows_error(32),
        _windows_error(33),
    ],
)
def test_manifest_save_retries_transient_replace_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: OSError
) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="OneDrive project")
    original_replace = os.replace
    attempts: list[tuple[Path, Path]] = []
    delays: list[float] = []

    def replace_after_lock(source: str | Path, destination: str | Path) -> None:
        attempts.append((Path(source), Path(destination)))
        if len(attempts) == 1:
            raise failure
        original_replace(source, destination)

    monkeypatch.setattr("leaps.models.os.replace", replace_after_lock)
    monkeypatch.setattr("leaps.models.time.sleep", delays.append)

    manifest.save(path)

    assert len(attempts) == 2
    assert delays == [0.05]
    assert attempts[0][0].parent == path.parent
    assert attempts[0][0] != path.with_suffix(".json.tmp")
    assert attempts[0][0].name.startswith("project.json.")
    assert attempts[0][1] == path
    assert json.loads(path.read_text(encoding="utf-8"))["name"] == "OneDrive project"
    assert not list(tmp_path.glob("project.json.*.tmp"))


def test_manifest_save_uses_six_attempts_with_exponential_delays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "project.json"
    original = ProjectManifest(name="Last successful version")
    original.save(path)
    previous_payload = path.read_bytes()
    previous_updated_at = original.updated_at
    original.name = "Blocked update"
    attempts: list[Path] = []
    delays: list[float] = []

    def keep_locked(source: str | Path, _destination: str | Path) -> None:
        attempts.append(Path(source))
        raise PermissionError(errno.EACCES, "Access is denied", str(path))

    monkeypatch.setattr("leaps.models.os.replace", keep_locked)
    monkeypatch.setattr("leaps.models.time.sleep", delays.append)

    with pytest.raises(LEAPSError) as error:
        original.save(path)

    assert error.value.code == "PROJECT_MANIFEST_SAVE_BLOCKED"
    assert "OneDrive" in error.value.message
    assert len(attempts) == 6
    assert len(set(attempts)) == 1
    assert delays == [0.05, 0.1, 0.2, 0.4, 0.8]
    assert path.read_bytes() == previous_payload
    assert original.updated_at == previous_updated_at
    assert not list(tmp_path.glob("project.json.*.tmp"))


def test_manifest_save_does_not_retry_nontransient_io_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="Last successful version")
    manifest.save(path)
    previous_payload = path.read_bytes()
    manifest.name = "Update with no disk space"
    attempts = 0
    delays: list[float] = []

    def disk_full(_source: str | Path, _destination: str | Path) -> None:
        nonlocal attempts
        attempts += 1
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr("leaps.models.os.replace", disk_full)
    monkeypatch.setattr("leaps.models.time.sleep", delays.append)

    with pytest.raises(LEAPSError) as error:
        manifest.save(path)

    assert error.value.code == "PROJECT_MANIFEST_SAVE_FAILED"
    assert attempts == 1
    assert delays == []
    assert path.read_bytes() == previous_payload
    assert not list(tmp_path.glob("project.json.*.tmp"))


def test_manifest_save_flushes_and_fsyncs_unique_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="Durable project")
    original_replace = os.replace
    fsynced: list[int] = []
    sources: list[Path] = []

    def record_fsync(file_descriptor: int) -> None:
        fsynced.append(file_descriptor)

    def record_replace(source: str | Path, destination: str | Path) -> None:
        sources.append(Path(source))
        original_replace(source, destination)

    monkeypatch.setattr("leaps.models.os.fsync", record_fsync)
    monkeypatch.setattr("leaps.models.os.replace", record_replace)

    manifest.save(path)

    assert len(fsynced) == 1
    assert len(sources) == 1
    assert sources[0].parent == path.parent
    assert sources[0].name.startswith("project.json.")
    assert sources[0].suffix == ".tmp"
    assert not sources[0].exists()


def test_manifest_serialization_failure_restores_timestamp_and_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "project.json"
    manifest = ProjectManifest(name="Serializable project")
    manifest.save(path)
    previous_payload = path.read_bytes()
    previous_updated_at = manifest.updated_at
    manifest.settings["not_json"] = object()

    with pytest.raises(TypeError):
        manifest.save(path)

    assert manifest.updated_at == previous_updated_at
    assert path.read_bytes() == previous_payload
    assert not list(tmp_path.glob("project.json.*.tmp"))
