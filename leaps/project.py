from __future__ import annotations

import errno
import json
import os
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .models import LEAPSError, ProjectManifest, StageID, StageState, StageStatus, utc_now


class ProjectWorkspace:
    WORKSPACE_NAME = "LEAPS"
    LEGACY_WORKSPACE_NAME = ".leaps"
    GENERATED_TOP_LEVEL_ENTRIES = {
        "project.json",
        "project.json.tmp",
        "logs",
        "cache",
        "checkpoints",
        "outputs",
        "tmp",
        ".DS_Store",
        "Thumbs.db",
        "desktop.ini",
    }

    def __init__(self, root: Path, manifest: ProjectManifest) -> None:
        self.root = root.resolve()
        self.workspace = self.root / self.WORKSPACE_NAME
        self.manifest_path = self.workspace / "project.json"
        self.manifest = manifest
        for directory in (
            self.logs_dir,
            self.cache_dir,
            self.checkpoints_dir,
            self.outputs_dir,
            self.temporary_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        for stage in StageID:
            self._recover_failed_transaction(stage)

    @property
    def logs_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def cache_dir(self) -> Path:
        return self.workspace / "cache"

    @property
    def checkpoints_dir(self) -> Path:
        return self.workspace / "checkpoints"

    @property
    def outputs_dir(self) -> Path:
        return self.workspace / "outputs"

    @property
    def temporary_dir(self) -> Path:
        return self.workspace / "tmp"

    @classmethod
    def create(cls, root: str | Path, name: str | None = None) -> ProjectWorkspace:
        root_path = Path(root).expanduser().resolve()
        root_path.mkdir(parents=True, exist_ok=True)
        if cls.has_workspace(root_path):
            raise cls._workspace_conflict(
                root_path,
                "A LEAPS workspace already exists in this observing run.",
            )
        manifest = ProjectManifest(name=name or root_path.name)
        project = cls(root_path, manifest)
        project.save()
        return project

    @classmethod
    def open(cls, root: str | Path) -> ProjectWorkspace:
        root_path = Path(root).expanduser().resolve()
        workspace = cls._existing_workspace(root_path)
        if workspace is not None:
            manifest = cls._load_manifest(workspace / "project.json")
            if workspace.name == cls.LEGACY_WORKSPACE_NAME:
                workspace = cls._migrate_legacy_workspace(root_path, workspace)
                try:
                    project = cls(root_path, manifest)
                    project._rewrite_legacy_references()
                    project._discard_appledouble_raw_references()
                    return project
                except Exception as exc:
                    rollback_error: OSError | None = None
                    legacy = root_path / cls.LEGACY_WORKSPACE_NAME
                    try:
                        if not os.path.lexists(legacy):
                            workspace.rename(legacy)
                    except OSError as rollback_exc:
                        rollback_error = rollback_exc
                    details = str(exc)
                    if rollback_error is not None:
                        details += f"\nRollback failed: {rollback_error}"
                    raise LEAPSError(
                        "PROJECT_MIGRATION_FAILED",
                        "The legacy project could not be migrated safely",
                        (
                            "LEAPS returned the project to .leaps/."
                            if rollback_error is None
                            else "The project files remain intact, but the folder name needs review."
                        ),
                        [
                            "Check folder permissions",
                            "Close other applications using the project",
                            f"Review {root_path}",
                            "Try again",
                        ],
                        stage=StageID.DATA_TARGET,
                        technical_details=details,
                    ) from exc
            project = cls(root_path, manifest)
            project._discard_appledouble_raw_references()
            return project
        return cls.import_hops(root_path)

    @classmethod
    def has_workspace(cls, root: str | Path) -> bool:
        root_path = Path(root).expanduser().resolve()
        return any(
            os.path.lexists(root_path / name)
            for name in (cls.WORKSPACE_NAME, cls.LEGACY_WORKSPACE_NAME)
        )

    @classmethod
    def has_project(cls, root: str | Path) -> bool:
        root_path = Path(root).expanduser().resolve()
        return any(
            (root_path / name / "project.json").is_file()
            for name in (cls.WORKSPACE_NAME, cls.LEGACY_WORKSPACE_NAME)
        )

    @classmethod
    def _existing_workspace(cls, root: Path) -> Path | None:
        visible = root / cls.WORKSPACE_NAME
        legacy = root / cls.LEGACY_WORKSPACE_NAME
        visible_exists = os.path.lexists(visible)
        legacy_exists = os.path.lexists(legacy)
        if visible_exists and legacy_exists:
            raise cls._workspace_conflict(
                root,
                "Both LEAPS/ and the legacy .leaps/ folder are present.",
            )
        workspace = visible if visible_exists else legacy if legacy_exists else None
        if workspace is None:
            return None
        if workspace.is_symlink():
            raise cls._workspace_conflict(
                root,
                f"{workspace.name}/ is a symbolic link and cannot be used safely.",
            )
        manifest = workspace / "project.json"
        if not workspace.is_dir() or not manifest.is_file() or manifest.is_symlink():
            raise cls._workspace_conflict(
                root,
                f"{workspace.name}/ exists but is not a valid LEAPS project folder.",
            )
        try:
            unrelated = sorted(
                child.name
                for child in workspace.iterdir()
                if not cls._is_generated_workspace_entry(child.name)
            )
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_WORKSPACE_UNREADABLE",
                "The project folder could not be inspected",
                f"LEAPS could not safely read {workspace}.",
                ["Check folder permissions", "Close other applications using the folder", "Try again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        if unrelated:
            names = ", ".join(unrelated[:3])
            remainder = len(unrelated) - 3
            if remainder > 0:
                names += f", and {remainder} more"
            raise cls._workspace_conflict(
                root,
                f"{workspace.name}/ contains files LEAPS did not create: {names}.",
            )
        return workspace

    @classmethod
    def _is_generated_workspace_entry(cls, name: str) -> bool:
        """Recognize LEAPS entries and their macOS AppleDouble companions."""
        entry = name[2:] if name.startswith("._") else name
        if entry in cls.GENERATED_TOP_LEVEL_ENTRIES:
            return True
        prefix = "project.json."
        suffix = ".tmp"
        if entry.startswith(prefix) and entry.endswith(suffix):
            token = entry[len(prefix) : -len(suffix)]
            if len(token) == 32 and set(token) <= set("0123456789abcdef"):
                return True
        return False

    @classmethod
    def _migrate_legacy_workspace(cls, root: Path, legacy: Path) -> Path:
        visible = root / cls.WORKSPACE_NAME
        if os.path.lexists(visible):
            raise cls._workspace_conflict(
                root,
                "LEAPS/ already exists, so the legacy project was not moved.",
            )
        try:
            legacy.rename(visible)
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_MIGRATION_FAILED",
                "The legacy project could not be moved",
                "LEAPS left the existing .leaps folder unchanged.",
                ["Check folder permissions", "Close other applications using the project", "Try again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        return visible

    @staticmethod
    def _load_manifest(path: Path) -> ProjectManifest:
        try:
            return ProjectManifest.load(path)
        except Exception as exc:
            raise LEAPSError(
                "PROJECT_MANIFEST_INVALID",
                "The LEAPS project information is damaged",
                f"The project manifest at {path} could not be read.",
                ["Restore project.json from a backup", "Export diagnostics", "Reset the project data"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc

    @classmethod
    def _workspace_conflict(cls, root: Path, message: str) -> LEAPSError:
        return LEAPSError(
            "PROJECT_WORKSPACE_CONFLICT",
            "The project folder needs attention",
            message,
            [
                f"Review {root / cls.WORKSPACE_NAME}",
                "Rename unrelated files or keep only one LEAPS project folder",
                "Try opening the observing run again",
            ],
            stage=StageID.DATA_TARGET,
        )

    @classmethod
    def import_hops(cls, root: Path) -> ProjectWorkspace:
        project = cls.create(root, root.name)
        legacy_log = root / "log.yaml"
        if legacy_log.exists():
            try:
                import yaml

                values = yaml.safe_load(legacy_log.read_text(encoding="utf-8")) or {}
                project.manifest.target_name = str(values.get("target_name", ""))
                coordinates = str(values.get("target_ra_dec", "")).split()
                if len(coordinates) == 2:
                    project.manifest.target_ra, project.manifest.target_dec = coordinates
                for stage in StageID:
                    if values.get(f"{stage.value}_complete"):
                        project.manifest.stages[stage.value] = StageState(
                            status=StageStatus.COMPLETE,
                            summary="Imported from HOPS",
                        )
                project.manifest.warnings.append(
                    {
                        "code": "IMPORTED_HOPS",
                        "message": "Legacy HOPS state was imported without changing its files.",
                    }
                )
                project.save()
            except Exception as exc:
                project.manifest.warnings.append({"code": "HOPS_IMPORT_PARTIAL", "message": str(exc)})
                project.save()
        return project

    def save(self) -> None:
        self.manifest.save(self.manifest_path)

    def _discard_appledouble_raw_references(self) -> None:
        """Remove macOS resource-fork sidecars imported by older LEAPS builds."""
        changed = False
        for category, paths in self.manifest.raw_files.items():
            retained = [path for path in paths if not Path(path).name.startswith("._")]
            if retained != paths:
                self.manifest.raw_files[category] = retained
                changed = True
        if changed:
            self.manifest.warnings.append(
                {
                    "code": "APPLEDOUBLE_FILES_IGNORED",
                    "message": (
                        "macOS ._ metadata sidecars on the observing drive were ignored; "
                        "the corresponding FITS images were not changed."
                    ),
                }
            )
            self.save()

    def reduced_fits_files(self, stage: StageID | None = None) -> list[Path]:
        """Return real reduced FITS images, excluding external-drive metadata sidecars."""
        from .fits_inventory import is_fits_path

        reduction = self.outputs_dir / StageID.REDUCTION.value
        try:
            return sorted(
                path
                for path in reduction.iterdir()
                if path.is_file() and is_fits_path(path)
            )
        except FileNotFoundError:
            return []
        except OSError as exc:
            raise self._storage_access_failure(reduction, "read", exc, stage) from exc

    def verify_process_access(self, stage: StageID | None = None) -> None:
        """Verify representative input reads and a reversible workspace write.

        This deliberately opens at most one assigned raw frame per category so
        process startup remains fast even for very large observing runs. Later
        stages also check one real reduced FITS file. The temporary write is
        confined to LEAPS/tmp and is deleted immediately.
        """
        checked: set[Path] = set()
        for paths in self.manifest.raw_files.values():
            path = next(
                (
                    self.resolve(relative)
                    for relative in paths
                    if not Path(relative).name.startswith("._")
                ),
                None,
            )
            if path is None or path in checked:
                continue
            checked.add(path)
            try:
                with path.open("rb") as handle:
                    handle.read(1)
            except OSError as exc:
                raise self._storage_access_failure(path, "read", exc, stage) from exc

        if stage not in {None, StageID.DATA_TARGET, StageID.REDUCTION}:
            reduced = self.reduced_fits_files(stage)
            if reduced:
                path = reduced[0]
                try:
                    with path.open("rb") as handle:
                        handle.read(1)
                except OSError as exc:
                    raise self._storage_access_failure(path, "read", exc, stage) from exc

        probe = self.temporary_dir / f".access-check-{uuid.uuid4().hex}.tmp"
        try:
            with probe.open("xb") as handle:
                handle.write(b"LEAPS access check\n")
                handle.flush()
            probe.unlink()
        except OSError as exc:
            raise self._storage_access_failure(probe, "write", exc, stage) from exc
        finally:
            try:
                probe.unlink(missing_ok=True)
            except OSError:
                pass

    @staticmethod
    def _storage_access_failure(
        path: Path,
        operation: str,
        exc: OSError,
        stage: StageID | None,
    ) -> LEAPSError:
        denied = isinstance(exc, PermissionError) or exc.errno in {errno.EACCES, errno.EPERM}
        code = "PROJECT_STORAGE_ACCESS_DENIED" if denied else "PROJECT_STORAGE_UNAVAILABLE"
        if operation == "read":
            message = (
                "LEAPS could not read a file required for this step. The observing drive may be "
                "disconnected, unavailable, or blocked by system permissions."
            )
        else:
            message = (
                "LEAPS could not write its temporary workspace. The observing drive may be "
                "read-only, unavailable, full, or blocked by system permissions."
            )
        return LEAPSError(
            code,
            "LEAPS cannot access the project location",
            message,
            [
                "Choose the observing-run folder again to renew access",
                "On macOS, allow LEAPS under Privacy & Security > Files and Folders",
                "Confirm the external drive is connected and mounted read/write",
                "Retry the process",
            ],
            stage=stage,
            technical_details=f"{operation.title()} access failed for {path}\n{type(exc).__name__}: {exc}",
        )

    def _rewrite_legacy_references(self) -> None:
        prefix = f"{self.LEGACY_WORKSPACE_NAME}/"
        replacement = f"{self.WORKSPACE_NAME}/"
        changed = False
        for state in self.manifest.stages.values():
            for attribute in ("checkpoint", "output_path"):
                value = getattr(state, attribute)
                if isinstance(value, str) and value.startswith(prefix):
                    setattr(state, attribute, replacement + value[len(prefix) :])
                    changed = True
        if changed:
            self.save()

    def workspace_size(self) -> int:
        total = 0
        if not self.workspace.exists() or self.workspace.is_symlink():
            return total
        for directory, subdirectories, filenames in os.walk(self.workspace, followlinks=False):
            base = Path(directory)
            subdirectories[:] = [
                name for name in subdirectories if not (base / name).is_symlink()
            ]
            for name in filenames:
                try:
                    total += (base / name).lstat().st_size
                except OSError:
                    continue
        return total

    def delete_generated_data(self) -> int:
        """Delete only this validated LEAPS workspace, never the observing-run root."""
        workspace = self.workspace
        if workspace.parent != self.root or workspace.name not in {
            self.WORKSPACE_NAME,
            self.LEGACY_WORKSPACE_NAME,
        }:
            raise LEAPSError(
                "PROJECT_RESET_UNSAFE_PATH",
                "Project reset was stopped",
                "The generated-data folder is not a direct LEAPS workspace beside the FITS data.",
                ["Open the observing run again", "Export diagnostics"],
                stage=StageID.DATA_TARGET,
                technical_details=str(workspace),
            )
        if workspace.is_symlink() or self.manifest_path.is_symlink():
            raise LEAPSError(
                "PROJECT_RESET_SYMLINK",
                "Project reset was stopped",
                "LEAPS will not delete a workspace or manifest reached through a symbolic link.",
                ["Replace the link with a normal project folder", "Remove it manually after inspection"],
                stage=StageID.DATA_TARGET,
            )
        if not workspace.exists():
            return 0
        on_disk = self._load_manifest(self.manifest_path)
        if on_disk.project_id != self.manifest.project_id:
            raise LEAPSError(
                "PROJECT_RESET_ID_MISMATCH",
                "Project reset was stopped",
                "The project on disk no longer matches the project open in LEAPS.",
                ["Close and reopen the observing run", "Try reset again"],
                stage=StageID.DATA_TARGET,
            )
        removed_bytes = self.workspace_size()
        staging = self.root / f".LEAPS-reset-{uuid.uuid4().hex[:10]}"
        try:
            workspace.rename(staging)
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_RESET_FAILED",
                "Project data could not be reset",
                "LEAPS did not remove any project files.",
                ["Check folder permissions", "Close other applications using the folder", "Try again"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        try:
            shutil.rmtree(staging)
        except OSError as exc:
            raise LEAPSError(
                "PROJECT_RESET_INCOMPLETE",
                "Project reset needs attention",
                f"The active project was removed, but some generated data remains at {staging}.",
                ["Delete the remaining reset folder manually", "Verify available disk access"],
                stage=StageID.DATA_TARGET,
                technical_details=str(exc),
            ) from exc
        return removed_bytes

    def relative(self, path: str | Path) -> str:
        resolved = Path(path).expanduser().resolve()
        try:
            return resolved.relative_to(self.root).as_posix()
        except ValueError:
            return resolved.as_posix()

    def resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else (self.root / candidate).resolve()

    def set_stage(
        self,
        stage: StageID,
        status: StageStatus,
        summary: str,
        *,
        progress: float | None = None,
        checkpoint: str | None = None,
        output_path: str | Path | None = None,
    ) -> None:
        state = self.manifest.stages[stage.value]
        state.status = status
        state.summary = summary
        state.updated_at = utc_now()
        if progress is not None:
            state.progress = progress
        if checkpoint is not None:
            state.checkpoint = checkpoint
        if output_path is not None:
            state.output_path = self.relative(output_path)
        if status == StageStatus.COMPLETE:
            stages = list(StageID)
            next_index = stages.index(stage) + 1
            if next_index < len(stages):
                next_stage = self.manifest.stages[stages[next_index].value]
                if next_stage.status == StageStatus.LOCKED:
                    next_stage.status = StageStatus.READY
                    next_stage.summary = "Ready"
        self.save()

    def begin_transaction(self, stage: StageID) -> tuple[Path, Path]:
        target = self.outputs_dir / stage.value
        pending = self.temporary_dir / f"{stage.value}-pending"
        self._recover_failed_transaction(stage)
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        return pending, target

    def _recover_failed_transaction(self, stage: StageID) -> None:
        """Restore output retained after a rare finalize/rollback double failure."""
        target = self.outputs_dir / stage.value
        pending = self.temporary_dir / f"{stage.value}-pending"
        marker = self._transaction_rollback_marker(target)
        if not marker.is_file():
            return
        previous: Path | None = None
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            previous = Path(str(payload["previous"]))
            had_previous = bool(payload["had_previous"])
            if not self._is_transaction_backup(previous, target):
                raise ValueError(f"Unrecognized transaction backup: {previous}")

            if had_previous:
                if target.exists() and previous.exists():
                    if pending.exists():
                        raise FileExistsError(
                            errno.EEXIST,
                            "Both current and pending transaction outputs exist",
                            str(pending),
                        )
                    target.replace(pending)
                if previous.exists() and not target.exists():
                    previous.replace(target)
                if not target.exists():
                    raise FileNotFoundError(
                        errno.ENOENT,
                        "The previous transaction output could not be restored",
                        str(target),
                    )
            elif target.exists():
                if pending.exists():
                    raise FileExistsError(
                        errno.EEXIST,
                        "Both current and pending transaction outputs exist",
                        str(pending),
                    )
                target.replace(pending)

            try:
                if pending.exists():
                    shutil.rmtree(pending)
            except OSError:
                pass
            try:
                marker.unlink(missing_ok=True)
            except OSError:
                pass
        except BaseException as exc:
            raise LEAPSError(
                "PROJECT_TRANSACTION_RECOVERY_FAILED",
                "The previous result needs attention",
                (
                    "LEAPS retained output from an interrupted rollback but could not safely "
                    "restore the last successful result."
                ),
                [
                    "Close applications using the project folder",
                    "Wait for cloud syncing to finish",
                    "Retry opening the project",
                    "Export diagnostics if this repeats",
                ],
                stage=stage,
                technical_details=(
                    f"Recovery marker: {marker}\nCurrent output: {target}\n"
                    f"Previous output: {previous}\nPending output: {pending}\n"
                    f"{type(exc).__name__}: {exc}"
                ),
            ) from exc

    def _transaction_rollback_marker(self, target: Path) -> Path:
        return self.temporary_dir / f".{target.name}-transaction-rollback.json"

    @staticmethod
    def _is_transaction_backup(path: Path, target: Path) -> bool:
        if path.parent != target.parent:
            return False
        base = f"{target.name}-previous"
        if path.name == base:
            return True
        prefix = f"{base}-"
        suffix = path.name.removeprefix(prefix)
        return (
            path.name.startswith(prefix)
            and len(suffix) == 10
            and set(suffix) <= set("0123456789abcdef")
        )

    def discard_pending_transaction(self, stage: StageID) -> bool:
        """Remove only the recognized temporary output for one stage."""
        pending = self.temporary_dir / f"{stage.value}-pending"
        if pending.is_symlink():
            pending.unlink()
            return True
        if pending.exists():
            shutil.rmtree(pending)
            return True
        return False

    def commit_transaction(
        self,
        pending: Path,
        target: Path,
        *,
        finalize: Callable[[], None] | None = None,
    ) -> None:
        """Install pending output, optionally finalizing related persisted state.

        When ``finalize`` is provided, the previous output is retained until
        that callback succeeds. A callback failure restores the prior output so
        an unsuccessful manifest save cannot make new data look authoritative.
        """
        previous = target.with_name(target.name + "-previous")
        generated_backups = [previous]
        generated_backups.extend(
            path
            for path in target.parent.glob(f"{previous.name}-*")
            if len(path.name.removeprefix(f"{previous.name}-")) == 10
            and set(path.name.removeprefix(f"{previous.name}-"))
            <= set("0123456789abcdef")
        )
        if previous.exists() or previous.is_symlink():
            previous = target.with_name(
                f"{target.name}-previous-{uuid.uuid4().hex[:10]}"
            )
        had_previous = target.exists()
        if had_previous:
            target.replace(previous)
        try:
            pending.replace(target)
        except BaseException:
            if previous.exists() and not target.exists():
                previous.replace(target)
            raise
        if finalize is not None:
            try:
                finalize()
            except BaseException as exc:
                rollback_error: BaseException | None = None
                try:
                    target.replace(pending)
                    if had_previous and previous.exists():
                        previous.replace(target)
                except BaseException as rollback_exc:
                    rollback_error = rollback_exc
                if rollback_error is not None:
                    marker = self._transaction_rollback_marker(target)
                    marker_error = ""
                    try:
                        marker.write_text(
                            json.dumps(
                                {
                                    "target": str(target),
                                    "previous": str(previous),
                                    "pending": str(pending),
                                    "had_previous": had_previous,
                                },
                                indent=2,
                            ),
                            encoding="utf-8",
                        )
                    except OSError as marker_exc:
                        marker_error = (
                            f"\nRecovery marker write failed: "
                            f"{type(marker_exc).__name__}: {marker_exc}"
                        )
                    try:
                        stage = StageID(target.name)
                    except ValueError:
                        stage = None
                    raise LEAPSError(
                        "PROJECT_TRANSACTION_ROLLBACK_FAILED",
                        "The previous result needs attention",
                        (
                            "LEAPS could not finish saving the new result or fully restore "
                            "the previous output. Both copies were retained where possible."
                        ),
                        [
                            "Close applications using the project folder",
                            "Wait for cloud syncing to finish",
                            "Retry the step",
                            "Export diagnostics if this repeats",
                        ],
                        stage=stage,
                        technical_details=(
                            f"Finalize failed: {type(exc).__name__}: {exc}\n"
                            f"Rollback failed: {type(rollback_error).__name__}: {rollback_error}\n"
                            f"Current output: {target}\nPrevious output: {previous}\n"
                            f"Pending output: {pending}\nRecovery marker: {marker}"
                            f"{marker_error}"
                        ),
                    ) from exc
                raise
        for stale in {*generated_backups, previous}:
            self._discard_transaction_backup(stale)

    @staticmethod
    def _discard_transaction_backup(path: Path) -> bool:
        """Best-effort cleanup after an output has already been replaced.

        Windows may keep an image or text output locked briefly after it has
        been displayed or read. That must not turn a completed transaction into
        a failure; a later transaction will retry any retained backup.
        """
        try:
            if path.is_symlink():
                path.unlink()
            elif path.exists():
                shutil.rmtree(path)
            return True
        except OSError:
            return False

    def update_settings(self, **values: Any) -> None:
        self.manifest.settings.update(values)
        self.save()
