from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .models import ProjectManifest, StageID, StageState, StageStatus, utc_now


class ProjectWorkspace:
    WORKSPACE_NAME = ".leaps"

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
        manifest = ProjectManifest(name=name or root_path.name)
        project = cls(root_path, manifest)
        project.save()
        return project

    @classmethod
    def open(cls, root: str | Path) -> ProjectWorkspace:
        root_path = Path(root).expanduser().resolve()
        manifest_path = root_path / cls.WORKSPACE_NAME / "project.json"
        if manifest_path.exists():
            return cls(root_path, ProjectManifest.load(manifest_path))
        return cls.import_hops(root_path)

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
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir(parents=True)
        return pending, target

    def commit_transaction(self, pending: Path, target: Path) -> None:
        previous = target.with_name(target.name + "-previous")
        if previous.exists():
            shutil.rmtree(previous)
        if target.exists():
            target.replace(previous)
        pending.replace(target)
        if previous.exists():
            shutil.rmtree(previous)

    def update_settings(self, **values: Any) -> None:
        self.manifest.settings.update(values)
        self.save()
