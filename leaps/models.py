from __future__ import annotations

import errno
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

_MANIFEST_REPLACE_RETRY_DELAYS = (0.05, 0.1, 0.2, 0.4, 0.8)
_TRANSIENT_MANIFEST_ERRNOS = {errno.EACCES, errno.EPERM, errno.EBUSY}
_TRANSIENT_MANIFEST_WINERRORS = {5, 32, 33}


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def target_fingerprint(ra: str, dec: str) -> str:
    """Return a stable identity for state that depends on target coordinates."""
    normalized = " ".join(f"{ra.strip()} {dec.strip()}".casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class StageID(StrEnum):
    DATA_TARGET = "data_target"
    REDUCTION = "reduction"
    INSPECTION = "inspection"
    ALIGNMENT = "alignment"
    PHOTOMETRY = "photometry"
    LIGHT_CURVE = "light_curve"
    FITTING = "fitting"
    SECONDARY_ECLIPSE = "secondary_eclipse"


class StageStatus(StrEnum):
    LOCKED = "locked"
    READY = "ready"
    RUNNING = "running"
    NEEDS_ATTENTION = "needs_attention"
    COMPLETE = "complete"


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class StageState:
    status: StageStatus = StageStatus.LOCKED
    summary: str = "Locked"
    progress: float = 0.0
    checkpoint: str | None = None
    output_path: str | None = None
    updated_at: str = field(default_factory=utc_now)
    warning_codes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class StageEvent:
    stage: StageID
    status: JobStatus
    message: str
    current: int = 0
    total: int = 0
    checkpoint: str | None = None
    diagnostic_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.current / self.total)


class LEAPSError(RuntimeError):
    """A user-actionable failure that is safe to show in the UI."""

    def __init__(
        self,
        code: str,
        title: str,
        message: str,
        recovery: list[str],
        *,
        stage: StageID | None = None,
        technical_details: str | None = None,
        diagnostic_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.title = title
        self.message = message
        self.recovery = recovery
        self.stage = stage
        self.technical_details = technical_details or ""
        self.diagnostic_id = diagnostic_id or f"LEAPS-{uuid.uuid4().hex[:10].upper()}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "title": self.title,
            "message": self.message,
            "recovery": self.recovery,
            "stage": self.stage.value if self.stage else None,
            "technical_details": self.technical_details,
            "diagnostic_id": self.diagnostic_id,
        }


def _is_transient_manifest_replace_error(exc: OSError) -> bool:
    return (
        isinstance(exc, PermissionError)
        or exc.errno in _TRANSIENT_MANIFEST_ERRNOS
        or getattr(exc, "winerror", None) in _TRANSIENT_MANIFEST_WINERRORS
    )


def _manifest_save_error(path: Path, exc: OSError, *, blocked: bool) -> LEAPSError:
    if blocked:
        return LEAPSError(
            "PROJECT_MANIFEST_SAVE_BLOCKED",
            "The project could not be saved",
            (
                "Windows or a cloud-sync service such as OneDrive kept the project information "
                "locked. LEAPS left the previous project information unchanged."
            ),
            [
                "Wait for OneDrive or another cloud-sync service to finish syncing",
                "Close other applications using the project folder",
                "Pause syncing briefly if the problem continues",
                "Retry the step",
            ],
            technical_details=(
                f"Atomic replacement failed for {path}\n{type(exc).__name__}: {exc}"
            ),
        )
    return LEAPSError(
        "PROJECT_MANIFEST_SAVE_FAILED",
        "The project could not be saved",
        "LEAPS could not write the project information. The previous file was left unchanged.",
        [
            "Confirm the project folder is writable and has free space",
            "Close other applications using the project folder",
            "Retry the step",
        ],
        technical_details=f"Save failed for {path}\n{type(exc).__name__}: {exc}",
    )


@dataclass(slots=True)
class ProjectManifest:
    schema_version: int = 2
    project_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Untitled transit"
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    target_name: str = ""
    target_ra: str = ""
    target_dec: str = ""
    global_profile: dict[str, Any] = field(default_factory=dict)
    raw_files: dict[str, list[str]] = field(
        default_factory=lambda: {
            key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")
        }
    )
    stages: dict[str, StageState] = field(default_factory=dict)
    asset_versions: dict[str, str] = field(default_factory=dict)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for index, stage in enumerate(StageID):
            if stage.value not in self.stages:
                is_completed_fit = (
                    stage == StageID.SECONDARY_ECLIPSE
                    and self.stages.get(StageID.FITTING.value, StageState()).status
                    == StageStatus.COMPLETE
                )
                status = StageStatus.READY if index == 0 or is_completed_fit else StageStatus.LOCKED
                summary = "Ready" if index == 0 or is_completed_fit else "Locked"
                self.stages[stage.value] = StageState(status=status, summary=summary)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for value in payload["stages"].values():
            value["status"] = str(value["status"])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectManifest:
        source_version = int(payload.get("schema_version", 1))
        had_light_curve_stage = StageID.LIGHT_CURVE.value in payload.get("stages", {})
        stages = {
            key: StageState(
                status=StageStatus(value.get("status", StageStatus.LOCKED)),
                summary=value.get("summary", "Locked"),
                progress=float(value.get("progress", 0.0)),
                checkpoint=value.get("checkpoint"),
                output_path=value.get("output_path"),
                updated_at=value.get("updated_at", utc_now()),
                warning_codes=list(value.get("warning_codes", [])),
            )
            for key, value in payload.get("stages", {}).items()
        }
        known = {field.name for field in cls.__dataclass_fields__.values()}
        values = {key: value for key, value in payload.items() if key in known and key != "stages"}
        manifest = cls(stages=stages, **values)
        if source_version < 2:
            manifest.schema_version = 2
            photometry = manifest.stages[StageID.PHOTOMETRY.value]
            if not had_light_curve_stage and photometry.status == StageStatus.COMPLETE:
                manifest.stages[StageID.LIGHT_CURVE.value] = StageState(
                    status=StageStatus.READY,
                    summary="Review comparison stars",
                )
        return manifest

    @classmethod
    def load(cls, path: Path) -> ProjectManifest:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path = Path(path)
        previous_updated_at = self.updated_at
        self.updated_at = utc_now()
        temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        saved = False
        try:
            payload = json.dumps(self.to_dict(), indent=2, sort_keys=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            for attempt in range(len(_MANIFEST_REPLACE_RETRY_DELAYS) + 1):
                try:
                    os.replace(temporary, path)
                    saved = True
                    return
                except OSError as exc:
                    if not _is_transient_manifest_replace_error(exc):
                        raise _manifest_save_error(path, exc, blocked=False) from exc
                    if attempt == len(_MANIFEST_REPLACE_RETRY_DELAYS):
                        raise _manifest_save_error(path, exc, blocked=True) from exc
                    time.sleep(_MANIFEST_REPLACE_RETRY_DELAYS[attempt])
        except LEAPSError:
            raise
        except OSError as exc:
            raise _manifest_save_error(path, exc, blocked=False) from exc
        finally:
            if not saved:
                self.updated_at = previous_updated_at
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
