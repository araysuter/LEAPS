from __future__ import annotations

import json
import platform
import threading
import traceback
import zipfile
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from .models import LEAPSError, StageID
from .project import ProjectWorkspace


class DiagnosticLogger:
    def __init__(self, project: ProjectWorkspace) -> None:
        self.project = project
        self.path = project.logs_dir / "leaps.jsonl"
        self._lock = threading.Lock()

    def record(self, event: str, *, stage: StageID | None = None, **data: Any) -> None:
        payload = {
            "timestamp": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "event": event,
            "stage": stage.value if stage else None,
            **data,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")

    def failure(self, exc: BaseException, stage: StageID | None = None) -> LEAPSError:
        if isinstance(exc, LEAPSError):
            failure = exc
        else:
            failure = LEAPSError(
                "UNEXPECTED_FAILURE",
                "This step could not be completed",
                "LEAPS kept the last successful result. You can retry or export diagnostics.",
                ["Retry the step", "Export diagnostics if the problem repeats"],
                stage=stage,
                technical_details="".join(traceback.format_exception(exc)),
            )
        self.record("failure", stage=stage, **failure.as_dict())
        return failure

    def export_bundle(self, destination: str | Path, sample_headers: list[Path] | None = None) -> Path:
        destination_path = Path(destination).expanduser().resolve()
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        system = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "packages": {
                package: self._version(package)
                for package in (
                    "leaps-exoplanet",
                    "PySide6",
                    "numpy",
                    "astropy",
                    "astroquery",
                    "photutils",
                    "scipy",
                )
            },
        }
        with zipfile.ZipFile(destination_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("system.json", json.dumps(system, indent=2))
            archive.writestr(
                "project.json", json.dumps(self.project.manifest.to_dict(), indent=2, default=str)
            )
            if self.path.exists():
                archive.write(self.path, "logs/leaps.jsonl")
            for index, path in enumerate(sample_headers or []):
                header = self._sanitized_header(path)
                archive.writestr(
                    f"headers/header-{index + 1}.json", json.dumps(header, indent=2, default=str)
                )
        self.record("diagnostic_bundle_exported", destination=destination_path.name)
        return destination_path

    @staticmethod
    def _version(package: str) -> str:
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return "not installed"

    def _sanitized_header(self, path: Path) -> dict[str, Any]:
        try:
            from astropy.io import fits

            with fits.open(path, memmap=True) as hdus:
                header = next((hdu.header for hdu in hdus if hdu.header), {})
            blocked = {"COMMENT", "HISTORY", "OBSERVER", "AUTHOR", "CREATOR"}
            return {
                str(key): str(value)[:500]
                for key, value in header.items()
                if key not in blocked
                and not any(token in key.upper() for token in ("USER", "OWNER", "EMAIL", "PHONE", "ADDRESS"))
            }
        except Exception as exc:
            return {"error": str(exc), "file": path.name}
