from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from .models import LEAPSError


@dataclass(slots=True)
class OfflineAsset:
    asset_id: str
    label: str
    estimated_bytes: int
    version: str
    url: str | None = None
    sha256: str | None = None
    filename: str | None = None
    installed: bool = False

    @property
    def display_size(self) -> str:
        value = float(self.estimated_bytes)
        for suffix in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or suffix == "TB":
                return f"{value:.1f} {suffix}"
            value /= 1024
        return f"{value:.1f} TB"


DEFAULT_ASSETS = [
    OfflineAsset("exoclock", "ExoClock catalogues and ephemerides", 80_000_000, "current"),
    OfflineAsset("nasa", "NASA Exoplanet Archive snapshot", 25_000_000, "current"),
    OfflineAsset("pylightcurve", "PyLightcurve photometry databases", 160_000_000, "current"),
    OfflineAsset("exotethys", "ExoTETHyS filter tables", 120_000_000, "current"),
    OfflineAsset("phoenix", "PHOENIX limb-darkening model grid", 3_200_000_000, "current"),
]


class OfflineDataManager:
    def __init__(self, root: str | Path, manifest_url: str | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_url = manifest_url or os.getenv(
            "LEAPS_ASSET_MANIFEST_URL",
            "https://github.com/MrRayBob/LEAPS/releases/latest/download/offline-assets.json",
        )
        self.assets = [replace(asset) for asset in DEFAULT_ASSETS]
        self.refresh_installed()

    @property
    def total_estimated_bytes(self) -> int:
        return sum(asset.estimated_bytes for asset in self.assets if not asset.installed)

    @property
    def free_bytes(self) -> int:
        return shutil.disk_usage(self.root).free

    def load_remote_manifest(self, timeout: float = 15.0) -> list[OfflineAsset]:
        try:
            with urllib.request.urlopen(self.manifest_url, timeout=timeout) as response:
                payload = json.load(response)
            remote = [OfflineAsset(**record) for record in payload.get("assets", [])]
            remote_ids = {asset.asset_id for asset in remote}
            regional = [
                asset
                for asset in self.assets
                if asset.asset_id.startswith("gaia-") and asset.asset_id not in remote_ids
            ]
            self.assets = [*remote, *regional]
            self.refresh_installed()
            return self.assets
        except Exception as exc:
            raise LEAPSError(
                "OFFLINE_MANIFEST_UNAVAILABLE",
                "Offline data list could not be updated",
                "LEAPS kept the existing data list. Check the connection and try again.",
                ["Retry", "Continue with currently installed data"],
                technical_details=str(exc),
            ) from exc

    def refresh_installed(self) -> None:
        for asset in self.assets:
            asset.installed = self.validate(asset)

    def validate(self, asset: OfflineAsset) -> bool:
        """Validate a cached payload without requiring a network connection."""
        marker_path = self.root / asset.asset_id / "installed.json"
        if not marker_path.exists():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if str(marker.get("version")) != str(asset.version):
                return False
            filename = marker.get("filename") or asset.filename
            if not filename:
                return False
            payload = marker_path.parent / Path(str(filename)).name
            if not payload.is_file():
                return False
            expected = asset.sha256 or marker.get("sha256")
            return not expected or _sha256(payload) == expected
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def download_all(
        self,
        progress: Callable[[str, int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        required = self.total_estimated_bytes
        if required > self.free_bytes:
            raise LEAPSError(
                "OFFLINE_DISK_SPACE",
                "There is not enough free space",
                f"Offline data needs approximately {format_bytes(required)}, but only {format_bytes(self.free_bytes)} is free.",
                ["Choose another storage location", "Remove unused offline data"],
            )
        missing_urls = [asset.label for asset in self.assets if not asset.installed and not asset.url]
        if missing_urls:
            self.load_remote_manifest()
        for asset in self.assets:
            if asset.installed:
                continue
            if cancelled and cancelled():
                return
            self.download(asset, progress=progress, cancelled=cancelled)

    def download(
        self,
        asset: OfflineAsset,
        progress: Callable[[str, int, int], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Path:
        if not asset.url:
            raise LEAPSError(
                "OFFLINE_ASSET_URL_MISSING",
                f"{asset.label} is not available yet",
                "Refresh the offline data list and try again.",
                ["Refresh data list", "Continue online"],
            )
        folder = self.root / asset.asset_id
        folder.mkdir(parents=True, exist_ok=True)
        filename = (
            asset.filename or Path(urllib.parse.urlparse(asset.url).path).name or f"{asset.asset_id}.dat"
        )
        destination = folder / filename
        partial = destination.with_suffix(destination.suffix + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(asset.url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            response = urllib.request.urlopen(request, timeout=60)
            if offset and getattr(response, "status", None) != 206:
                offset = 0
            with response, partial.open("ab" if offset else "wb") as handle:
                total = int(response.headers.get("Content-Length", 0)) + offset
                current = offset
                while True:
                    if cancelled and cancelled():
                        return partial
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
                    current += len(block)
                    if progress:
                        progress(asset.label, current, total)
            if asset.sha256 and _sha256(partial) != asset.sha256:
                raise ValueError("Checksum did not match the asset manifest")
            partial.replace(destination)
            marker = {
                "asset_id": asset.asset_id,
                "version": asset.version,
                "sha256": asset.sha256,
                "filename": destination.name,
            }
            (folder / "installed.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
            asset.installed = True
            return destination
        except Exception as exc:
            raise LEAPSError(
                "OFFLINE_DOWNLOAD_FAILED",
                f"{asset.label} could not be downloaded",
                "The partial download was kept so Retry can resume it.",
                ["Retry download", "Check network and certificate settings"],
                technical_details=str(exc),
            ) from exc

    def remove(self, asset_id: str) -> None:
        folder = self.root / asset_id
        if folder.exists():
            shutil.rmtree(folder)
        self.refresh_installed()

    def add_gaia_region(
        self, ra: float, dec: float, radius: float, estimated_bytes: int = 75_000_000
    ) -> None:
        region_key = f"{ra:.4f}_{dec:+.4f}_{radius:.2f}".replace("+", "p").replace("-", "m")
        key = f"gaia-{region_key}"
        if any(asset.asset_id == key for asset in self.assets):
            return
        self.assets.append(
            OfflineAsset(
                key, f"Gaia DR3 region near {ra:.3f}, {dec:+.3f} ({radius:.2f} deg)", estimated_bytes, "DR3"
            )
        )


def format_bytes(value: int) -> str:
    size = float(value)
    for suffix in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or suffix == "TB":
            return f"{size:.1f} {suffix}"
        size /= 1024
    return f"{size:.1f} TB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
