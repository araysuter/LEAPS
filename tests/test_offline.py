from __future__ import annotations

import json
from pathlib import Path

from leaps.models import LEAPSError
from leaps.offline import OfflineDataManager


def test_offline_readiness_size_and_removal(tmp_path: Path) -> None:
    manager = OfflineDataManager(tmp_path)
    original = manager.total_estimated_bytes
    asset = manager.assets[0]
    marker = tmp_path / asset.asset_id / "installed.json"
    marker.parent.mkdir()
    payload = marker.parent / "catalogue.dat"
    payload.write_bytes(b"validated offline data")
    marker.write_text(json.dumps({"version": asset.version, "filename": payload.name}))
    manager.refresh_installed()
    assert manager.total_estimated_bytes == original - asset.estimated_bytes
    manager.remove(asset.asset_id)
    assert not marker.exists()
    assert manager.total_estimated_bytes == original


def test_gaia_is_added_by_project_region_not_as_full_catalogue(tmp_path: Path) -> None:
    manager = OfflineDataManager(tmp_path)
    manager.add_gaia_region(293.7328, 36.8155, 0.5)
    gaia = [asset for asset in manager.assets if asset.asset_id.startswith("gaia-")]
    assert len(gaia) == 1
    assert "region near" in gaia[0].label


def test_offline_download_stops_before_network_when_disk_is_full(tmp_path: Path, monkeypatch) -> None:
    manager = OfflineDataManager(tmp_path)
    monkeypatch.setattr(OfflineDataManager, "free_bytes", property(lambda self: 0))
    try:
        manager.download_all()
    except LEAPSError as failure:
        assert failure.code == "OFFLINE_DISK_SPACE"
    else:
        raise AssertionError("disk exhaustion did not produce a typed failure")
