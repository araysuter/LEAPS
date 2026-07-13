from __future__ import annotations

import json
from pathlib import Path

import pytest

from leaps.models import LEAPSError
from leaps.targets import ResolvedTarget, TargetNameResolver


def test_target_name_resolves_from_offline_nasa_snapshot_and_cache(tmp_path: Path) -> None:
    snapshot = tmp_path / "planets.json"
    snapshot.write_text(
        json.dumps(
            {
                "snapshot_date": "2026-07-01",
                "planets": [
                    {
                        "pl_name": "TrES-3 b",
                        "hostname": "TrES-3",
                        "ra": 268.029167,
                        "dec": 37.546111,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "target-cache.json"
    resolver = TargetNameResolver(cache_path=cache, nasa_snapshot=snapshot)
    result = resolver.resolve("TrES-3")
    assert result.name == "TrES-3 b"
    assert result.ra.startswith("17:52:")
    assert result.dec.startswith("+37:32:")
    assert result.source == "NASA Exoplanet Archive (offline)"

    snapshot.unlink()
    cached = TargetNameResolver(cache_path=cache).resolve("tres 3")
    assert cached == result


def test_target_name_not_found_is_a_recoverable_failure(tmp_path: Path, monkeypatch) -> None:
    resolver = TargetNameResolver(cache_path=tmp_path / "cache.json")
    monkeypatch.setattr(TargetNameResolver, "_from_simbad", staticmethod(lambda name: None))
    with pytest.raises(LEAPSError) as error:
        resolver.resolve("Definitely Not A Real Star")
    assert error.value.code == "TARGET_NAME_NOT_FOUND"
    assert "enter RA/DEC manually" in error.value.message


def test_target_lookup_fills_blank_coordinates_but_keeps_manual_values(qapp) -> None:
    from leaps.ui.pages import DataTargetPage

    page = DataTargetPage()
    requests: list[str] = []
    page.targetLookupRequested.connect(requests.append)
    page.name.setText("TrES-3")
    page._request_target_lookup()
    assert requests == ["TrES-3"]

    resolved = ResolvedTarget("TrES-3", "17:52:07.03", "+37:32:46.10", "SIMBAD via ExoClock")
    page.apply_target_resolution("TrES-3", resolved)
    assert page.ra.text() == resolved.ra
    assert page.dec.text() == resolved.dec
    assert "Coordinates found" in page.target_lookup_status.text()

    page.name.setText("WTS-2")
    page.ra.setText("19:00:00")
    page.dec.setText("+36:00:00")
    page._lookup_requested_name = ""
    page._request_target_lookup()
    page.apply_target_resolution("WTS-2", ResolvedTarget("WTS-2", "19:34:55.87", "+36:48:55.79", "SIMBAD"))
    assert page.ra.text() == "19:00:00"
    assert page.dec.text() == "+36:00:00"
    assert "existing RA/DEC were kept" in page.target_lookup_status.text()


def test_target_lookup_replaces_unchanged_saved_coordinates(qapp) -> None:
    from leaps.ui.pages import DataTargetPage

    page = DataTargetPage()
    page.name.setText("TrES-3")
    page.ra.setText("17:52:06.99")
    page.dec.setText("+37:32:46.15")
    page.mark_current_coordinates_as_saved()
    page.name.setText("WTS-2")
    page._request_target_lookup()
    result = ResolvedTarget("WTS-2", "19:34:55.87", "+36:48:55.79", "SIMBAD")
    page.apply_target_resolution("WTS-2", result)
    assert page.ra.text() == result.ra
    assert page.dec.text() == result.dec


def test_main_window_runs_target_lookup_off_the_ui_thread(qapp, monkeypatch) -> None:
    from PySide6.QtTest import QTest

    from leaps.ui.main_window import MainWindow

    window = MainWindow(demo=True)
    result = ResolvedTarget("TrES-3", "17:52:07.185", "+37:32:46.237", "test catalogue")
    monkeypatch.setattr(window.target_resolver, "resolve", lambda name: result)
    window.data_page.name.setText("TrES-3")
    window.data_page.ra.clear()
    window.data_page.dec.clear()
    window.data_page._request_target_lookup()
    for _ in range(50):
        qapp.processEvents()
        if window.target_lookup_runner.current is None:
            break
        QTest.qWait(10)
    qapp.processEvents()
    assert window.data_page.ra.text() == result.ra
    assert window.data_page.dec.text() == result.dec
    assert window.target_lookup_runner.current is None
    window.close()


def test_target_lookup_timeout_reports_no_coordinates_and_ignores_late_results(qapp) -> None:
    from leaps.ui.main_window import MainWindow

    window = MainWindow(demo=True)
    window.data_page.name.setText("Unknown target")
    window.data_page._lookup_requested_name = "Unknown target"
    window.data_page.target_lookup_status.setText("Looking up Unknown target…")
    window._active_target_lookup_name = "Unknown target"

    window._target_name_lookup_timed_out()
    assert "No coordinates found" in window.data_page.target_lookup_status.text()
    assert window.status_text.text() == "Ready"

    window._target_name_resolved(
        "Unknown target",
        ResolvedTarget("Late result", "12:00:00", "+10:00:00", "late catalogue"),
    )
    assert window.data_page.ra.text() != "12:00:00"
    assert "No coordinates found" in window.data_page.target_lookup_status.text()
    window.close()
