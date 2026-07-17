from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QColor, QPixmap

from leaps.catalog import PlanetParameters
from leaps.ui.pages import FittingPage


def _parameters() -> PlanetParameters:
    return PlanetParameters(
        name="TrES-3b",
        ra="17:52:07.0185",
        dec="+37:32:46.237",
        period=1.306186314,
        mid_time=2457657.754796,
        rp_over_rs=0.16309,
        sma_over_rs=6.0,
        inclination=82.0,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=-0.19,
        temperature=5650.0,
        logg=4.58,
        source="ExoClock",
    )


def _ready_page() -> FittingPage:
    page = FittingPage()
    page.set_planet_candidates([_parameters()])
    page.set_observation_metadata("Cousins_R", 30.0)
    page.set_observatory_metadata(
        "SARA-ORM",
        28.76117,
        -17.87808,
        source="science FITS",
    )
    return page


def _write_preview(path: Path, color: str) -> None:
    pixmap = QPixmap(24, 24)
    pixmap.fill(QColor(color))
    assert pixmap.save(str(path))


def test_clear_preview_removes_stale_result_and_preserves_setup(qapp, tmp_path: Path) -> None:
    page = _ready_page()
    preview_path = tmp_path / "stale-preview.png"
    _write_preview(preview_path, "cyan")
    page.show_preview(
        preview_path,
        planet="TrES-3b",
        passband="COUSINS_R",
        residual_std=0.0028,
    )
    setup = page.values()

    message = "Comparison stars changed. Run Preview Fit to regenerate the fitting result."
    page.clear_preview(message)

    assert page.values() == setup
    assert page._preview_path is None
    assert page._preview_pixmap.isNull()
    assert page._rendered_preview_pixmap.isNull()
    assert page.preview_image.isHidden()
    assert page.preview_image.pixmap().isNull()
    assert not page.view_in_files.isEnabled()
    assert not page.full.isEnabled()
    assert page.preview.isEnabled()
    assert page.preview.property("primary") is True
    assert page.full.property("primary") is False
    assert page.message.text() == message
    page.close()


def test_clear_preview_requires_a_fresh_preview_before_full_fit(qapp, tmp_path: Path) -> None:
    page = _ready_page()
    stale_path = tmp_path / "stale-preview.png"
    _write_preview(stale_path, "cyan")
    page.show_preview(
        stale_path,
        planet="TrES-3b",
        passband="COUSINS_R",
        residual_std=None,
    )

    page.clear_preview("The approved light curve changed.")
    page._refresh_actions()
    assert not page.full.isEnabled()

    fresh_path = tmp_path / "fresh-preview.png"
    _write_preview(fresh_path, "magenta")
    page.show_preview(
        fresh_path,
        planet="TrES-3b",
        passband="COUSINS_R",
        residual_std=0.0015,
    )

    assert page._preview_path == fresh_path
    assert not page._preview_pixmap.isNull()
    assert page.full.isEnabled()
    assert page.view_in_files.isEnabled()
    assert page.preview.property("primary") is False
    assert page.full.property("primary") is True
    page.close()
