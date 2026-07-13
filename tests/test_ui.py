from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from leaps.fits_inventory import FrameRecord
from leaps.models import StageID, StageState, StageStatus
from leaps.project import ProjectWorkspace
from leaps.ui.main_window import MainWindow
from leaps.ui.pages import DataTargetPage, PlateSolvePage
from leaps.ui.widgets import FITSWorkspace, InfoButton, StageNavButton


def test_macos_app_icon_has_native_size_and_transparent_corners() -> None:
    path = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-app-icon.png"
    icon = Image.open(path)
    assert icon.size == (1024, 1024)
    assert icon.mode == "RGBA"
    alpha = icon.getchannel("A")
    assert all(alpha.getpixel(point) == 0 for point in ((0, 0), (1023, 0), (0, 1023), (1023, 1023)))
    assert alpha.getbbox() is not None


def test_demo_window_opens_selected_plate_solve_state(qapp) -> None:
    window = MainWindow(demo=True)
    window.show()
    qapp.processEvents()
    assert window.stack.currentWidget() is window.plate_page
    assert window.stage_buttons[StageID.PHOTOMETRY].active
    assert window.plate_page.inspector.retry.isEnabled()
    window.close()


def test_scientific_controls_have_accessible_information_buttons(qapp) -> None:
    window = MainWindow(demo=True)
    info_buttons = window.findChildren(InfoButton)
    assert len(info_buttons) >= 15
    assert all(button.toolTip().strip() for button in info_buttons)
    assert all(button.accessibleName() == "Information" for button in info_buttons)
    window.close()


def test_information_button_opens_immediately_on_hover_and_click(qapp) -> None:
    button = InfoButton("Explains this scientific setting.")
    button.show()
    qapp.processEvents()

    QTest.mouseMove(button, QPoint(button.width() // 2, button.height() // 2))
    qapp.processEvents()
    assert button._popover.isVisible()
    assert button._popover.label.text() == "Explains this scientific setting."
    assert button._popover.width() >= 360

    button._popover.hide()
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert button._popover.isVisible()
    assert button._pinned is True
    button._popover.hide()
    button.close()


def test_locked_stage_uses_prominent_solid_lock_without_enlarging_other_states(qapp) -> None:
    button = StageNavButton(StageID.FITTING, "Fitting")
    button.update_state(StageState(status=StageStatus.LOCKED, summary="Locked"))
    qapp.processEvents()
    assert button.status_icon.pixmap().size().width() == 30
    assert button.status_icon.width() == 36
    assert button.title.objectName() == "stageTitle"
    assert not button.isEnabled()

    button.update_state(StageState(status=StageStatus.READY, summary="Ready"))
    qapp.processEvents()
    assert button.status_icon.pixmap().size().width() == 23
    assert button.isEnabled()


def test_frame_assignment_cards_use_live_filename_classifiers(qapp) -> None:
    page = DataTargetPage()
    records = [
        FrameRecord("bias_001.fits", "bias", 1.0, "", (20, 20), 16, 0.0, "a"),
        FrameRecord("d_001.fits", "dark", 1.0, "", (20, 20), 16, 30.0, "b"),
        FrameRecord("flat_001.fits", "flat", 1.0, "", (20, 20), 16, 5.0, "c"),
        FrameRecord("image_001.fits", "science", 1.0, "", (20, 20), 16, 30.0, "d"),
    ]
    page.set_records(records)
    assert page.assignment_cards["bias"].count.text() == "1 selected"
    assert page.assignment_cards["dark"].count.text() == "0 selected"
    assert page.assignment_cards["flat"].count.text() == "1 selected"
    assert page.assignment_cards["science"].count.text() == "1 selected"
    assert page.counts.text() == "3 assigned · 1 unmatched"

    page.assignment_cards["dark"].classifier.setText("Dark, D")
    qapp.processEvents()
    assert page.assignment_cards["dark"].count.text() == "1 selected"
    assert page.counts.text() == "4 assigned · 0 unmatched"


def test_frame_assignment_counts_filename_matches_before_header_scan(qapp, tmp_path) -> None:
    for index in range(21):
        (tmp_path / f"bias_{index + 1:03d}.fits").touch()
    for index in range(7):
        (tmp_path / f"dark_60_{index + 1:03d}.fits").touch()
    for index in range(5):
        (tmp_path / f"flat_{index + 1:03d}.fits").touch()
    for index in range(12):
        (tmp_path / f"TrES-3_Cousins_R_{index + 1:03d}.fits").touch()

    page = DataTargetPage()
    page.preview_folder(tmp_path)
    assert page.assignment_cards["bias"].count.text() == "21 selected"
    assert page.assignment_cards["dark"].count.text() == "7 selected"
    assert page.assignment_cards["flat"].count.text() == "5 selected"
    assert page.assignment_cards["science"].count.text() == "0 selected"
    assert page.counts.text() == "33 assigned · 12 unmatched"

    page.assignment_cards["science"].classifier.setText("TrES-3")
    qapp.processEvents()
    assert page.assignment_cards["science"].count.text() == "12 selected"
    assert page.counts.text() == "45 assigned · 0 unmatched"


def test_saved_assignments_restore_without_rescanning(qapp) -> None:
    page = DataTargetPage()
    page.restore_project_assignments(
        {
            "bias": [f"bias_{index:03d}.fits" for index in range(21)],
            "dark": [f"dark_{index:03d}.fits" for index in range(7)],
            "flat": [f"flat_{index:03d}.fits" for index in range(5)],
            "science": [f"TrES-3_{index:03d}.fits" for index in range(354)],
        },
        {"bias": "Bias", "dark": "Dark", "flat": "Flat", "science": "TrES-3"},
        {"bias": False, "dark": False, "flat": False},
    )
    assert page.assignment_cards["bias"].count.text() == "21 selected"
    assert page.assignment_cards["science"].count.text() == "354 selected"
    assert page.counts.text() == "387 assigned · 0 unmatched"


def test_real_fits_workspace_pan_zoom_invert_and_reset(qapp, tmp_path) -> None:
    frame = tmp_path / "TrES-3_reference.fits"
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    fits.writeto(frame, data, overwrite=True)
    workspace = FITSWorkspace(tmp_path / "missing-demo.png")
    workspace.resize(800, 650)
    workspace.show()
    workspace.load_fits(frame, 1.2)
    qapp.processEvents()

    initial_key = workspace.image.image_item.pixmap().cacheKey()
    workspace.mode_buttons["invert"].click()
    qapp.processEvents()
    assert workspace.image.image_item.pixmap().cacheKey() != initial_key
    workspace.zoom.setCurrentText("200%")
    assert workspace.image.transform().m11() == 2.0
    workspace.reset_view()
    assert workspace.zoom.currentText() == "Fit"
    assert workspace.filename.text() == "FITS: TrES-3_reference.fits"
    assert workspace.dimensions.text() == "64 × 64 px"
    workspace.close()


def test_real_project_replaces_demo_target_and_restores_counts(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:06.99"
    project.manifest.target_dec = "+37:32:46.15"
    project.manifest.raw_files["bias"] = [f"bias_{index:03d}.fits" for index in range(21)]
    project.manifest.raw_files["science"] = [f"TrES-3_{index:03d}.fits" for index in range(4)]
    project.save()
    reduction = project.outputs_dir / "reduction"
    reduction.mkdir()
    frame = reduction / "r_00001_TrES-3.fits"
    fits.writeto(frame, np.arange(64 * 64, dtype=np.float32).reshape(64, 64))

    window = MainWindow(demo=True)
    window.set_project(project)
    qapp.processEvents()

    assert window.plate_page.inspector.target_name.text() == "TrES-3"
    assert window.plate_page.workspace.filename.text() == "FITS: r_00001_TrES-3.fits"
    assert window.plate_page.workspace.image.data is not None
    assert window.data_page.assignment_cards["bias"].count.text() == "21 selected"
    assert window.data_page.assignment_cards["science"].count.text() == "4 selected"
    window.close()


def test_photometry_comparisons_can_be_reviewed_before_run(qapp, tmp_path) -> None:
    page = PlateSolvePage(tmp_path / "missing-preview.png")
    page.clear_selection()
    page.set_target(12.0, 14.0, radius=8.0, label="TrES-3", verified=False)
    page.add_comparison(24.0, 26.0, radius=8.0)
    page.add_comparison(30.0, 32.0, radius=8.0)
    page.set_comparison_active(0, False)

    selected: list[list[tuple[float, float]]] = []
    page.runRequested.connect(lambda comparisons, _radius: selected.append(comparisons))
    page.inspector.run.click()
    qapp.processEvents()

    assert page.inspector.comparison_selection.text() == "Comparisons: 1 active · 2 selected"
    assert page.workspace.image.marker_items["comparison-1"][0].opacity() < 0.5
    assert selected == [[(30.0, 32.0)]]
    page.close()


def test_data_setup_orders_run_first_populates_target_and_highlights_errors(qapp) -> None:
    page = DataTargetPage()
    assert page.content_layout.itemAt(0).widget() is page.folder_card
    assert page.content_layout.itemAt(1).widget() is page.target_card
    assert page.content_layout.itemAt(2).widget() is page.frames_card

    record = FrameRecord(
        "TrES-3_Cousins_R_001.fits",
        "science",
        0.98,
        "",
        (2048, 2048),
        16,
        30.0,
        "checksum",
        "TrES-3 b",
        "17:52:07.00",
        "+37:32:46.20",
    )
    page.populate_target_from_records([record])
    assert page.name.text() == "TrES-3 b"
    assert page.ra.text() == "17:52:07.00"
    assert page.dec.text() == "+37:32:46.20"
    assert "Detected from FITS header" in page.target_source.text()

    page.show_error("Coordinates need attention", "target")
    qapp.processEvents()
    assert page.target_card.property("validationError") is True
    assert page.folder_card.property("validationError") is False
    page.clear_section_errors()
    assert page.target_card.property("validationError") is False
