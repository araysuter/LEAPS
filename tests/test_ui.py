from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from PySide6.QtCore import QPoint, QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QWidget

import leaps.ui.main_window as main_window_module
from leaps.fits_inventory import FITSInventory, FrameRecord
from leaps.models import (
    JobStatus,
    LEAPSError,
    ProjectManifest,
    StageEvent,
    StageID,
    StageState,
    StageStatus,
)
from leaps.project import ProjectWorkspace
from leaps.science import InspectionResult
from leaps.ui.main_window import MainWindow, ProjectResetDialog
from leaps.ui.pages import (
    ComparisonStarsPage,
    DataTargetPage,
    FITSHeaderDialog,
    FittingPage,
    InspectionPage,
    LightCurvePage,
    PlateSolvePage,
    ProcessingPage,
    SecondaryEclipsePage,
)
from leaps.ui.widgets import FITSWorkspace, InfoButton, StageNavButton


def test_macos_app_icon_has_native_size_and_transparent_corners() -> None:
    path = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-app-icon.png"
    icon = Image.open(path)
    assert icon.size == (1024, 1024)
    assert icon.mode == "RGBA"
    alpha = icon.getchannel("A")
    assert all(alpha.getpixel(point) == 0 for point in ((0, 0), (1023, 0), (0, 1023), (1023, 1023)))
    assert alpha.getbbox() is not None

    assets = path.parent
    assert (assets / "leaps-logo-source.png").is_file()
    with Image.open(assets / "leaps-app-icon.icns") as native_macos_icon:
        assert native_macos_icon.format == "ICNS"
        assert native_macos_icon.size == (1024, 1024)
    with Image.open(assets / "leaps-app-icon.ico") as native_windows_icon:
        assert native_windows_icon.format == "ICO"
        assert native_windows_icon.size == (256, 256)


def test_shared_leaps_mark_is_centered_in_its_tile() -> None:
    path = Path(__file__).parents[1] / "leaps" / "assets" / "leaps-mark.png"
    mark = Image.open(path)
    assert mark.size == (512, 512)
    assert mark.mode == "RGBA"
    assert mark.getpixel((0, 0))[3] == 0
    pixels = np.asarray(mark.convert("RGB"), dtype=float)
    luminosity = pixels.mean(axis=2)
    saturation = pixels.max(axis=2) - pixels.min(axis=2)
    white = (luminosity > 120) & (saturation < 55)
    cyan = (
        (pixels[:, :, 2] > 100)
        & (pixels[:, :, 1] > 75)
        & (pixels[:, :, 0] < 80)
        & ((pixels[:, :, 1] - pixels[:, :, 0]) > 25)
    )
    y, x = np.where(white | cyan)
    mark_center = ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)
    tile_center = ((pixels.shape[1] - 1) / 2, (pixels.shape[0] - 1) / 2)
    assert abs(mark_center[0] - tile_center[0]) <= 1
    assert abs(mark_center[1] - tile_center[1]) <= 1


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


def test_sidebar_does_not_show_inert_collapse_control(qapp) -> None:
    window = MainWindow(demo=True)
    assert not any(
        button.toolTip() == "Collapse the workflow sidebar."
        for button in window.sidebar.findChildren(QPushButton)
    )
    window.close()


def test_sidebar_scrolls_workflow_rows_without_hiding_tools_at_minimum_height(qapp) -> None:
    window = MainWindow(demo=True)
    window.resize(window.minimumWidth(), window.minimumHeight())
    window.show()
    qapp.processEvents()

    stages = list(window.stage_buttons.values())
    scroll_bar = window.workflow_scroll.verticalScrollBar()
    assert scroll_bar.maximum() > 0
    assert all(button.parentWidget() is window.workflow_scroll.widget() for button in stages)
    assert all(button.height() == 66 for button in stages)
    assert all(
        following.geometry().top() > previous.geometry().bottom()
        for previous, following in zip(stages, stages[1:])
    )
    assert all(button.isVisible() for button in window.tool_buttons.values())
    assert all(
        not window.workflow_scroll.isAncestorOf(button)
        for button in window.tool_buttons.values()
    )
    assert (
        max(button.geometry().bottom() for button in window.tool_buttons.values())
        < window.sidebar.height()
    )

    scroll_bar.setValue(0)
    window.open_stage(StageID.SECONDARY_ECLIPSE)
    qapp.processEvents()
    active = window.stage_buttons[StageID.SECONDARY_ECLIPSE]
    active_top = active.mapTo(window.workflow_scroll.viewport(), QPoint(0, 0)).y()
    assert scroll_bar.value() > 0
    assert active_top >= 0
    assert active_top + active.height() <= window.workflow_scroll.viewport().height()

    window.resize(1487, 1018)
    qapp.processEvents()
    assert scroll_bar.maximum() == 0
    window.close()


def test_light_curve_is_required_workflow_stage_not_a_tool(qapp) -> None:
    window = MainWindow(demo=True)
    stages = list(window.stage_buttons)
    assert stages.index(StageID.LIGHT_CURVE) == stages.index(StageID.PHOTOMETRY) + 1
    assert stages.index(StageID.FITTING) == stages.index(StageID.LIGHT_CURVE) + 1
    assert stages.index(StageID.SECONDARY_ECLIPSE) == stages.index(StageID.FITTING) + 1
    assert "light_curve" not in window.tool_buttons
    assert window.pages[StageID.LIGHT_CURVE] is window.light_curve_page
    assert window.pages[StageID.SECONDARY_ECLIPSE] is window.secondary_eclipse_page
    window.close()


def test_secondary_eclipse_page_needs_full_fit_context_before_analysis(qapp) -> None:
    page = SecondaryEclipsePage()
    assert not page.analyze.isEnabled()

    from leaps.catalog import PlanetParameters

    parameters = PlanetParameters(
        name="WASP-12 b",
        ra="06:30:32.79",
        dec="+29:40:20.3",
        period=1.09142,
        mid_time=2454508.97682,
        rp_over_rs=0.117,
        sma_over_rs=3.0,
        inclination=83.4,
        eccentricity=0.0,
        periastron=0.0,
        metallicity=0.3,
        temperature=6300.0,
        logg=4.2,
        source="ExoClock",
    )
    page.set_fit_context(parameters, passband="COUSINS_R", duration_hours=2.9)

    assert page.analyze.isEnabled()
    assert page.expected_phase.value() == 0.5
    assert page.duration_hours.value() == 2.9
    assert "WASP-12 b" in page.fit_context.text()
    page.close()


def test_secondary_eclipse_page_reloads_saved_setup_and_flags_strong_control(qapp, tmp_path) -> None:
    page = SecondaryEclipsePage()
    from leaps.catalog import PlanetParameters

    page.set_fit_context(
        PlanetParameters(
            name="WASP-18 b",
            ra="01:37:25.07",
            dec="-45:40:40.1",
            period=0.94145,
            mid_time=2458354.45,
            rp_over_rs=0.1018,
            sma_over_rs=3.48,
            inclination=83.5,
            eccentricity=0.0,
            periastron=0.0,
            metallicity=0.0,
            temperature=6400.0,
            logg=4.3,
            source="Test",
        )
    )
    page.show_saved_result(
        {
            "light_curve": "gaussian",
            "baseline": "quadratic",
            "expected_phase": 0.5003,
            "duration_hours": 2.21,
            "message": "A positive fixed-phase eclipse is recovered.",
            "outcome": "candidate",
            "outcome_label": "Candidate signal · independent check required",
            "depth_ppm": 377.0,
            "depth_uncertainty_ppm": 14.0,
            "significance": 26.3,
            "red_noise_beta": 1.76,
            "local_points": 51_507,
            "in_eclipse_points": 10_501,
            "event_count": 170,
            "control_significance": 5.6,
        },
        tmp_path / "missing-preview.png",
    )

    assert page.light_curve.currentData() == "gaussian"
    assert page.baseline.currentData() == "quadratic"
    assert page.expected_phase.value() == 0.5003
    assert page.duration_hours.value() == 2.21
    assert "Nearby control phase: 5.6σ" in page.message.text()
    assert "review" in page.metric_values["control"].text()
    page.close()


def test_inspection_page_links_plots_and_saves_individual_frame_decisions(
    qapp, tmp_path: Path
) -> None:
    reduction = tmp_path / "reduction"
    reduction.mkdir()
    frames = []
    for index in range(3):
        name = f"r_{index:05d}.fits"
        fits.writeto(
            reduction / name,
            np.full((32, 32), 100.0 + index, dtype=np.float32),
            fits.Header({"BITPIX": -32, "HOPSPSF": 2.0}),
        )
        frames.append(
            {
                "file": name,
                "index": index + 1,
                "jd": 2460000.0 + index / 1440,
                "elapsed_hours": index / 60,
                "sky": 100.0 + index,
                "sky_std": 5.0,
                "psf": 2.0 + index / 10,
                "hard_excluded": index == 2,
                "hard_exclusion_reason": "Invalid PSF" if index == 2 else "",
                "manual_excluded": False,
                "excluded": index == 2,
                "suggest_exclude": index == 1,
            }
        )
    result = InspectionResult(
        frames,
        median_sky=101.0,
        median_psf=2.1,
        included_count=2,
        excluded_count=1,
        suggested_count=1,
        time_axis="elapsed_hours",
    )
    page = InspectionPage(tmp_path / "missing-preview.png")
    page.resize(1200, 800)
    page.show()
    page.set_result(result, reduction)
    qapp.processEvents()

    assert page.run.text() == "Run Inspection Again"
    assert page.psf_plot.time_axis == "elapsed_hours"
    assert page.include_review.property("activeToggle") is True
    assert page.exclude_review.property("cancelActive") is True
    drafts = []
    page.draftChanged.connect(drafts.append)
    page.exclude_review.click()
    assert drafts[-1]["r_00001.fits"] is True
    assert page.sky_plot.frames[1]["excluded"] is True
    assert page.sky_plot.frames[0]["excluded"] is False
    assert page.sky_plot.frames[2]["excluded"] is True
    page.include_review.click()
    assert drafts[-1]["r_00001.fits"] is False
    assert page.sky_plot.frames[1]["excluded"] is False
    assert page.sky_plot.frames[2]["excluded"] is True

    point = page.sky_plot._points[1][1].toPoint()
    QTest.mouseClick(page.sky_plot, Qt.MouseButton.LeftButton, pos=point)
    qapp.processEvents()
    assert page.selected_index == 1
    assert page.psf_plot.selected_index == 1
    assert "Suggested for review" in page.frame_status.text()

    page.exclude.click()
    assert drafts[-1]["r_00001.fits"] is True
    assert page.exclude.property("cancelActive") is True
    assert page.sky_plot.frames[1]["excluded"] is True
    page.include.click()
    assert page.include.property("activeToggle") is True

    page.select_frame(2)
    assert not page.include.isEnabled()
    assert not page.exclude.isEnabled()
    assert "Invalid PSF" in page.frame_status.text()
    page.close()


def test_inspection_change_invalidates_downstream_and_only_clears_changed_reference(
    qapp, tmp_path: Path
) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings=settings)
    project = ProjectWorkspace.create(tmp_path / "project")
    for stage in (
        StageID.ALIGNMENT,
        StageID.PHOTOMETRY,
        StageID.LIGHT_CURVE,
        StageID.FITTING,
        StageID.SECONDARY_ECLIPSE,
    ):
        project.manifest.stages[stage.value] = StageState(
            status=StageStatus.COMPLETE,
            summary="Complete",
        )
    project.manifest.settings.update(
        {
            "plate_solution": {"target_xy": [10, 20]},
            "photometry": {"target": [10, 20]},
            "fitting_setup": {"manual": True},
            "light_curve_review": {"active_comparisons": [True]},
        }
    )
    project.save()
    window.project = project

    window._invalidate_after_inspection("r_00001.fits", "r_00001.fits")
    assert project.manifest.stages[StageID.ALIGNMENT.value].status == StageStatus.READY
    assert project.manifest.stages[StageID.PHOTOMETRY.value].status == StageStatus.LOCKED
    assert "plate_solution" in project.manifest.settings
    assert "photometry" in project.manifest.settings
    assert "fitting_setup" in project.manifest.settings
    assert "light_curve_review" not in project.manifest.settings

    window._invalidate_after_inspection("r_00001.fits", "r_00002.fits")
    assert "plate_solution" not in project.manifest.settings
    assert "photometry" not in project.manifest.settings
    assert "fitting_setup" in project.manifest.settings
    window.close()


def test_light_curve_page_defaults_comparisons_on_and_keeps_one_active(qapp, tmp_path) -> None:
    preview = tmp_path / "light-curves.png"
    image = Image.new("RGB", (800, 600), "#0b2638")
    image.save(preview)
    result = type(
        "ReviewResult",
        (),
        {
            "active_comparisons": [True, True, True],
            "preview_path": preview,
            "frame_count": 354,
            "curves": [
                type("Curve", (), {"label": label, "missing_frames": 0})
                for label in ("Target", "C1", "C2", "C3")
            ],
        },
    )()
    page = LightCurvePage()
    page.set_review(result)
    assert page.active_comparisons() == [True, True, True]
    assert page.continue_button.isEnabled()

    selections = []
    page.selectionChanged.connect(selections.append)
    page.comparison_checks[1].setChecked(False)
    assert selections[-1] == [True, False, True]
    page.comparison_checks[0].setChecked(False)
    page.comparison_checks[2].setChecked(False)
    assert page.active_comparisons() == [False, False, True]
    assert "At least one" in page.message.text()
    page.close()


def test_information_button_opens_immediately_on_hover_and_click(qapp) -> None:
    host = QWidget()
    layout = QHBoxLayout(host)
    button = InfoButton("Explains this scientific setting.")
    elsewhere = QPushButton("Elsewhere")
    layout.addWidget(button)
    layout.addWidget(elsewhere)
    host.show()
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
    assert button._pinned is False

    QTest.mouseMove(elsewhere, QPoint(elsewhere.width() // 2, elsewhere.height() // 2))
    qapp.processEvents()
    assert not button._popover.isVisible()
    button._popover.hide()
    host.close()


def test_fit_preview_reveal_selects_file_in_platform_file_manager(
    tmp_path, monkeypatch
) -> None:
    preview = tmp_path / "fit-preview.png"
    preview.write_bytes(b"preview")
    calls = []
    monkeypatch.setattr(
        main_window_module,
        "_start_detached",
        lambda program, arguments: calls.append((program, arguments)) or True,
    )

    for platform, program, arguments in (
        ("darwin", "/usr/bin/open", ["-R", str(preview)]),
        ("win32", "explorer.exe", ["/select,", str(preview)]),
    ):
        monkeypatch.setattr(main_window_module.sys, "platform", platform)
        main_window_module._reveal_in_file_manager(preview)
        assert calls[-1] == (program, arguments)


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


def test_running_stage_spins_only_in_sidebar(qapp) -> None:
    button = StageNavButton(StageID.REDUCTION, "Reduction")
    button.update_state(StageState(status=StageStatus.RUNNING, summary="Processing"))
    assert button._spinner_timer.isActive()
    before = button._spinner_phase
    button._advance_spinner()
    assert button._spinner_phase != before

    button.update_state(StageState(status=StageStatus.COMPLETE, summary="Complete"))
    assert not button._spinner_timer.isActive()
    assert button._spinner_phase == 0


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


def test_data_target_filter_matches_hops_and_never_autoselects_from_fits(qapp) -> None:
    page = DataTargetPage()
    labels = [page.filter.itemText(index) for index in range(page.filter.count())]
    values = [page.filter.itemData(index) for index in range(page.filter.count())]

    assert labels == [
        "No filter chosen",
        "Clear",
        "Luminance",
        "U",
        "B",
        "V",
        "R",
        "I",
        "H",
        "J",
        "K",
        "Astrodon ExoPlanet-BB",
        "u'",
        "g'",
        "r'",
        "z'",
        "i'",
    ]
    assert values[0] is None
    assert values[1:] == [
        "clear",
        "luminance",
        "JOHNSON_U",
        "JOHNSON_B",
        "JOHNSON_V",
        "COUSINS_R",
        "COUSINS_I",
        "2mass_h",
        "2mass_j",
        "2mass_ks",
        "exoplanets_bb",
        "sdss_u",
        "sdss_g",
        "sdss_r",
        "sdss_z",
        "sdss_i",
    ]

    page.folder.setText("/tmp/example")
    page.set_records(
        [
            FrameRecord(
                "image_001.fits",
                "science",
                1.0,
                "",
                (20, 20),
                16,
                30.0,
                "checksum",
                filter_name="COUSINS_R",
                raw_filter="R",
            )
        ]
    )

    assert page.filter.currentIndex() == 0
    assert page.filter.currentData() is None
    assert "FITS header reports: R" in page.detected_filter.text()
    page.close()


def test_fits_header_viewer_reads_every_hdu_without_modifying_source(qapp, tmp_path) -> None:
    path = tmp_path / "image_001.fits"
    primary = fits.PrimaryHDU(np.ones((4, 4)), header=fits.Header({"FILTER": "R"}))
    extension = fits.ImageHDU(np.zeros((2, 2)), name="CALIBRATION")
    extension.header["EXPTIME"] = 30.0
    fits.HDUList([primary, extension]).writeto(path)
    before = path.read_bytes()

    text = FITSHeaderDialog.read_headers(path)

    assert "HDU 0 — PRIMARY" in text
    assert "FILTER" in text
    assert "HDU 1 — CALIBRATION" in text
    assert "EXPTIME" in text
    assert path.read_bytes() == before


def test_data_target_header_button_opens_first_assigned_science_frame(
    qapp, tmp_path, monkeypatch
) -> None:
    for index in (1, 2):
        fits.writeto(
            tmp_path / f"image_{index:03d}.fits",
            np.ones((4, 4)),
            header=fits.Header({"OBJECT": f"Frame {index}"}),
        )
    page = DataTargetPage()
    page.folder.setText(str(tmp_path))
    page.set_records(FITSInventory(tmp_path).discover())
    captured: dict[str, str] = {}

    def capture(dialog) -> None:
        captured["title"] = dialog.windowTitle()
        captured["text"] = dialog.header_text.toPlainText()

    monkeypatch.setattr(FITSHeaderDialog, "exec", capture)
    page._view_first_science_header()

    assert "image_001.fits" in captured["title"]
    assert "Frame 1" in captured["text"]
    assert "Frame 2" not in captured["text"]
    page.close()


def test_data_target_requires_explicit_filter_before_confirmation(qapp, tmp_path) -> None:
    window = MainWindow(demo=True)
    window.save_data_target(
        {
            "root": str(tmp_path),
            "target_name": "TrES-3",
            "ra": "17:52:07.00",
            "dec": "+37:32:46.20",
            "filter": None,
            "waivers": {"bias": True, "dark": True, "flat": True},
            "assignments": {
                "science": ["image_001.fits"],
                "bias": [],
                "dark": [],
                "dark_flat": [],
                "flat": [],
                "unknown": [],
            },
            "frame_classifiers": {},
        }
    )

    assert "Choose the observation filter" in window.data_page.validation.text()
    assert window.data_page.target_card.property("validationError") is True
    assert not ProjectWorkspace.has_project(tmp_path)
    window.close()


def _saved_data_target_values(root: Path, filter_name: str) -> dict[str, object]:
    return {
        "root": str(root),
        "target_name": "TrES-3",
        "ra": "17:52:07.00",
        "dec": "+37:32:46.20",
        "filter": filter_name,
        "waivers": {"bias": True, "dark": True, "flat": True},
        "assignments": {
            "science": ["image_001.fits"],
            "bias": [],
            "dark": [],
            "dark_flat": [],
            "flat": [],
            "unknown": [],
        },
        "frame_classifiers": {},
    }


def _project_ready_for_inspection(root: Path) -> ProjectWorkspace:
    project = ProjectWorkspace.create(root, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:07.00"
    project.manifest.target_dec = "+37:32:46.20"
    project.manifest.raw_files["science"] = ["image_001.fits"]
    project.manifest.settings["filter"] = "COUSINS_R"
    project.set_stage(
        StageID.DATA_TARGET,
        StageStatus.COMPLETE,
        "Target selected",
        progress=1.0,
    )
    project.set_stage(
        StageID.REDUCTION,
        StageStatus.COMPLETE,
        "Complete",
        progress=1.0,
    )
    return project


def test_resume_stage_follows_each_persisted_workflow_boundary() -> None:
    stages = list(StageID)
    for resume_index, expected in enumerate(stages):
        manifest = ProjectManifest()
        for index, stage in enumerate(stages):
            manifest.stages[stage.value] = StageState(
                status=(
                    StageStatus.COMPLETE
                    if index < resume_index
                    else StageStatus.READY
                    if index == resume_index
                    else StageStatus.LOCKED
                )
            )
        assert MainWindow._resume_stage(manifest) == expected

    for stage in stages:
        manifest.stages[stage.value] = StageState(status=StageStatus.COMPLETE)
    assert MainWindow._resume_stage(manifest) == StageID.SECONDARY_ECLIPSE


def test_saved_filter_restores_without_using_detected_fits_value(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.settings["filter"] = "sdss_z"
    project.manifest.settings["observation_metadata"] = {
        "filter": "COUSINS_R",
        "filter_status": "detected",
    }
    project.save()
    window = MainWindow(demo=True)

    window.set_project(project)

    assert window.data_page.filter.currentData() == "sdss_z"
    assert "COUSINS_R" in window.data_page.detected_filter.text()
    window.close()


def test_changing_filter_after_reduction_locks_every_downstream_stage(
    qapp, tmp_path
) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:07.00"
    project.manifest.target_dec = "+37:32:46.20"
    project.manifest.raw_files["science"] = ["image_001.fits"]
    project.manifest.settings["filter"] = "COUSINS_R"
    for stage in StageID:
        project.manifest.stages[stage.value] = StageState(
            status=StageStatus.COMPLETE,
            summary="Complete",
        )
    project.manifest.stages[StageID.ALIGNMENT.value] = StageState(
        status=StageStatus.READY,
        summary="Ready",
    )
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)

    window.save_data_target(_saved_data_target_values(tmp_path, "JOHNSON_V"))

    updated = window.project
    assert updated is not None
    assert updated.manifest.settings["filter"] == "JOHNSON_V"
    assert updated.manifest.stages[StageID.REDUCTION.value].status == StageStatus.READY
    assert "Filter changed" in updated.manifest.stages[StageID.REDUCTION.value].summary
    for stage in (
        StageID.INSPECTION,
        StageID.ALIGNMENT,
        StageID.PHOTOMETRY,
        StageID.LIGHT_CURVE,
        StageID.FITTING,
        StageID.SECONDARY_ECLIPSE,
    ):
        assert updated.manifest.stages[stage.value].status == StageStatus.LOCKED
    window.close()


def test_reconfirming_same_filter_preserves_completed_processing(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:07.00"
    project.manifest.target_dec = "+37:32:46.20"
    project.manifest.raw_files["science"] = ["image_001.fits"]
    project.manifest.settings["filter"] = "COUSINS_R"
    for stage in StageID:
        project.manifest.stages[stage.value] = StageState(
            status=StageStatus.COMPLETE,
            summary="Complete",
        )
    project.manifest.stages[StageID.ALIGNMENT.value] = StageState(
        status=StageStatus.READY,
        summary="Ready",
    )
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)

    window.save_data_target(_saved_data_target_values(tmp_path, "COUSINS_R"))

    updated = window.project
    assert updated is not None
    assert updated.manifest.stages[StageID.REDUCTION.value].status == StageStatus.COMPLETE
    assert updated.manifest.stages[StageID.FITTING.value].status == StageStatus.COMPLETE
    assert window.stack.currentWidget() is window.alignment_page
    window.close()


def test_confirming_a_selected_existing_project_resumes_without_overwriting_it(
    qapp, tmp_path
) -> None:
    project = _project_ready_for_inspection(tmp_path)
    saved_manifest = project.manifest_path.read_bytes()
    window = MainWindow(demo=True)

    window.save_data_target(
        {
            "root": str(tmp_path),
            "target_name": "Scan-derived value that must not replace the project",
            "ra": "",
            "dec": "",
            "filter": None,
            "waivers": {"bias": False, "dark": False, "flat": False},
            "assignments": {
                "science": [],
                "bias": [],
                "dark": [],
                "dark_flat": [],
                "flat": [],
                "unknown": [],
            },
            "frame_classifiers": {},
        }
    )

    assert window.project is not None
    assert window.project.manifest.project_id == project.manifest.project_id
    assert window.stack.currentWidget() is window.inspection_page
    assert window.data_page.name.text() == "TrES-3"
    assert project.manifest_path.read_bytes() == saved_manifest
    window.close()


def test_choosing_an_existing_project_folder_resumes_without_rescanning(
    qapp, tmp_path, monkeypatch
) -> None:
    root = tmp_path / "WASP-19" / "LEAPS"
    project = _project_ready_for_inspection(root)
    saved_manifest = project.manifest_path.read_bytes()
    window = MainWindow(demo=True)
    starts: list[object] = []
    monkeypatch.setattr(window.runner, "start", lambda *args, **kwargs: starts.append(args))

    window.scan_folder(root)

    assert not starts
    assert window.project is not None
    assert window.project.root == root
    assert window.stack.currentWidget() is window.inspection_page
    assert project.manifest_path.read_bytes() == saved_manifest
    window.close()


def test_incomplete_project_workspace_selection_scans_the_observing_run(
    qapp, tmp_path, monkeypatch
) -> None:
    root = tmp_path / "incomplete"
    project = ProjectWorkspace.create(root, "Incomplete")
    window = MainWindow(demo=True)
    window.data_page.folder.setText(str(project.workspace))
    scanned_roots: list[Path] = []

    class RecordingInventory:
        def __init__(self, selected: Path) -> None:
            scanned_roots.append(selected)

        def discover(self) -> list[FrameRecord]:
            return []

    monkeypatch.setattr(main_window_module, "FITSInventory", RecordingInventory)
    monkeypatch.setattr(
        window.runner,
        "start",
        lambda function, *args, **kwargs: function(),
    )

    window.scan_folder(project.workspace)

    assert scanned_roots == [root.resolve()]
    assert window.data_page.folder.text() == str(root.resolve())
    assert window.project is None
    window.close()


def test_reduction_receives_confirmed_project_filter(qapp, tmp_path, monkeypatch) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.settings["filter"] = "JOHNSON_B"
    project.save()
    window = MainWindow(demo=True)
    window.set_project(project)
    monkeypatch.setattr(window, "_ensure_runner_idle", lambda *_args: True)
    captured: dict[str, object] = {}

    def capture_start(_function, *_args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(window.runner, "start", capture_start)
    window.run_stage(StageID.REDUCTION)

    assert captured["config"].filter_name == "JOHNSON_B"
    window.close()


def test_data_target_page_exposes_tess_light_curve_import(qapp) -> None:
    page = DataTargetPage()
    assert page.import_tess.text() == "Import TESS light curves"
    assert "PDCSAP" in page.import_tess.toolTip()
    page.show_tess_import_result("Imported 1,234 TESS points.")
    assert not page.tess_import_status.isHidden()
    assert "1,234" in page.tess_import_status.text()
    page.close()


def test_observing_run_picker_retries_selected_external_folder_after_access_denial(
    qapp, tmp_path, monkeypatch
) -> None:
    page = DataTargetPage()
    starts: list[str] = []
    selections = iter((str(tmp_path), str(tmp_path)))
    preflight_calls: list[Path] = []

    def choose(_parent, _title, start, _options):
        starts.append(start)
        return next(selections)

    def preflight(folder: Path) -> None:
        preflight_calls.append(folder)
        if len(preflight_calls) == 1:
            raise LEAPSError(
                "OBSERVING_RUN_ACCESS_DENIED",
                "LEAPS cannot access the observing run",
                "The selected folder is not readable.",
                ["Choose the folder again to grant access"],
                stage=StageID.DATA_TARGET,
            )

    monkeypatch.setattr("leaps.ui.pages.QFileDialog.getExistingDirectory", choose)
    monkeypatch.setattr("leaps.ui.pages.preflight_observing_run_access", preflight)
    monkeypatch.setattr(page, "_confirm_folder_access_retry", lambda _folder, _failure: True)
    scanned: list[Path] = []
    page.scanRequested.connect(scanned.append)

    page._choose_folder()

    assert preflight_calls == [tmp_path, tmp_path]
    assert starts[1] == str(tmp_path)
    assert scanned == [tmp_path]
    assert page.folder.text() == str(tmp_path)
    page.close()


def test_folder_access_prompt_explains_external_drive_permission(qapp, tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeMessageBox:
        class Icon:
            Warning = object()

        class ButtonRole:
            RejectRole = object()
            AcceptRole = object()

        def __init__(self, _parent) -> None:
            self.grant = None

        def setIcon(self, _icon) -> None:
            pass

        def setWindowTitle(self, title: str) -> None:
            captured["title"] = title

        def setText(self, message: str) -> None:
            captured["message"] = message

        def setInformativeText(self, message: str) -> None:
            captured["information"] = message

        def setDetailedText(self, details: str) -> None:
            captured["details"] = details

        def addButton(self, label: str, _role):
            button = object()
            if "Grant Access" in label:
                self.grant = button
            return button

        def setDefaultButton(self, _button) -> None:
            pass

        def setEscapeButton(self, _button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self.grant

    monkeypatch.setattr("leaps.ui.pages.QMessageBox", FakeMessageBox)
    monkeypatch.setattr("leaps.ui.pages.sys.platform", "darwin")
    page = DataTargetPage()
    failure = LEAPSError(
        "OBSERVING_RUN_ACCESS_DENIED",
        "LEAPS cannot access the observing run",
        "The selected folder is not readable.",
        ["Choose the folder again"],
        stage=StageID.DATA_TARGET,
        technical_details="PermissionError: Operation not permitted",
    )

    assert page._confirm_folder_access_retry(tmp_path / "External SSD", failure)
    assert captured["title"] == "Folder access required"
    assert "External SSD" in str(captured["message"])
    assert "external SSD" in str(captured["information"])
    assert "Files and Folders" in str(captured["information"])
    page.close()


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


def test_frame_assignments_do_not_show_calibration_waiver_checkboxes(qapp) -> None:
    page = DataTargetPage()
    labels = [checkbox.text() for checkbox in page.frames_card.findChildren(QCheckBox)]
    assert not any(label.startswith("Continue without") for label in labels)


def test_missing_calibration_confirmation_has_cancel_and_acknowledge(qapp, monkeypatch) -> None:
    buttons: list[str] = []
    messages: list[str] = []

    class FakeMessageBox:
        acknowledge_next = True

        class Icon:
            Warning = object()

        class ButtonRole:
            RejectRole = object()
            AcceptRole = object()

        def __init__(self, parent) -> None:
            self.acknowledge = None
            self.cancel = None

        def setIcon(self, icon) -> None:
            pass

        def setWindowTitle(self, title: str) -> None:
            assert title == "Missing calibration frames"

        def setText(self, text: str) -> None:
            messages.append(text)

        def setInformativeText(self, text: str) -> None:
            assert "reduce the quality" in text

        def addButton(self, text: str, role):
            button = object()
            buttons.append(text)
            if text == "Acknowledge":
                self.acknowledge = button
            else:
                self.cancel = button
            return button

        def setDefaultButton(self, button) -> None:
            pass

        def setEscapeButton(self, button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self.acknowledge if self.acknowledge_next else self.cancel

    monkeypatch.setattr("leaps.ui.main_window.QMessageBox", FakeMessageBox)
    window = MainWindow(demo=True)
    assert window._confirm_missing_calibration_frames(["bias", "dark", "flat"])
    assert buttons == ["Cancel", "Acknowledge"]
    assert "Bias, Darks, Flats" in messages[0]
    FakeMessageBox.acknowledge_next = False
    assert not window._confirm_missing_calibration_frames(["bias"])
    window.close()


def test_real_fits_workspace_pan_zoom_invert_flip_and_reset(qapp, tmp_path) -> None:
    frame = tmp_path / "TrES-3_reference.fits"
    data = np.arange(64 * 64, dtype=np.float32).reshape(64, 64)
    fits.writeto(frame, data, overwrite=True)
    workspace = FITSWorkspace(tmp_path / "missing-demo.png")
    workspace.resize(800, 650)
    workspace.show()
    workspace.load_fits(frame, 1.2)
    qapp.processEvents()

    assert workspace.toolbar.height() == 62
    assert workspace.toolbar.layout().contentsMargins().bottom() == 11
    assert workspace.toolbar.layout().spacing() == 2
    initial_key = workspace.image.image_item.pixmap().cacheKey()
    workspace.mode_buttons["invert"].click()
    qapp.processEvents()
    assert workspace.image.image_item.pixmap().cacheKey() != initial_key
    workspace.place_target_marker(10.0, 20.0)
    original_center = workspace.image.marker_items["target"][0].rect().center()
    assert original_center.x() == 10.0
    assert original_center.y() == 43.0

    workspace.mode_buttons["flip x"].click()
    qapp.processEvents()
    flip_button = workspace.mode_buttons["flip x"]
    rendered_button = flip_button.grab().toImage()
    expected_border = "#55d4bd"
    assert rendered_button.pixelColor(rendered_button.width() // 2, 0).name() == expected_border
    assert (
        rendered_button.pixelColor(rendered_button.width() // 2, rendered_button.height() - 1).name()
        == expected_border
    )
    assert rendered_button.pixelColor(0, rendered_button.height() // 2).name() == expected_border
    assert (
        rendered_button.pixelColor(rendered_button.width() - 1, rendered_button.height() // 2).name()
        == expected_border
    )
    flipped_x_center = workspace.image.marker_items["target"][0].rect().center()
    assert workspace.mode_buttons["flip x"].property("activeToggle") is True
    assert workspace.image.flipped_x is True
    assert flipped_x_center.x() == 53.0
    assert flipped_x_center.y() == 43.0

    workspace.mode_buttons["flip y"].click()
    qapp.processEvents()
    flipped_xy_center = workspace.image.marker_items["target"][0].rect().center()
    assert workspace.mode_buttons["flip y"].property("activeToggle") is True
    assert workspace.image.flipped_y is True
    assert flipped_xy_center.x() == 53.0
    assert flipped_xy_center.y() == 20.0
    assert workspace.image._fits_from_scene(*workspace.image._scene_from_fits(10.0, 20.0)) == (
        10.0,
        20.0,
    )

    workspace.zoom.setCurrentText("200%")
    assert workspace.image.transform().m11() == 2.0
    workspace.reset_view()
    assert workspace.zoom.currentText() == "Fit"
    assert not workspace.mode_buttons["flip x"].isChecked()
    assert not workspace.mode_buttons["flip y"].isChecked()
    assert workspace.image.flipped_x is False
    assert workspace.image.flipped_y is False
    assert workspace.filename.text() == "FITS: TrES-3_reference.fits"
    assert workspace.dimensions.text() == "64 × 64 px"
    workspace.close()


def test_middle_mouse_drag_pans_fits_workspace_without_leaving_selection_mode(qapp, tmp_path) -> None:
    frame = tmp_path / "TrES-3_reference.fits"
    fits.writeto(frame, np.arange(256 * 256, dtype=np.float32).reshape(256, 256), overwrite=True)
    workspace = FITSWorkspace(tmp_path / "missing-demo.png")
    workspace.resize(800, 650)
    workspace.show()
    workspace.load_fits(frame, 1.2)
    qapp.processEvents()
    workspace.zoom.setCurrentText("400%")
    workspace.begin_selection("comparison")
    viewport = workspace.image.viewport()
    start = QPoint(viewport.width() // 2, viewport.height() // 2)
    before = (
        workspace.image.horizontalScrollBar().value(),
        workspace.image.verticalScrollBar().value(),
    )

    QTest.mousePress(viewport, Qt.MouseButton.MiddleButton, pos=start)
    QTest.mouseMove(viewport, QPoint(start.x() - 30, start.y() - 24))
    QTest.mouseRelease(viewport, Qt.MouseButton.MiddleButton, pos=QPoint(start.x() - 30, start.y() - 24))
    qapp.processEvents()

    after = (
        workspace.image.horizontalScrollBar().value(),
        workspace.image.verticalScrollBar().value(),
    )
    assert after != before
    assert workspace.image.mode == "select"
    assert workspace.image.selection_role == "comparison"
    workspace.close()


def test_run_and_cancel_buttons_show_active_processing_state(qapp, tmp_path) -> None:
    processing = ProcessingPage(StageID.REDUCTION, "Reduction", "Calibrate frames", [])
    assert processing.run.text() == "Run Reduction"
    assert processing.cancel.text() == "Cancel"
    assert not processing.cancel.isEnabled()

    processing.set_busy(True)
    qapp.processEvents()
    assert processing.run.text() == "Running Reduction…"
    assert processing.run.property("running") is True
    assert processing.run.icon().isNull()
    assert not processing.run.isEnabled()
    assert processing.cancel.property("cancelActive") is True
    assert processing.cancel.isEnabled()

    processing.set_busy(False)
    assert processing.run.text() == "Run Reduction"
    assert processing.run.property("running") is False
    assert processing.run.isEnabled()
    assert processing.cancel.property("cancelActive") is False
    assert not processing.cancel.isEnabled()

    photometry = PlateSolvePage(tmp_path / "missing-preview.png")
    photometry.inspector.set_busy(True)
    assert photometry.inspector.run.text() == "Running Photometry…"
    assert photometry.inspector.run.icon().isNull()
    assert photometry.inspector.cancel.isEnabled()
    assert photometry.inspector.cancel.property("cancelActive") is True
    photometry.inspector.set_busy(False)
    assert photometry.inspector.run.text() == "Run HOPS photometry"

    comparison = ComparisonStarsPage()
    comparison.set_busy(True)
    assert comparison.run.text() == "Running Photometry…"
    assert comparison.run.icon().isNull()
    assert comparison.cancel.isEnabled()
    comparison.set_busy(False)
    assert comparison.run.text() == "Run photometry"

    fitting = FittingPage()
    fitting.set_busy(True, full=True)
    assert fitting.full.text() == "Running Full Fit…"
    assert fitting.full.property("running") is True
    assert fitting.full.icon().isNull()
    assert fitting.cancel.isEnabled()
    fitting.set_busy(False)
    assert fitting.preview.text() == "Preview Fit"
    assert fitting.full.text() == "Run Full Fit"
    assert not fitting.cancel.isEnabled()


def test_fitting_catalog_and_observation_have_separate_spacious_cards(qapp) -> None:
    fitting = FittingPage()
    fitting.resize(1000, 900)
    fitting.catalog_source.setText("ExoClock · matched to the project coordinates")
    fitting.observation_source.setText(
        "Cousins R (COUSINS_R) · 30 s exposures · detected from science FITS · "
        "observer location not set, so Preview Fit will detrend against time"
    )
    fitting.show()
    qapp.processEvents()

    catalog_card = fitting.catalog_source.parentWidget()
    observation_card = fitting.observation_source.parentWidget()
    assert catalog_card.objectName() == "fittingMetadataCard"
    assert observation_card.objectName() == "fittingMetadataCard"
    assert catalog_card is not observation_card
    assert catalog_card.geometry().bottom() < observation_card.geometry().top()
    assert fitting.catalog_source.height() >= 36
    assert fitting.observation_source.height() >= 54
    fitting.close()


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
    assert not window.data_page.project_actions.isHidden()
    assert window.data_page.reveal_project.isEnabled()
    assert window.data_page.reset_project.isEnabled()
    window.close()


def test_open_existing_project_accepts_run_or_leaps_folder_and_opens_eclipse(qapp, tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    root = tmp_path / "WASP-18_TESS"
    project = ProjectWorkspace.create(root, "WASP-18 b — TESS")
    project.manifest.target_name = "WASP-18"
    project.manifest.stages[StageID.FITTING.value] = StageState(
        status=StageStatus.COMPLETE, summary="Imported primary-transit fit"
    )
    project.save()
    fitting = project.outputs_dir / StageID.FITTING.value
    fitting.mkdir()
    (fitting / "fit-summary.json").write_text(
        json.dumps(
            {
                "passband": "TESS",
                "light_curve": "aperture",
                "parameters": {
                    "name": "WASP-18 b",
                    "ra": "01:37:25.07",
                    "dec": "-45:40:40.10",
                    "period": 0.941452,
                    "mid_time": 2458354.458,
                    "rp_over_rs": 0.1018,
                    "sma_over_rs": 3.48,
                    "inclination": 83.5,
                    "eccentricity": 0.0,
                    "periastron": 0.0,
                    "metallicity": 0.0,
                    "temperature": 6432.0,
                    "logg": 4.31,
                    "source": "Test",
                    "source_date": "",
                },
                "fitted_ephemeris": {"period": 0.941452, "mid_time": 2458354.458},
            }
        ),
        encoding="utf-8",
    )

    window = MainWindow(settings=settings)
    window.open_existing_project(root / "LEAPS")
    qapp.processEvents()

    assert window.project is not None
    assert window.project.root == root
    assert window.stack.currentWidget() is window.secondary_eclipse_page
    assert window.data_page.open_existing_project.text() == "Open project"
    window.close()


def test_recent_project_startup_resumes_the_first_unfinished_stage(qapp, tmp_path) -> None:
    root = tmp_path / "recent"
    project = _project_ready_for_inspection(root)
    settings = QSettings(str(tmp_path / "recent.ini"), QSettings.Format.IniFormat)
    settings.setValue("projects/recent", str(root))

    window = MainWindow(settings=settings)

    assert window.project is not None
    assert window.project.manifest.project_id == project.manifest.project_id
    assert window.stack.currentWidget() is window.inspection_page
    assert window.stage_buttons[StageID.INSPECTION].active
    window.close()


def test_resume_prioritizes_a_stage_that_needs_attention(qapp, tmp_path) -> None:
    project = _project_ready_for_inspection(tmp_path)
    project.manifest.stages[StageID.REDUCTION.value] = StageState(
        status=StageStatus.NEEDS_ATTENTION,
        summary="Needs attention",
    )
    project.save()
    window = MainWindow(demo=True)

    window.open_existing_project(tmp_path)

    assert window.stack.currentWidget() is window.reduction_page
    window.close()


def test_pixel_scale_uses_editable_psf_estimate_and_restores_it_when_cleared(
    qapp, tmp_path
) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.target_name = "TrES-3"
    project.manifest.target_ra = "17:52:06.99"
    project.manifest.target_dec = "+37:32:46.15"
    project.save()
    reduction = project.outputs_dir / "reduction"
    reduction.mkdir()
    frame = reduction / "r_00001_TrES-3.fits"
    header = fits.Header({"HOPSPSF": 4.0})
    fits.writeto(
        frame,
        np.arange(64 * 64, dtype=np.float32).reshape(64, 64),
        header,
    )

    window = MainWindow(demo=True)
    window.set_project(project)
    qapp.processEvents()

    field = window.plate_page.inspector.pixel_scale
    assert field.text() == ""
    assert field.placeholderText() == "0.500 (estimated)"
    assert window.plate_page.inspector.effective_pixel_scale == 0.5
    assert window.plate_page.workspace.scale.text() == "Pixel scale: estimated"

    field.setText("1.25")
    qapp.processEvents()
    assert project.manifest.settings["pixel_scale"] == 1.25
    assert window.plate_page.workspace.scale.text() == 'Pixel scale: 1.25 "/pixel'
    assert ProjectWorkspace.open(tmp_path).manifest.settings["pixel_scale"] == 1.25

    field.clear()
    qapp.processEvents()
    assert "pixel_scale" not in project.manifest.settings
    assert field.placeholderText() == "0.500 (estimated)"
    assert window.plate_page.inspector.effective_pixel_scale == 0.5
    assert window.plate_page.workspace.scale.text() == "Pixel scale: estimated"
    assert "pixel_scale" not in ProjectWorkspace.open(tmp_path).manifest.settings
    window.close()


def test_project_reset_dialog_shows_scope_and_requires_exact_project_name(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    project.manifest.raw_files["science"] = ["light_001.fits", "light_002.fits"]
    project.save()
    dialog = ProjectResetDialog(project)
    dialog.show()
    qapp.processEvents()

    labels = "\n".join(label.text() for label in dialog.findChildren(QLabel))
    assert str(project.workspace) in labels
    assert "Raw files preserved: 2" in labels
    assert not dialog.reset_button.isEnabled()
    dialog.confirmation.setText("wrong project")
    assert not dialog.reset_button.isEnabled()
    dialog.confirmation.setText("TrES-3")
    assert dialog.reset_button.isEnabled()
    dialog.close()


def test_project_reset_state_clears_only_recent_project_and_current_ui(qapp, tmp_path) -> None:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("setup/complete", True)
    settings.setValue("window/geometry", b"kept")
    project = ProjectWorkspace.create(tmp_path / "run", "TrES-3")
    window = MainWindow(settings=settings)
    window.set_project(project)
    assert settings.value("projects/recent") == str(project.root)

    window._clear_current_project(project.root)

    assert not settings.contains("projects/recent")
    assert settings.value("setup/complete", type=bool) is True
    assert settings.value("window/geometry") == b"kept"
    assert window.project is None
    assert window.data_page.folder.text() == ""
    assert window.data_page.name.text() == ""
    assert window.data_page.project_actions.isHidden()
    assert window.stack.currentWidget() is window.data_page
    window.close()


def test_project_reset_is_disabled_while_processing(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "TrES-3")
    window = MainWindow(demo=True)
    window.set_project(project)

    window.runner.busyChanged.emit(True)
    assert not window.data_page.reset_project.isEnabled()
    assert window.data_page.reveal_project.isEnabled()
    window.runner.busyChanged.emit(False)
    assert window.data_page.reset_project.isEnabled()
    window.close()


def test_new_observing_run_hides_old_reset_and_rebinds_it_after_confirmation(
    qapp, tmp_path, monkeypatch
) -> None:
    old_project = ProjectWorkspace.create(tmp_path / "old-run", "WASP-41 b")
    new_project = ProjectWorkspace.create(tmp_path / "new-run", "TrES-3")
    for name in ("._cache", "._project.json", "._outputs"):
        (new_project.workspace / name).write_bytes(b"macOS metadata")
    window = MainWindow(demo=True)
    window.set_project(old_project)
    monkeypatch.setattr(window.runner, "start", lambda *_args, **_kwargs: None)

    window.data_page.folder.setText(str(new_project.root))
    window.scan_folder(new_project.root)

    assert window.project is old_project
    assert window.data_page.project_actions.isHidden()
    assert not window.data_page.reset_project.isEnabled()

    window.save_data_target(_saved_data_target_values(new_project.root, "COUSINS_R"))

    assert window.project is not None
    assert window.project.root == new_project.root
    assert not window.data_page.project_actions.isHidden()
    captured: list[ProjectWorkspace] = []

    class RejectingResetDialog:
        def __init__(self, project: ProjectWorkspace, _parent) -> None:
            captured.append(project)

        @staticmethod
        def exec():
            return main_window_module.QDialog.DialogCode.Rejected

    monkeypatch.setattr(main_window_module, "ProjectResetDialog", RejectingResetDialog)
    window.request_project_reset()

    assert [project.root for project in captured] == [new_project.root]
    window.close()


def test_process_start_stops_and_requests_access_when_project_preflight_fails(
    qapp, tmp_path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External run")
    window = MainWindow(demo=True)
    window.set_project(project)
    failure = LEAPSError(
        "PROJECT_STORAGE_ACCESS_DENIED",
        "LEAPS cannot access the project location",
        "The external drive is blocked.",
        ["Choose the folder again"],
        stage=StageID.REDUCTION,
    )
    requested: list[LEAPSError] = []

    def deny(_stage: StageID | None = None) -> None:
        raise failure

    monkeypatch.setattr(project, "verify_process_access", deny)
    monkeypatch.setattr(window, "_request_project_access", requested.append)

    assert not window._ensure_runner_idle("run Reduction", StageID.REDUCTION)
    assert requested == [failure]
    assert window._ensure_runner_idle("scan the observing run", StageID.DATA_TARGET)
    window.close()


def test_inspection_preflight_requests_access_when_reduced_frame_is_denied(
    qapp, tmp_path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External run")
    reduction = project.outputs_dir / StageID.REDUCTION.value
    reduction.mkdir()
    reduced = reduction / "r_00001.fits"
    fits.writeto(reduced, np.zeros((4, 4), dtype=np.float32))
    window = MainWindow(demo=True)
    window.set_project(project)
    requested: list[LEAPSError] = []
    original_open = Path.open

    def deny_reduced(path: Path, *args, **kwargs):
        if path == reduced and args and args[0] == "rb":
            raise PermissionError(1, "Operation not permitted")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_reduced)
    monkeypatch.setattr(window, "_request_project_access", requested.append)

    assert not window._ensure_runner_idle("run Inspection", StageID.INSPECTION)
    assert len(requested) == 1
    assert requested[0].code == "PROJECT_STORAGE_ACCESS_DENIED"
    assert requested[0].stage is StageID.INSPECTION
    window.close()


def test_target_selection_failure_never_tells_user_to_choose_another_star(qapp) -> None:
    window = MainWindow(demo=True)
    failures: list[LEAPSError] = []
    window._handle_error = failures.append
    generic = LEAPSError(
        "PHOTOMETRY_STAR_NOT_FOUND",
        "No acceptable star was found at that position",
        "Click closer to the center of an unsaturated star.",
        ["Choose another star", "Adjust advanced detection settings"],
        stage=StageID.PHOTOMETRY,
    )

    window._photometry_star_selection_failed("target", generic)

    assert failures[0].code == "PHOTOMETRY_TARGET_NOT_CENTERED"
    assert "not replaced" in failures[0].message
    assert "another star" not in " ".join(failures[0].recovery).casefold()
    assert "same target" in " ".join(failures[0].recovery).casefold()
    window.close()


def test_project_access_dialog_reopens_native_picker_and_rechecks_project(
    qapp, tmp_path, monkeypatch
) -> None:
    project = ProjectWorkspace.create(tmp_path, "External run")
    window = MainWindow(demo=True)
    window.set_project(project)
    captured: dict[str, str] = {}

    class FakeMessageBox:
        class Icon:
            Warning = object()

        class ButtonRole:
            RejectRole = object()
            ActionRole = object()
            AcceptRole = object()

        def __init__(self, _parent) -> None:
            self.grant = None

        def setIcon(self, _icon) -> None:
            pass

        def setWindowTitle(self, title: str) -> None:
            captured["title"] = title

        def setText(self, message: str) -> None:
            captured["message"] = message

        def setInformativeText(self, information: str) -> None:
            captured["information"] = information

        def setDetailedText(self, _details: str) -> None:
            pass

        def addButton(self, label: str, _role):
            button = object()
            if "Grant Access" in label:
                self.grant = button
            return button

        def setDefaultButton(self, _button) -> None:
            pass

        def setEscapeButton(self, _button) -> None:
            pass

        def exec(self) -> None:
            pass

        def clickedButton(self):
            return self.grant

    checks: list[StageID | None] = []
    monkeypatch.setattr(main_window_module, "QMessageBox", FakeMessageBox)
    monkeypatch.setattr(main_window_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getExistingDirectory",
        lambda *_args: str(project.root),
    )
    monkeypatch.setattr(project, "verify_process_access", checks.append)
    failure = LEAPSError(
        "PROJECT_STORAGE_ACCESS_DENIED",
        "LEAPS cannot access the project location",
        "The external drive is blocked.",
        ["Choose the folder again"],
        stage=StageID.REDUCTION,
        technical_details="PermissionError: Operation not permitted",
    )

    window._request_project_access(failure)

    assert captured["title"] == "Project access required"
    assert "native macOS picker" in captured["information"]
    assert checks == [StageID.REDUCTION]
    assert "access restored" in window.status_text.text().casefold()
    window.close()


def test_cancelled_processing_is_ready_to_resume_without_error_dialog(qapp, tmp_path) -> None:
    project = ProjectWorkspace.create(tmp_path, "NGTS-10")
    window = MainWindow(demo=True)
    window.set_project(project)
    cancelled = LEAPSError(
        "JOB_CANCELLED",
        "Processing was safely cancelled",
        "Verified checkpoints were kept.",
        ["Resume"],
        stage=StageID.REDUCTION,
    )

    window._stage_failed(StageID.REDUCTION, cancelled)

    state = project.manifest.stages[StageID.REDUCTION.value]
    assert state.status == StageStatus.READY
    assert state.checkpoint == "cancelled"
    assert window.last_failure is None
    assert "cancelled" in window.pages[StageID.REDUCTION].status.text().casefold()
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


def test_photometry_progress_filename_does_not_resize_inspector(qapp) -> None:
    window = MainWindow(demo=True)
    window.show()
    qapp.processEvents()
    inspector = window.plate_page.inspector

    window._stage_event(
        StageEvent(
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            "Measured r_00008_short.fits",
            8,
            354,
        )
    )
    qapp.processEvents()
    first_width = inspector.width()
    window._stage_event(
        StageEvent(
            StageID.PHOTOMETRY,
            JobStatus.RUNNING,
            "Measured r_00128_TrES-3_Cousins_R_extremely_long_filename.fits",
            128,
            354,
        )
    )
    qapp.processEvents()

    assert inspector.banner_title.text() == "Measuring frame 128 of 354"
    assert inspector.banner_title.toolTip().endswith("extremely_long_filename.fits")
    assert inspector.width() == first_width
    window.close()


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
