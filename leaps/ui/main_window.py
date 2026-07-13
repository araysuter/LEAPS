from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from leaps.catalog import PlanetCatalogResolver
from leaps.diagnostics import DiagnosticLogger
from leaps.exports import TransitExporter
from leaps.fits_inventory import FITSInventory, FrameRecord, validate_coordinates
from leaps.models import LEAPSError, ProjectManifest, StageEvent, StageID, StageStatus
from leaps.offline import OfflineDataManager
from leaps.project import ProjectWorkspace
from leaps.science import (
    AlignmentService,
    FittingService,
    InspectionService,
    PhotometryService,
    PlateSolveService,
    ReductionConfig,
    ReductionService,
)

from .pages import (
    ComparisonStarsPage,
    DataTargetPage,
    FittingPage,
    ObservingPlannerPage,
    PlateSolvePage,
    ProcessingPage,
    ReportsPage,
    SimpleToolPage,
)
from .settings_dialog import SettingsDialog, default_offline_root
from .theme import COLORS
from .widgets import StageNavButton, ToolNavButton, icon
from .workers import TaskRunner

STAGE_LABELS = {
    StageID.DATA_TARGET: "Data & Target",
    StageID.REDUCTION: "Reduction",
    StageID.INSPECTION: "Inspection",
    StageID.ALIGNMENT: "Alignment",
    StageID.PHOTOMETRY: "Photometry",
    StageID.FITTING: "Fitting",
}


class MainWindow(QMainWindow):
    projectChanged = Signal(object)
    offlineProgress = Signal(str, object, object)

    def __init__(self, *, demo: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("LEAPS — Exoplanet Transit Analysis")
        self.setMinimumSize(1120, 720)
        self.resize(1440, 960)
        self.settings = QSettings()
        self.project: ProjectWorkspace | None = None
        self.logger: DiagnosticLogger | None = None
        self.records: list[FrameRecord] = []
        self.last_failure: LEAPSError | None = None
        self.runner = TaskRunner(self)
        self.offline_manager = OfflineDataManager(default_offline_root())
        self._build_ui()
        self._connect()
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(30_000)
        self.autosave_timer.timeout.connect(self.autosave_project)
        self.autosave_timer.start()
        geometry = self.settings.value("window/geometry")
        if geometry and not demo:
            self.restoreGeometry(geometry)
        if demo:
            self.resize(1487, 1018)
            self._load_demo_state()
        else:
            recent = self.settings.value("projects/recent", "")
            if recent and (Path(recent) / ProjectWorkspace.WORKSPACE_NAME / "project.json").exists():
                try:
                    self.set_project(ProjectWorkspace.open(recent))
                except Exception:
                    pass

    def _build_ui(self) -> None:
        shell = QWidget()
        shell.setObjectName("appShell")
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        main = QHBoxLayout()
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(0)
        self.sidebar = self._build_sidebar()
        main.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        main.addWidget(self.stack, 1)
        shell_layout.addLayout(main, 1)
        shell_layout.addWidget(self._build_status_bar())
        self.setCentralWidget(shell)

        asset = Path(__file__).resolve().parents[1] / "assets" / "demo-starfield.png"
        self.pages: dict[str | StageID, QWidget] = {}
        self.data_page = DataTargetPage()
        self.reduction_page = ProcessingPage(
            StageID.REDUCTION,
            "Reduction",
            "Apply bias, dark, and flat calibration while keeping every raw FITS frame read-only.",
            [
                (
                    "Median-combine calibration frames",
                    "Reject isolated cosmic rays when building master bias, dark, and flat frames.",
                ),
                (
                    "Preserve last successful reduction",
                    "New outputs replace the previous reduction only after the complete run succeeds.",
                ),
            ],
        )
        self.inspection_page = ProcessingPage(
            StageID.INSPECTION,
            "Inspection",
            "Review sky background and point-spread changes before alignment.",
            [
                (
                    "Suggest outlier exclusions",
                    "Flag frames with unusual sky background or PSF without removing them automatically.",
                ),
                ("Keep manual exclusions", "Retain user decisions when this stage is resumed or rerun."),
            ],
        )
        self.alignment_page = ProcessingPage(
            StageID.ALIGNMENT,
            "Alignment",
            "Register the reduced sequence against a stable reference frame.",
            [
                (
                    "Use the first accepted frame as reference",
                    "Match subsequent star fields to the first non-excluded reduced frame.",
                ),
                (
                    "Checkpoint each frame",
                    "Allow a safely cancelled run to resume from the latest verified result.",
                ),
            ],
        )
        self.plate_page = PlateSolvePage(asset)
        self.fitting_page = FittingPage()
        for stage, page in (
            (StageID.DATA_TARGET, self.data_page),
            (StageID.REDUCTION, self.reduction_page),
            (StageID.INSPECTION, self.inspection_page),
            (StageID.ALIGNMENT, self.alignment_page),
            (StageID.PHOTOMETRY, self.plate_page),
            (StageID.FITTING, self.fitting_page),
        ):
            self.pages[stage] = page
            self.stack.addWidget(page)
        self.comparison_page = ComparisonStarsPage()
        self.pages["apertures"] = self.comparison_page
        self.stack.addWidget(self.comparison_page)
        for key, title, subtitle, icon_name in (
            (
                "light_curve",
                "Light Curve",
                "Inspect normalized flux, uncertainty, and excluded frames across the observing run.",
                "fa6s.chart-line",
            ),
            (
                "diagnostics",
                "Diagnostics",
                "Export a redacted support ZIP with logs, versions, settings, stage summaries, and sanitized FITS headers.",
                "fa6s.stethoscope",
            ),
        ):
            page = SimpleToolPage(title, subtitle, icon_name)
            self.pages[key] = page
            self.stack.addWidget(page)
        self.reports_page = ReportsPage()
        self.pages["reports"] = self.reports_page
        self.stack.addWidget(self.reports_page)
        planner = ObservingPlannerPage()
        self.pages["planner"] = planner
        self.stack.addWidget(planner)
        self.stack.setCurrentWidget(self.data_page)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(265)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 18, 0, 8)
        layout.setSpacing(0)
        brand = QWidget()
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(20, 2, 16, 16)
        mark = QLabel()
        mark.setPixmap(
            QPixmap(str(Path(__file__).resolve().parents[1] / "assets" / "leaps-mark.png")).scaled(
                48, 48, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )
        brand_layout.addWidget(mark)
        word = QVBoxLayout()
        word.setSpacing(0)
        name = QLabel("LEAPS")
        name.setStyleSheet("font-size: 25px; font-weight: 750; letter-spacing: 1px;")
        descriptor = QLabel("Exoplanet Transit Analysis")
        descriptor.setStyleSheet(f"color: {COLORS['cyan']}; font-size: 10px;")
        word.addWidget(name)
        word.addWidget(descriptor)
        brand_layout.addLayout(word, 1)
        layout.addWidget(brand)
        workflow = QLabel("WORKFLOW")
        workflow.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 11px; font-weight: 650; padding: 10px 19px 6px;"
        )
        layout.addWidget(workflow)
        self.stage_buttons: dict[StageID, StageNavButton] = {}
        for stage in StageID:
            button = StageNavButton(stage, STAGE_LABELS[stage])
            self.stage_buttons[stage] = button
            layout.addWidget(button)
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"color: {COLORS['border_soft']}; margin: 8px 20px;")
        layout.addWidget(divider)
        tools_label = QLabel("TOOLS")
        tools_label.setStyleSheet(
            f"color: {COLORS['muted']}; font-size: 11px; font-weight: 650; padding: 4px 19px 6px;"
        )
        layout.addWidget(tools_label)
        self.tool_buttons: dict[str, ToolNavButton] = {}
        for key, label, icon_name in (
            ("apertures", "Apertures", "fa6s.bullseye"),
            ("light_curve", "Light Curve", "fa6s.chart-line"),
            ("diagnostics", "Diagnostics", "fa6s.stethoscope"),
            ("reports", "Reports", "fa6s.file-lines"),
            ("planner", "Observing Planner", "fa6s.moon"),
            ("settings", "Settings", "fa6s.gear"),
        ):
            button = ToolNavButton(label, icon_name)
            self.tool_buttons[key] = button
            layout.addWidget(button)
        layout.addStretch()
        collapse = QPushButton()
        collapse.setIcon(icon("fa6s.angles-left", COLORS["muted"]))
        collapse.setToolTip("Collapse the workflow sidebar.")
        collapse.setFixedSize(38, 34)
        collapse.setStyleSheet("border: 0; background: transparent;")
        layout.addWidget(collapse, 0, Qt.AlignmentFlag.AlignRight)
        return sidebar

    def _build_status_bar(self) -> QFrame:
        frame = QFrame()
        frame.setFixedHeight(68)
        frame.setStyleSheet(f"background: #081725; border-top: 1px solid {COLORS['border_soft']};")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(20, 0, 20, 0)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text = QLabel("Ready")
        self.status_text.setStyleSheet("font-weight: 600;")
        self.autosave = QLabel("No project open")
        self.autosave.setObjectName("muted")
        layout.addWidget(self.status_dot)
        layout.addWidget(self.status_text)
        layout.addWidget(QLabel("·"))
        layout.addWidget(self.autosave)
        layout.addStretch()
        self.project_label = QLabel("")
        self.project_label.setObjectName("muted")
        layout.addWidget(self.project_label)
        return frame

    def _connect(self) -> None:
        for stage, button in self.stage_buttons.items():
            button.clicked.connect(self.open_stage)
        for key, button in self.tool_buttons.items():
            if key == "settings":
                button.clicked.connect(self.open_settings)
            elif key == "diagnostics":
                button.clicked.connect(self.export_diagnostics)
            else:
                button.clicked.connect(lambda checked=False, page_key=key: self.open_tool(page_key))
        self.data_page.scanRequested.connect(self.scan_folder)
        self.data_page.saveRequested.connect(self.save_data_target)
        for page in (self.reduction_page, self.inspection_page, self.alignment_page):
            page.runRequested.connect(self.run_stage)
            page.cancelRequested.connect(self.runner.cancel)
        self.plate_page.retryRequested.connect(self.retry_plate_solve)
        self.plate_page.copyDiagnosticsRequested.connect(self.copy_diagnostics)
        self.plate_page.manualTargetPlaced.connect(self.manual_target_placed)
        self.comparison_page.rankRequested.connect(self.rank_comparison_stars)
        self.comparison_page.runRequested.connect(self.run_photometry)
        self.fitting_page.previewRequested.connect(lambda values: self.run_fitting(values, full=False))
        self.fitting_page.fullFitRequested.connect(lambda values: self.run_fitting(values, full=True))
        self.reports_page.openFolderRequested.connect(self.open_outputs_folder)
        self.reports_page.exportExoClockRequested.connect(lambda: self.export_transit("exoclock"))
        self.reports_page.exportETDRequested.connect(lambda: self.export_transit("etd"))
        self.offlineProgress.connect(self._offline_progress)

    def _load_demo_state(self) -> None:
        manifest = ProjectManifest(
            name="WTS-2 b · 2026-06-28",
            target_name="WTS-2 b",
            target_ra="19:34:55.87",
            target_dec="+36:48:55.79",
        )
        summaries = {
            StageID.DATA_TARGET: "Target selected",
            StageID.REDUCTION: "Calibrated",
            StageID.INSPECTION: "Looks good",
            StageID.ALIGNMENT: "Solution found",
        }
        for stage, summary in summaries.items():
            manifest.stages[stage.value].status = StageStatus.COMPLETE
            manifest.stages[stage.value].summary = summary
        manifest.stages[StageID.PHOTOMETRY.value].status = StageStatus.READY
        manifest.stages[StageID.PHOTOMETRY.value].summary = "Plate solve"
        manifest.stages[StageID.FITTING.value].status = StageStatus.LOCKED
        manifest.stages[StageID.FITTING.value].summary = "Locked"
        self._apply_manifest(manifest)
        self.open_stage(StageID.PHOTOMETRY)
        self.status_text.setText("Session saved")
        self.autosave.setText("autosave 1m ago")
        self.project_label.setText("WTS-2 b · 200 science frames")

    def set_project(self, project: ProjectWorkspace) -> None:
        self.project = project
        self.logger = DiagnosticLogger(project)
        self.settings.setValue("projects/recent", str(project.root))
        self.data_page.folder.setText(str(project.root))
        self.data_page.name.setText(project.manifest.target_name)
        self.data_page.ra.setText(project.manifest.target_ra)
        self.data_page.dec.setText(project.manifest.target_dec)
        self.data_page.set_assignment_patterns(project.manifest.settings.get("frame_classifiers", {}))
        self.plate_page.inspector.target_name.setText(project.manifest.target_name or "Unnamed target")
        self.plate_page.inspector.coordinates.setText(
            f"{project.manifest.target_ra}  {project.manifest.target_dec}"
        )
        self.plate_page.inspector.pixel_scale.setText(
            f"{float(project.manifest.settings.get('pixel_scale', 1.2)):.2f} arcsec/pixel"
        )
        self._apply_manifest(project.manifest)
        self.project_label.setText(project.manifest.name)
        self.status_text.setText("Session saved")
        self.autosave.setText("autosaved just now")
        try:
            import astropy.units as units
            from astropy.coordinates import SkyCoord

            coordinate = SkyCoord(
                project.manifest.target_ra,
                project.manifest.target_dec,
                unit=(units.hourangle, units.deg),
            )
            self.offline_manager.add_gaia_region(coordinate.ra.deg, coordinate.dec.deg, 0.5)
        except Exception:
            pass
        self.projectChanged.emit(project)

    def _apply_manifest(self, manifest: ProjectManifest) -> None:
        for stage, button in self.stage_buttons.items():
            button.update_state(manifest.stages[stage.value])

    def open_stage(self, stage: StageID) -> None:
        self.stack.setCurrentWidget(self.pages[stage])
        for key, button in self.stage_buttons.items():
            button.set_active(key == stage)

    def open_tool(self, key: str) -> None:
        self.stack.setCurrentWidget(self.pages[key])
        for button in self.stage_buttons.values():
            button.set_active(False)

    def scan_folder(self, root: Path) -> None:
        def scan(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            return FITSInventory(root).discover()

        self.status_text.setText("Scanning FITS headers…")
        self.runner.start(
            scan, result=self._scan_complete, error=self._handle_error, finished=self._scan_finished
        )

    def _scan_complete(self, records: list[FrameRecord]) -> None:
        self.records = records
        self.data_page.set_records(records)

    def _scan_finished(self) -> None:
        self.data_page.scan_progress.setVisible(False)
        self.status_text.setText("Ready")

    def save_data_target(self, values: dict[str, Any]) -> None:
        try:
            if not values["root"]:
                raise LEAPSError(
                    "PROJECT_FOLDER_REQUIRED",
                    "Choose the observing run",
                    "Select the folder that contains the FITS images.",
                    ["Choose folder"],
                    stage=StageID.DATA_TARGET,
                )
            ra, dec = validate_coordinates(values["ra"], values["dec"])
            grouped = values["assignments"]
            if not grouped["science"]:
                raise LEAPSError(
                    "SCIENCE_FRAMES_REQUIRED",
                    "No science frames were assigned",
                    "Confirm at least one light/science exposure before continuing.",
                    ["Review frame assignments"],
                    stage=StageID.DATA_TARGET,
                )
            missing = [
                kind for kind in ("bias", "dark", "flat") if not grouped[kind] and not values["waivers"][kind]
            ]
            if missing:
                names = ", ".join(missing)
                raise LEAPSError(
                    "CALIBRATION_CONFIRMATION_REQUIRED",
                    "Calibration decision required",
                    f"No {names} frames were found. Add them or explicitly accept the corresponding waiver.",
                    ["Add calibration frames", "Confirm the waiver"],
                    stage=StageID.DATA_TARGET,
                )
            root = Path(values["root"])
            project = (
                ProjectWorkspace.open(root)
                if (root / ProjectWorkspace.WORKSPACE_NAME / "project.json").exists()
                else ProjectWorkspace.create(root, values["target_name"] or root.name)
            )
            project.manifest.target_name = values["target_name"]
            project.manifest.target_ra = ra
            project.manifest.target_dec = dec
            project.manifest.raw_files = grouped
            project.manifest.settings["calibration_waivers"] = values["waivers"]
            project.manifest.settings["frame_classifiers"] = values["frame_classifiers"]
            project.set_stage(StageID.DATA_TARGET, StageStatus.COMPLETE, "Target selected", progress=1.0)
            self.set_project(project)
            self.open_stage(StageID.REDUCTION)
        except BaseException as exc:
            failure = self._as_failure(exc, StageID.DATA_TARGET)
            self.data_page.show_error(f"{failure.title}: {failure.message}")

    def run_stage(self, stage: StageID) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "Choose and confirm an observing run before processing.",
                    ["Open Data & Target"],
                    stage=stage,
                )
            )
            return
        page = self.pages[stage]
        assert isinstance(page, ProcessingPage)
        functions = {
            StageID.REDUCTION: (ReductionService().run, {"config": ReductionConfig()}),
            StageID.INSPECTION: (InspectionService().run, {}),
            StageID.ALIGNMENT: (AlignmentService().run, {}),
        }
        function, kwargs = functions[stage]
        page.set_busy(True)
        self.project.set_stage(stage, StageStatus.RUNNING, "Processing", progress=0.0)
        self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['cyan']};")
        self.status_text.setText(f"Running {STAGE_LABELS[stage]}…")
        self.runner.start(
            function,
            self.project,
            event=self._stage_event,
            result=lambda result, current_stage=stage: self._stage_complete(current_stage, result),
            error=lambda exc, current_stage=stage: self._stage_failed(current_stage, exc),
            finished=lambda current_page=page: current_page.set_busy(False),
            **kwargs,
        )

    def _stage_event(self, event: StageEvent) -> None:
        page = self.pages[event.stage]
        if isinstance(page, ProcessingPage):
            page.update_event(event)
        if self.project:
            state = self.project.manifest.stages[event.stage.value]
            state.progress = event.fraction
            state.checkpoint = event.checkpoint
            state.summary = event.message

    def _stage_complete(self, stage: StageID, result: Any) -> None:
        if self.project:
            self.project.set_stage(
                stage,
                StageStatus.COMPLETE,
                "Complete",
                progress=1.0,
                output_path=getattr(result, "__fspath__", lambda: None)(),
            )
            self._apply_manifest(self.project.manifest)
        self.status_dot.setStyleSheet(f"color: {COLORS['green']};")
        self.status_text.setText(f"{STAGE_LABELS[stage]} complete")
        self.autosave.setText("autosaved just now")

    def _stage_failed(self, stage: StageID, exc: BaseException) -> None:
        failure = self._as_failure(exc, stage)
        if self.project:
            self.project.set_stage(stage, StageStatus.NEEDS_ATTENTION, "Needs attention")
            self._apply_manifest(self.project.manifest)
        page = self.pages[stage]
        if isinstance(page, ProcessingPage):
            page.set_failure(failure)
        elif stage == StageID.PHOTOMETRY:
            self.plate_page.inspector.set_failure(failure)
        self._show_failure(failure)

    def retry_plate_solve(self) -> None:
        if not self.project:
            return
        frames = sorted((self.project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        if not frames:
            self._handle_error(
                LEAPSError(
                    "PLATE_FRAME_REQUIRED",
                    "No reduced image is available",
                    "Run Reduction before plate solving.",
                    ["Open Reduction"],
                    stage=StageID.PHOTOMETRY,
                )
            )
            return
        self.runner.start(
            PlateSolveService().solve,
            frames[0],
            self.project.manifest.target_ra,
            self.project.manifest.target_dec,
            float(self.project.manifest.settings.get("pixel_scale", 1.2)),
            event=self._stage_event,
            result=self._plate_complete,
            error=lambda exc: self._stage_failed(StageID.PHOTOMETRY, exc),
        )

    def _plate_complete(self, result: Any) -> None:
        if self.project:
            self.project.manifest.settings["plate_solution"] = {
                "target_xy": result.target_xy,
                "identified_stars": result.identified_stars,
                "unverified": result.unverified,
            }
            self.project.set_stage(StageID.PHOTOMETRY, StageStatus.READY, "Plate solved", progress=0.2)
            self._apply_manifest(self.project.manifest)
        self.open_tool("apertures")

    def manual_target_placed(self, x: float, y: float) -> None:
        if self.project:
            self.project.manifest.settings["plate_solution"] = {
                "target_normalized": [x, y],
                "unverified": True,
            }
            self.project.manifest.warnings.append(
                {"code": "UNVERIFIED_WCS", "message": "Target was placed manually after plate solve failure."}
            )
            self.project.set_stage(
                StageID.PHOTOMETRY, StageStatus.READY, "Manual target · unverified WCS", progress=0.2
            )
            self._apply_manifest(self.project.manifest)

    def _photometry_inputs(self) -> tuple[Path, tuple[float, float]]:
        if not self.project:
            raise LEAPSError(
                "PROJECT_REQUIRED",
                "Open a project first",
                "Choose an observing run before photometry.",
                ["Open Data & Target"],
            )
        frames = sorted((self.project.outputs_dir / StageID.REDUCTION.value).glob("*.fit*"))
        if not frames:
            raise LEAPSError(
                "PHOTOMETRY_FRAME_REQUIRED",
                "No reduced frame is available",
                "Run Reduction before ranking comparison stars.",
                ["Open Reduction"],
                stage=StageID.PHOTOMETRY,
            )
        solution = self.project.manifest.settings.get("plate_solution", {})
        target = solution.get("target_xy")
        if target is None and solution.get("target_normalized"):
            from astropy.io import fits

            shape = fits.getdata(frames[0], memmap=True).shape
            nx, ny = shape[-1], shape[-2]
            normalized = solution["target_normalized"]
            target = (float(normalized[0]) * nx, float(normalized[1]) * ny)
        if target is None:
            raise LEAPSError(
                "TARGET_POSITION_REQUIRED",
                "The target position is not confirmed",
                "Complete plate solving or place the target manually first.",
                ["Open Photometry"],
                stage=StageID.PHOTOMETRY,
            )
        return frames[0], (float(target[0]), float(target[1]))

    def rank_comparison_stars(self) -> None:
        try:
            frame, target = self._photometry_inputs()
        except BaseException as exc:
            self._handle_error(exc)
            return

        def rank(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            return PhotometryService().rank_comparisons(frame, target)

        self.status_text.setText("Ranking comparison stars…")
        self.runner.start(
            rank,
            result=self.comparison_page.set_candidates,
            error=self._handle_error,
            finished=lambda: self.status_text.setText("Comparison ranking ready"),
        )

    def run_photometry(self, comparisons: list[tuple[float, float]], radius: float) -> None:
        try:
            _, target = self._photometry_inputs()
            if not self.project:
                return
            self.project.set_stage(StageID.PHOTOMETRY, StageStatus.RUNNING, "Measuring light curve")
            self.runner.start(
                PhotometryService().run,
                self.project,
                target,
                comparisons,
                radius,
                event=self._stage_event,
                result=lambda result: self._stage_complete(StageID.PHOTOMETRY, result),
                error=lambda exc: self._stage_failed(StageID.PHOTOMETRY, exc),
            )
            self.comparison_page.status.setText("Photometry is running in the background…")
        except BaseException as exc:
            self._handle_error(exc)

    def run_fitting(self, values: dict[str, Any], *, full: bool) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "A project light curve is required for fitting.",
                    ["Open Data & Target"],
                    stage=StageID.FITTING,
                )
            )
            return

        def fit(*, emit=None, token=None):
            if token:
                token.raise_if_cancelled()
            resolver = PlanetCatalogResolver(self.offline_manager.root / "nasa" / "planets.json")
            parameters = resolver.resolve(
                self.project.manifest.target_ra, self.project.manifest.target_dec, values["planet"]
            )
            parameters = replace(
                parameters,
                period=float(values["period"]),
                mid_time=float(values["mid_time"]),
                rp_over_rs=max(float(values["depth"]), 0.0) ** 0.5,
            )
            result = FittingService().run(
                self.project,
                parameters,
                full=full,
                exposure_time=float(self.project.manifest.settings.get("exposure_time", 30.0)),
                filter_name=str(self.project.manifest.settings.get("filter", "R")),
                latitude=float(self.project.manifest.global_profile.get("latitude", 0.0)),
                longitude=float(self.project.manifest.global_profile.get("longitude", 0.0)),
                iterations=int(values["iterations"]),
                burn_in=int(values["burn"]),
            )
            if token:
                token.raise_if_cancelled()
            return result

        self.status_text.setText("Running full fit…" if full else "Building fit preview…")
        self.runner.start(
            fit,
            result=lambda result: self._stage_complete(StageID.FITTING, result),
            error=lambda exc: self._stage_failed(StageID.FITTING, exc),
        )

    def _as_failure(self, exc: BaseException, stage: StageID | None = None) -> LEAPSError:
        if self.logger:
            return self.logger.failure(exc, stage)
        if isinstance(exc, LEAPSError):
            return exc
        return LEAPSError(
            "UNEXPECTED_FAILURE",
            "This step could not be completed",
            "LEAPS kept the last successful result. Retry or export diagnostics.",
            ["Retry", "Export diagnostics"],
            stage=stage,
            technical_details=str(exc),
        )

    def _handle_error(self, exc: BaseException) -> None:
        self._show_failure(self._as_failure(exc))

    def _show_failure(self, failure: LEAPSError) -> None:
        self.last_failure = failure
        self.status_dot.setStyleSheet(f"color: {COLORS['amber']};")
        self.status_text.setText(failure.title)
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle(failure.title)
        dialog.setText(failure.message)
        dialog.setInformativeText(
            "Suggested recovery: "
            + " · ".join(failure.recovery)
            + f"\n\nDiagnostic reference: {failure.diagnostic_id}"
        )
        if failure.technical_details:
            dialog.setDetailedText(failure.technical_details)
        dialog.exec()

    def copy_diagnostics(self) -> None:
        failure = self.last_failure
        payload = (
            failure.as_dict()
            if failure
            else {
                "failure": "PLATE_SOLVE_FAILED",
                "diagnostic_reference": "LEAPS-DEMO-503",
                "attempts": [
                    "coordinates validated",
                    "42 stars detected",
                    "Gaia HTTP 503",
                    "stopped after 3 bounded attempts",
                ],
            }
        )
        if self.project:
            payload["target"] = self.project.manifest.target_name
            payload["coordinates"] = f"{self.project.manifest.target_ra} {self.project.manifest.target_dec}"
            payload["pixel_scale"] = self.project.manifest.settings.get("pixel_scale", 1.2)
        QApplication.clipboard().setText(json.dumps(payload, indent=2))
        self.status_text.setText("Plate-solve diagnostics copied")

    def export_diagnostics(self) -> None:
        if not self.logger:
            self.open_tool("diagnostics")
            return
        default = str(self.project.root / f"LEAPS-diagnostics-{self.project.manifest.project_id[:8]}.zip")
        destination, _ = QFileDialog.getSaveFileName(
            self, "Export redacted diagnostics", default, "ZIP archive (*.zip)"
        )
        if destination:
            samples = [
                self.project.resolve(path) for path in self.project.manifest.raw_files.get("science", [])[:3]
            ]
            output = self.logger.export_bundle(destination, samples)
            self.status_text.setText(f"Diagnostics exported: {output.name}")

    def open_settings(self) -> None:
        dialog = SettingsDialog(self.offline_manager, self)
        dialog.offline.downloadAllRequested.connect(lambda: self._download_offline(dialog))
        dialog.offline.refreshRequested.connect(lambda: self._refresh_offline(dialog))
        dialog.offline.removeRequested.connect(lambda asset_id: self._remove_offline(dialog, asset_id))
        dialog.exec()

    def _download_offline(self, dialog: SettingsDialog) -> None:
        self._settings_dialog = dialog

        def download(*, emit=None, token=None):
            self.offline_manager.download_all(
                progress=self.offlineProgress.emit,
                cancelled=lambda: bool(token and token.cancelled),
            )
            return True

        dialog.offline.download.setEnabled(False)
        self.runner.start(
            download,
            result=lambda _: dialog.offline.finish_progress(),
            error=self._handle_error,
            finished=lambda: dialog.offline.download.setEnabled(True),
        )

    def _offline_progress(self, label: str, current: int, total: int) -> None:
        dialog = getattr(self, "_settings_dialog", None)
        if dialog is not None:
            dialog.offline.set_progress(label, current, total)

    def _refresh_offline(self, dialog: SettingsDialog) -> None:
        def refresh(*, emit=None, token=None):
            return self.offline_manager.load_remote_manifest()

        self.runner.start(refresh, result=lambda _: dialog.offline.refresh(), error=self._handle_error)

    def _remove_offline(self, dialog: SettingsDialog, asset_id: str) -> None:
        self.offline_manager.remove(asset_id)
        dialog.offline.refresh()

    def open_outputs_folder(self) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "Project outputs are created beside an observing run.",
                    ["Open Data & Target"],
                )
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.project.outputs_dir)))

    def export_transit(self, format_name: str) -> None:
        if not self.project:
            self._handle_error(
                LEAPSError(
                    "PROJECT_REQUIRED",
                    "Open a project first",
                    "A successful light curve is required for export.",
                    ["Open Data & Target"],
                )
            )
            return
        label = "ExoClock" if format_name == "exoclock" else "ETD"
        default = str(
            self.project.outputs_dir / f"{self.project.manifest.target_name or 'transit'}-{label}.txt"
        )
        destination, _ = QFileDialog.getSaveFileName(self, f"Export {label}", default, "Text table (*.txt)")
        if not destination:
            return
        try:
            exporter = TransitExporter(self.project)
            output = (
                exporter.export_exoclock(destination)
                if format_name == "exoclock"
                else exporter.export_etd(destination)
            )
            self.status_text.setText(f"{label} export created: {output.name}")
        except BaseException as exc:
            self._handle_error(exc)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.settings.setValue("window/geometry", self.saveGeometry())
        if self.project:
            self.project.save()
        event.accept()

    def autosave_project(self) -> None:
        if self.project:
            self.project.save()
            self.autosave.setText("autosaved just now")
