from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTime, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from leaps.fits_inventory import FITS_EXTENSIONS, FrameRecord
from leaps.models import LEAPSError, StageEvent, StageID
from leaps.targets import ResolvedTarget

from .theme import COLORS
from .widgets import ActionButton, FITSWorkspace, InfoButton, LabelWithInfo, PageHeader, icon


def _scroll_page(content: QWidget) -> QScrollArea:
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.Shape.NoFrame)
    scroll.setWidget(content)
    return scroll


class FrameAssignmentCard(QFrame):
    classifierChanged = Signal()

    def __init__(
        self,
        title: str,
        default_classifier: str,
        icon_name: str,
        tooltip: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("assignmentCard")
        self.setStyleSheet(
            f"QFrame#assignmentCard {{ background: {COLORS['canvas']}; border: 1px solid {COLORS['border']}; border-radius: 8px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 13, 14, 14)
        layout.setSpacing(9)

        heading = QHBoxLayout()
        category_icon = QLabel()
        category_icon.setPixmap(icon(icon_name, COLORS["cyan"]).pixmap(19, 19))
        heading.addWidget(category_icon)
        label = QLabel(title)
        label.setStyleSheet("font-size: 14px; font-weight: 650;")
        heading.addWidget(label)
        heading.addWidget(InfoButton(tooltip))
        heading.addStretch()
        self.count = QLabel("0 selected")
        self.count.setStyleSheet(
            f"color: {COLORS['green']}; background: {COLORS['surface_2']}; border-radius: 9px; padding: 3px 8px; font-weight: 600;"
        )
        heading.addWidget(self.count)
        layout.addLayout(heading)

        classifier_label = QLabel("Filename classifier")
        classifier_label.setObjectName("muted")
        layout.addWidget(classifier_label)
        self.classifier = QLineEdit(default_classifier)
        self.classifier.setPlaceholderText(default_classifier)
        self.classifier.setToolTip(
            "Case-insensitive filename text. Separate multiple classifiers with commas, for example: dark, d."
        )
        self.classifier.textChanged.connect(self.classifierChanged)
        layout.addWidget(self.classifier)

    def set_count(self, count: int) -> None:
        self.count.setText(f"{count} selected")
        color = COLORS["green"] if count else COLORS["muted"]
        self.count.setStyleSheet(
            f"color: {color}; background: {COLORS['surface_2']}; border-radius: 9px; padding: 3px 8px; font-weight: 600;"
        )


class DataTargetPage(QWidget):
    scanRequested = Signal(object)
    saveRequested = Signal(dict)
    targetLookupRequested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        header = PageHeader(
            "Data & Target", "Choose a FITS folder, verify the target, and confirm frame assignments."
        )
        outer.addWidget(header)

        body = QWidget()
        layout = QVBoxLayout(body)
        self.content_layout = layout
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        target_card = QFrame()
        target_card.setObjectName("card")
        self.target_card = target_card
        target_layout = QVBoxLayout(target_card)
        target_layout.setContentsMargins(18, 16, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Target coordinates")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "Coordinates are the canonical identity. The target name is a convenient label and is not required for plate solving."
            )
        )
        heading.addStretch()
        target_layout.addLayout(heading)
        form = QGridLayout()
        form.setHorizontalSpacing(15)
        form.setVerticalSpacing(10)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Optional, e.g. WTS-2 b")
        self.name.setToolTip(
            "Enter a SIMBAD star or exoplanet name. LEAPS looks up RA/DEC after you pause typing, press Enter, or leave the field."
        )
        self.target_lookup_timer = QTimer(self)
        self.target_lookup_timer.setSingleShot(True)
        self.target_lookup_timer.setInterval(800)
        self.target_lookup_timer.timeout.connect(self._request_target_lookup)
        self.name.returnPressed.connect(self._request_target_lookup)
        self.name.editingFinished.connect(self._request_target_lookup)
        self.name.textEdited.connect(self._target_name_edited)
        self.ra = QLineEdit()
        self.ra.setPlaceholderText("19:34:55.87")
        self.dec = QLineEdit()
        self.dec.setPlaceholderText("+36:48:55.79")
        form.addWidget(
            LabelWithInfo(
                "Target name", "A familiar label for reports. Plate solving uses the coordinates below."
            ),
            0,
            0,
        )
        form.addWidget(self.name, 0, 1)
        self.target_lookup_status = QLabel("Enter a name to look up coordinates automatically.")
        self.target_lookup_status.setObjectName("muted")
        form.addWidget(self.target_lookup_status, 1, 1)
        form.addWidget(
            LabelWithInfo(
                "Right ascension", "ICRS right ascension in hours, minutes, and seconds: hh:mm:ss."
            ),
            2,
            0,
        )
        form.addWidget(self.ra, 2, 1)
        form.addWidget(
            LabelWithInfo(
                "Declination", "ICRS declination in signed degrees, minutes, and seconds: +dd:mm:ss."
            ),
            3,
            0,
        )
        form.addWidget(self.dec, 3, 1)
        form.setColumnStretch(1, 1)
        target_layout.addLayout(form)
        self.target_source = QLabel("")
        self.target_source.setVisible(False)
        self.target_source.setWordWrap(True)
        target_layout.addWidget(self.target_source)

        folder_card = QFrame()
        folder_card.setObjectName("card")
        self.folder_card = folder_card
        folder_layout = QVBoxLayout(folder_card)
        folder_layout.setContentsMargins(18, 16, 18, 18)
        row = QHBoxLayout()
        title = QLabel("Observing run")
        title.setObjectName("sectionTitle")
        row.addWidget(title)
        row.addWidget(
            InfoButton(
                "LEAPS reads raw FITS files in place. It stores only project state and generated outputs in a portable .leaps folder beside them."
            )
        )
        row.addStretch()
        folder_layout.addLayout(row)
        pick = QHBoxLayout()
        self.folder = QLineEdit()
        self.folder.setReadOnly(True)
        self.folder.setPlaceholderText("Select the folder containing the FITS run")
        browse = ActionButton(
            "Choose folder",
            "fa6s.folder-open",
            tooltip="Choose a folder containing science and calibration FITS frames.",
        )
        browse.clicked.connect(self._choose_folder)
        pick.addWidget(self.folder, 1)
        pick.addWidget(browse)
        folder_layout.addLayout(pick)
        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        folder_layout.addWidget(self.scan_progress)
        layout.addWidget(folder_card)
        layout.addWidget(target_card)

        frames_card = QFrame()
        frames_card.setObjectName("card")
        self.frames_card = frames_card
        frames_layout = QVBoxLayout(frames_card)
        frames_layout.setContentsMargins(18, 16, 18, 18)
        top = QHBoxLayout()
        title = QLabel("Frame assignments")
        title.setObjectName("sectionTitle")
        top.addWidget(title)
        top.addWidget(
            InfoButton(
                "LEAPS classifies frames from FITS headers and filenames. Review low-confidence or unknown assignments before continuing."
            )
        )
        top.addStretch()
        self.counts = QLabel("No FITS files scanned")
        self.counts.setObjectName("muted")
        top.addWidget(self.counts)
        frames_layout.addLayout(top)

        cards = QGridLayout()
        cards.setHorizontalSpacing(12)
        cards.setVerticalSpacing(12)
        card_definitions = (
            (
                "bias",
                "Bias",
                "Bias",
                "fa6s.sliders",
                "Zero-exposure calibration frames used to remove the detector's electronic offset.",
            ),
            (
                "dark",
                "Darks",
                "Dark",
                "fa6s.moon",
                "Closed-shutter calibration frames used to remove thermal signal at the science exposure time.",
            ),
            (
                "flat",
                "Flats",
                "Flat",
                "fa6s.sun",
                "Evenly illuminated calibration frames used to correct dust and sensitivity variations.",
            ),
            (
                "science",
                "Science Images",
                "Image",
                "fa6s.star",
                "The target-field images recorded across the transit observing run.",
            ),
        )
        self.assignment_cards: dict[str, FrameAssignmentCard] = {}
        for index, (key, label, default, icon_name, tip) in enumerate(card_definitions):
            card = FrameAssignmentCard(label, default, icon_name, tip)
            card.classifierChanged.connect(self._refresh_assignments)
            self.assignment_cards[key] = card
            cards.addWidget(card, index // 2, index % 2)
        cards.setColumnStretch(0, 1)
        cards.setColumnStretch(1, 1)
        frames_layout.addLayout(cards)

        hint = QLabel(
            "Classifiers match filenames without regard to capitalization. Use commas for aliases, such as “dark, d”. Each file is assigned to the first matching box."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        frames_layout.addWidget(hint)
        self.bias_waiver = QCheckBox(
            "Continue without bias frames — my acquisition workflow does not require a separate bias set"
        )
        self.dark_waiver = QCheckBox(
            "Continue without dark frames — I accept the additional calibration risk"
        )
        self.flat_waiver = QCheckBox(
            "Continue without flat frames — I accept uncorrected illumination and dust effects"
        )
        waivers_layout = QVBoxLayout()
        waivers_layout.setContentsMargins(0, 6, 0, 2)
        waivers_layout.setSpacing(10)
        for waiver, tip in (
            (
                self.bias_waiver,
                "Required only when no bias frames are assigned. This decision is saved in the project manifest.",
            ),
            (
                self.dark_waiver,
                "Required only when no dark frames are assigned. This decision is saved in the project manifest.",
            ),
            (
                self.flat_waiver,
                "Required only when no flat frames are assigned. This decision is saved in the project manifest.",
            ),
        ):
            waiver.setToolTip(tip)
            waivers_layout.addWidget(waiver)
        frames_layout.addLayout(waivers_layout)
        layout.addWidget(frames_card)

        footer = QHBoxLayout()
        self.validation = QLabel("")
        self.validation.setObjectName("muted")
        footer.addWidget(self.validation, 1)
        save = ActionButton(
            "Confirm data & target",
            "fa6s.arrow-right",
            primary=True,
            tooltip="Validate coordinates and frame assignments, save this project, and unlock Reduction.",
        )
        save.clicked.connect(self._save)
        footer.addWidget(save)
        layout.addLayout(footer)
        layout.addStretch()
        self.scroll = _scroll_page(body)
        outer.addWidget(self.scroll, 1)
        self.records: list[FrameRecord] = []
        self.file_paths: list[str] = []
        self._lookup_requested_name = ""
        self._lookup_coordinate_snapshot = ("", "")
        self._last_resolved_coordinates: tuple[str, str] | None = None
        self.assignments: dict[str, list[str]] = {
            key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")
        }

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose observing run")
        if folder:
            self.name.clear()
            self.ra.clear()
            self.dec.clear()
            self.target_source.setVisible(False)
            self._target_name_edited()
            self.folder.setText(folder)
            self.preview_folder(Path(folder))
            self.scan_progress.setRange(0, 0)
            self.scan_progress.setVisible(True)
            self.scanRequested.emit(Path(folder))

    def _target_name_edited(self) -> None:
        self._lookup_requested_name = ""
        self.target_lookup_status.setText("Enter a name to look up coordinates automatically.")
        self.target_lookup_status.setStyleSheet("")
        self.target_lookup_timer.stop()
        if len(self.name.text().strip()) >= 2:
            self.target_lookup_timer.start()

    def _request_target_lookup(self) -> None:
        self.target_lookup_timer.stop()
        name = self.name.text().strip()
        if not name or name.casefold() == self._lookup_requested_name.casefold():
            return
        self._lookup_requested_name = name
        self._lookup_coordinate_snapshot = (self.ra.text(), self.dec.text())
        self.target_lookup_status.setText(f"Looking up {name}…")
        self.target_lookup_status.setStyleSheet(f"color: {COLORS['cyan']};")
        self.targetLookupRequested.emit(name)

    def apply_target_resolution(self, requested_name: str, resolved: ResolvedTarget) -> None:
        if self.name.text().strip().casefold() != requested_name.strip().casefold():
            return
        current = (self.ra.text(), self.dec.text())
        safe_to_fill = current == self._lookup_coordinate_snapshot and (
            not any(current) or current == self._last_resolved_coordinates
        )
        if safe_to_fill:
            self.ra.setText(resolved.ra)
            self.dec.setText(resolved.dec)
            self._last_resolved_coordinates = (resolved.ra, resolved.dec)
            self.target_lookup_status.setText(f"Coordinates found via {resolved.source}.")
            self.target_lookup_status.setStyleSheet(f"color: {COLORS['green']};")
        else:
            self.target_lookup_status.setText(
                f"Coordinates found via {resolved.source}; existing RA/DEC were kept."
            )
            self.target_lookup_status.setStyleSheet(f"color: {COLORS['amber']};")

    def mark_current_coordinates_as_saved(self) -> None:
        """Allow a later successful name lookup to replace unchanged saved coordinates."""
        self._last_resolved_coordinates = (self.ra.text(), self.dec.text())

    def show_target_lookup_error(self, requested_name: str, message: str) -> None:
        if self.name.text().strip().casefold() != requested_name.strip().casefold():
            return
        self._lookup_requested_name = ""
        self.target_lookup_status.setText(message)
        self.target_lookup_status.setStyleSheet(f"color: {COLORS['amber']};")

    def set_records(self, records: list[FrameRecord]) -> None:
        self.scan_progress.setVisible(False)
        self.records = list(records)
        self.file_paths = [record.path for record in records]
        self._refresh_assignments()

    def populate_target_from_records(self, records: list[FrameRecord]) -> None:
        assigned_science = set(self.assignments["science"])
        candidates = sorted(
            records,
            key=lambda record: (
                record.path not in assigned_science,
                record.category != "science",
            ),
        )
        detected = next((record for record in candidates if record.target_ra and record.target_dec), None)
        if detected is None:
            self.target_source.setText(
                "No target coordinates were found in the FITS headers. Enter them manually."
            )
            self.target_source.setStyleSheet(f"color: {COLORS['amber']};")
            self.target_source.setVisible(True)
            return
        if detected.target_name:
            self.name.setText(detected.target_name)
        self.ra.setText(detected.target_ra)
        self.dec.setText(detected.target_dec)
        self.target_source.setText(f"Detected from FITS header · {detected.path}")
        self.target_source.setStyleSheet(f"color: {COLORS['green']};")
        self.target_source.setVisible(True)

    def preview_folder(self, root: Path) -> None:
        """Populate live counts from filenames before FITS header inspection finishes."""
        self.file_paths = [
            path.relative_to(root).as_posix()
            for path in sorted(root.rglob("*"))
            if path.is_file()
            and path.suffix.casefold() in FITS_EXTENSIONS
            and ".leaps" not in path.parts
            and not any(part.startswith("reduction") or part.startswith("photometry") for part in path.parts)
        ]
        self._refresh_assignments()

    def set_assignment_patterns(self, patterns: dict[str, str]) -> None:
        for key, value in patterns.items():
            if key in self.assignment_cards and value:
                editor = self.assignment_cards[key].classifier
                blocked = editor.blockSignals(True)
                editor.setText(str(value))
                editor.blockSignals(blocked)
        if self.file_paths:
            self._refresh_assignments()

    def restore_project_assignments(
        self,
        assignments: dict[str, list[str]],
        patterns: dict[str, str] | None = None,
        waivers: dict[str, bool] | None = None,
    ) -> None:
        """Restore saved assignments without requiring another FITS scan."""
        if patterns:
            self.set_assignment_patterns(patterns)
        keys = ("science", "bias", "dark", "dark_flat", "flat", "unknown")
        restored = {key: list(assignments.get(key, [])) for key in keys}
        self.file_paths = list(dict.fromkeys(path for key in keys for path in restored[key]))
        self.records = []
        self._set_assignments(restored)
        decisions = waivers or {}
        self.bias_waiver.setChecked(bool(decisions.get("bias", False)))
        self.dark_waiver.setChecked(bool(decisions.get("dark", False)))
        self.flat_waiver.setChecked(bool(decisions.get("flat", False)))

    def assignment_patterns(self) -> dict[str, str]:
        return {key: card.classifier.text().strip() for key, card in self.assignment_cards.items()}

    def _refresh_assignments(self) -> None:
        assignments = {key: [] for key in ("science", "bias", "dark", "dark_flat", "flat", "unknown")}
        order = ("bias", "dark", "flat", "science")
        patterns = {
            key: [token.strip().casefold() for token in card.classifier.text().split(",") if token.strip()]
            for key, card in self.assignment_cards.items()
        }
        for path in self.file_paths:
            stem = Path(path).stem.casefold()
            segments = [segment for segment in re.split(r"[^a-z0-9]+", stem) if segment]
            category = next(
                (
                    key
                    for key in order
                    if any(self._matches_classifier(stem, segments, token) for token in patterns[key])
                ),
                "unknown",
            )
            assignments[category].append(path)
        self._set_assignments(assignments)

    def _set_assignments(self, assignments: dict[str, list[str]]) -> None:
        self.assignments = assignments
        for key, card in self.assignment_cards.items():
            card.set_count(len(assignments[key]))
        assigned = sum(len(assignments[key]) for key in ("bias", "dark", "flat", "science"))
        unmatched = len(assignments["unknown"])
        if not self.file_paths:
            self.counts.setText("No FITS files scanned")
        else:
            self.counts.setText(f"{assigned} assigned · {unmatched} unmatched")

    @staticmethod
    def _matches_classifier(stem: str, segments: list[str], token: str) -> bool:
        if len(token) == 1:
            return any(segment == token or segment.startswith(token) for segment in segments)
        return token in stem

    def _save(self) -> None:
        self.saveRequested.emit(
            {
                "root": self.folder.text().strip(),
                "target_name": self.name.text().strip(),
                "ra": self.ra.text().strip(),
                "dec": self.dec.text().strip(),
                "waivers": {
                    "bias": self.bias_waiver.isChecked(),
                    "dark": self.dark_waiver.isChecked(),
                    "flat": self.flat_waiver.isChecked(),
                },
                "assignments": {key: list(paths) for key, paths in self.assignments.items()},
                "frame_classifiers": self.assignment_patterns(),
            }
        )

    def clear_section_errors(self) -> None:
        for card in (self.folder_card, self.target_card, self.frames_card):
            card.setProperty("validationError", False)
            card.style().unpolish(card)
            card.style().polish(card)
        self.validation.clear()

    def show_error(self, message: str, section: str | None = None) -> None:
        self.clear_section_errors()
        self.validation.setText(message)
        self.validation.setStyleSheet(f"color: {COLORS['amber']};")
        cards = {
            "folder": self.folder_card,
            "target": self.target_card,
            "frames": self.frames_card,
        }
        card = cards.get(section or "")
        if card is not None:
            card.setProperty("validationError", True)
            card.style().unpolish(card)
            card.style().polish(card)
            QTimer.singleShot(0, lambda: self.scroll.ensureWidgetVisible(card, 20, 20))


class ProcessingPage(QWidget):
    runRequested = Signal(object)
    cancelRequested = Signal()

    def __init__(
        self, stage: StageID, title: str, subtitle: str, options: list[tuple[str, str]], parent=None
    ) -> None:
        super().__init__(parent)
        self.stage = stage
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PageHeader(title, subtitle))
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)

        option_card = QFrame()
        option_card.setObjectName("card")
        option_layout = QVBoxLayout(option_card)
        option_layout.setContentsMargins(18, 16, 18, 18)
        heading = QLabel("Processing options")
        heading.setObjectName("sectionTitle")
        option_layout.addWidget(heading)
        self.option_widgets: dict[str, QCheckBox] = {}
        for label, tooltip in options:
            row = QHBoxLayout()
            check = QCheckBox(label)
            check.setChecked(True)
            check.setToolTip(tooltip)
            row.addWidget(check)
            row.addWidget(InfoButton(tooltip))
            row.addStretch()
            option_layout.addLayout(row)
            self.option_widgets[label] = check
        layout.addWidget(option_card)

        progress_card = QFrame()
        progress_card.setObjectName("card")
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(18, 16, 18, 18)
        top = QHBoxLayout()
        self.status = QLabel("Ready")
        self.status.setObjectName("sectionTitle")
        top.addWidget(self.status)
        top.addStretch()
        self.counter = QLabel("")
        self.counter.setObjectName("muted")
        top.addWidget(self.counter)
        progress_layout.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        progress_layout.addWidget(self.progress)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Progress details and checkpoints will appear here.")
        self.log.setMinimumHeight(220)
        progress_layout.addWidget(self.log)
        layout.addWidget(progress_card)

        actions = QHBoxLayout()
        actions.addStretch()
        self.cancel = ActionButton(
            "Cancel safely",
            "fa6s.stop",
            tooltip="Stop after the current safe checkpoint. Completed outputs remain intact.",
        )
        self.cancel.setEnabled(False)
        self.cancel.clicked.connect(self.cancelRequested)
        self.run = ActionButton(
            f"Run {title}",
            "fa6s.play",
            primary=True,
            tooltip=f"Run {title.lower()} in the background without freezing the interface.",
        )
        self.run.clicked.connect(lambda: self.runRequested.emit(self.stage))
        actions.addWidget(self.cancel)
        actions.addWidget(self.run)
        layout.addLayout(actions)
        layout.addStretch()
        outer.addWidget(_scroll_page(body), 1)

    def set_busy(self, busy: bool) -> None:
        self.run.setEnabled(not busy)
        self.cancel.setEnabled(busy)

    def update_event(self, event: StageEvent) -> None:
        self.status.setText(event.message)
        self.progress.setValue(round(event.fraction * 100))
        self.counter.setText(f"{event.current} of {event.total}" if event.total else "")
        self.log.appendPlainText(event.message)

    def set_failure(self, failure: LEAPSError) -> None:
        self.status.setText(failure.title)
        self.status.setStyleSheet(f"color: {COLORS['amber']};")
        self.log.appendPlainText(f"{failure.code}: {failure.message}")
        self.log.appendPlainText("Next: " + " · ".join(failure.recovery))


class TimelineRow(QWidget):
    def __init__(self, time: str, text: str, status: str, detail: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet("QWidget, QLabel { background: transparent; }")
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        row.setSpacing(11)
        icon_names = {
            "ok": ("fa6s.circle-check", COLORS["green"]),
            "error": ("fa6s.circle-xmark", COLORS["red"]),
            "paused": ("fa6s.circle-pause", COLORS["amber"]),
            "waiting": ("fa6s.circle", COLORS["muted_2"]),
        }
        name, color = icon_names[status]
        picture = QLabel()
        picture.setPixmap(icon(name, color).pixmap(21, 21))
        picture.setFixedWidth(24)
        row.addWidget(picture)
        timestamp = QLabel(time)
        timestamp.setObjectName("muted")
        timestamp.setFixedWidth(62 if time else 0)
        timestamp.setVisible(bool(time))
        row.addWidget(timestamp)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        primary = QLabel(text)
        primary.setWordWrap(True)
        primary.setMinimumWidth(0)
        labels.addWidget(primary)
        if detail:
            secondary = QLabel(detail)
            secondary.setObjectName("muted")
            secondary.setWordWrap(True)
            labels.addWidget(secondary)
        row.addLayout(labels, 1)


class RecoveryInspector(QFrame):
    retryRequested = Signal()
    manualRequested = Signal()
    comparisonRequested = Signal()
    rankRequested = Signal()
    runRequested = Signal(list, float)
    copyRequested = Signal()
    comparisonActiveChanged = Signal(int, bool)
    comparisonRemoved = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("recoveryInspector")
        self.setMinimumWidth(320)
        self.setMaximumWidth(420)
        self.setStyleSheet(
            f"QFrame#recoveryInspector {{background: {COLORS['surface']}; border-left: 1px solid {COLORS['border_soft']};}}"
            "QLabel, QPushButton { font-size: 14px; }"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.banner = QFrame()
        self.banner.setStyleSheet(
            f"background: {COLORS['surface_3']}; border-bottom: 1px solid {COLORS['border']};"
        )
        banner_layout = QHBoxLayout(self.banner)
        banner_layout.setContentsMargins(19, 13, 15, 13)
        self.banner_icon = QLabel()
        self.banner_icon.setPixmap(icon("fa6s.crosshairs", COLORS["cyan"]).pixmap(24, 24))
        self.banner_title = QLabel("Photometry setup")
        self.banner_title.setStyleSheet(f"color: {COLORS['cyan']}; font-size: 16px; font-weight: 650;")
        banner_layout.addWidget(self.banner_icon)
        banner_layout.addWidget(self.banner_title, 1)
        layout.addWidget(self.banner)

        scroll_content = QWidget()
        scroll_content.setObjectName("recoveryContent")
        scroll_content.setStyleSheet(
            f"QWidget#recoveryContent {{ background: {COLORS['surface']}; }} QLabel {{ background: transparent; }}"
        )
        content = QVBoxLayout(scroll_content)
        content.setContentsMargins(20, 17, 20, 22)
        content.setSpacing(15)
        self.explanation = QLabel(
            "Select the target and comparison stars in the real reduced FITS image. Plate solving can locate the target automatically, but it is optional."
        )
        self.explanation.setWordWrap(True)
        self.explanation.setStyleSheet("font-size: 14px; line-height: 1.4;")
        content.addWidget(self.explanation)
        content.addWidget(self._divider())

        row = QHBoxLayout()
        heading = QLabel("Solve attempt timeline")
        heading.setObjectName("sectionTitle")
        row.addWidget(heading)
        row.addWidget(
            InfoButton(
                "Every bounded plate-solve attempt is recorded so you can see exactly where and why it stopped."
            )
        )
        row.addStretch()
        content.addLayout(row)
        self.timeline = QVBoxLayout()
        self._demo_timeline()
        content.addLayout(self.timeline)
        content.addWidget(self._divider())

        target_header = QHBoxLayout()
        target_title = QLabel("Target information")
        target_title.setObjectName("sectionTitle")
        target_header.addWidget(target_title)
        target_header.addWidget(InfoButton("These validated values were used for all solve attempts."))
        target_header.addStretch()
        content.addLayout(target_header)
        info = QGridLayout()
        info.setVerticalSpacing(10)
        info.setColumnStretch(1, 1)
        self.target_name = QLabel("Unnamed target")
        self.target_name.setWordWrap(True)
        self.coordinates = QLabel("Coordinates not set")
        self.coordinates.setWordWrap(True)
        self.pixel_scale = QLabel("Not set")
        self.pixel_scale.setWordWrap(True)
        for value in (self.target_name, self.coordinates, self.pixel_scale):
            value.setMinimumWidth(0)
        info.addWidget(QLabel("Target"), 0, 0)
        info.addWidget(self.target_name, 0, 1)
        info.addWidget(
            LabelWithInfo(
                "Coordinates (ICRS)", "Validated ICRS sky coordinates used as the canonical target identity."
            ),
            1,
            0,
        )
        info.addWidget(self.coordinates, 1, 1)
        info.addWidget(
            LabelWithInfo(
                "Pixel scale",
                "Estimated sky angle covered by each image pixel. A close estimate helps match stars to Gaia.",
            ),
            2,
            0,
        )
        info.addWidget(self.pixel_scale, 2, 1)
        content.addLayout(info)

        tooltip_card = QFrame()
        tooltip_card.setObjectName("card")
        tip_layout = QHBoxLayout(tooltip_card)
        tip_layout.setContentsMargins(12, 10, 12, 10)
        tip_layout.addWidget(
            InfoButton(
                "Pixel scale can usually be calculated from camera pixel size and telescope focal length."
            )
        )
        tip = QLabel(
            "Estimated sky angle covered by each image pixel. A close estimate helps match stars to Gaia."
        )
        tip.setWordWrap(True)
        tip.setObjectName("muted")
        tip_layout.addWidget(tip, 1)
        content.addWidget(tooltip_card)

        self.retry = ActionButton(
            "Locate target from coordinates",
            "fa6s.rotate",
            primary=True,
            tooltip="Retry with validated coordinates and pixel scale, up to the remaining bounded attempts.",
        )
        self.retry.setMinimumHeight(44)
        self.retry.clicked.connect(self.retryRequested)
        self.manual = ActionButton(
            "Set target",
            "fa6s.crosshairs",
            tooltip="Click the target in the FITS image. Results remain clearly marked as unverified WCS.",
        )
        self.manual.setMinimumHeight(44)
        self.manual.setStyleSheet(
            f"color: {COLORS['cyan']}; border: 1px solid {COLORS['cyan']}; background: transparent;"
        )
        self.manual.clicked.connect(self.manualRequested)
        self.copy = ActionButton(
            "Copy diagnostics",
            "fa6s.copy",
            tooltip="Copy the failure code, diagnostic reference, attempt timeline, and validated inputs.",
        )
        self.copy.setStyleSheet(f"color: {COLORS['cyan']}; border: 0; background: transparent;")
        self.copy.clicked.connect(self.copyRequested)
        content.addWidget(self.retry)
        content.addWidget(self.manual)
        content.addWidget(self._divider())

        stars_heading = QHBoxLayout()
        stars_title = QLabel("Selected stars")
        stars_title.setObjectName("sectionTitle")
        stars_heading.addWidget(stars_title)
        stars_heading.addWidget(
            InfoButton(
                "HOPS refines each click to the nearest acceptable star and tracks it through the aligned sequence."
            )
        )
        stars_heading.addStretch()
        content.addLayout(stars_heading)
        self.target_selection = QLabel("Target: not selected")
        self.target_selection.setObjectName("muted")
        content.addWidget(self.target_selection)
        self.comparison_selection = QLabel("Comparisons: 0 selected")
        self.comparison_selection.setObjectName("muted")
        content.addWidget(self.comparison_selection)
        self.comparison_rows_widget = QWidget()
        self.comparison_rows = QVBoxLayout(self.comparison_rows_widget)
        self.comparison_rows.setContentsMargins(0, 0, 0, 0)
        self.comparison_rows.setSpacing(5)
        content.addWidget(self.comparison_rows_widget)
        self.add_comparison = ActionButton(
            "Add comparison star",
            "fa6s.plus",
            tooltip="Click another stable, unsaturated star in the FITS image.",
        )
        self.add_comparison.clicked.connect(self.comparisonRequested)
        content.addWidget(self.add_comparison)
        self.rank = ActionButton(
            "Suggest comparison stars",
            "fa6s.wand-magic-sparkles",
            tooltip="Rank nearby stars with usable brightness and automatically propose an ensemble.",
        )
        self.rank.clicked.connect(self.rankRequested)
        content.addWidget(self.rank)

        aperture_row = QHBoxLayout()
        aperture_row.addWidget(
            LabelWithInfo(
                "Aperture radius",
                "Radius in pixels. HOPS scales it with each frame's PSF when variable aperture is enabled.",
            )
        )
        self.aperture = QDoubleSpinBox()
        self.aperture.setRange(1.6, 100.0)
        self.aperture.setDecimals(1)
        self.aperture.setValue(8.0)
        self.aperture.setSuffix(" px")
        aperture_row.addWidget(self.aperture)
        content.addLayout(aperture_row)

        self.advanced_button = ActionButton(
            "Advanced settings",
            "fa6s.sliders",
            tooltip="Show the original HOPS aperture, sky, saturation, and camera controls.",
        )
        self.advanced_button.setCheckable(True)
        content.addWidget(self.advanced_button)
        self.advanced = QFrame()
        self.advanced.setObjectName("card")
        advanced_form = QFormLayout(self.advanced)
        advanced_form.setContentsMargins(12, 12, 12, 12)
        self.variable_aperture = QCheckBox("Scale aperture with PSF")
        self.variable_aperture.setChecked(True)
        self.geometric_center = QCheckBox("Use geometric center")
        self.sky_inner = QDoubleSpinBox()
        self.sky_inner.setRange(1.01, 20.0)
        self.sky_inner.setValue(1.7)
        self.sky_outer = QDoubleSpinBox()
        self.sky_outer.setRange(1.02, 30.0)
        self.sky_outer.setValue(2.4)
        self.saturation = QDoubleSpinBox()
        self.saturation.setRange(0.01, 1.0)
        self.saturation.setSingleStep(0.05)
        self.saturation.setValue(0.95)
        self.camera_gain = QDoubleSpinBox()
        self.camera_gain.setRange(0.01, 1000.0)
        self.camera_gain.setValue(1.0)
        self.camera_gain.setSuffix(" e⁻/ADU")
        advanced_form.addRow(self.variable_aperture)
        advanced_form.addRow(self.geometric_center)
        advanced_form.addRow("Inner sky ring", self.sky_inner)
        advanced_form.addRow("Outer sky ring", self.sky_outer)
        advanced_form.addRow("Saturation fraction", self.saturation)
        advanced_form.addRow("Camera gain", self.camera_gain)
        self.advanced.setVisible(False)
        self.advanced_button.toggled.connect(self.advanced.setVisible)
        content.addWidget(self.advanced)

        self.run = ActionButton(
            "Run HOPS photometry",
            "fa6s.play",
            primary=True,
            tooltip="Track the selected stars and calculate aperture and Gaussian light curves in the background.",
        )
        self.run.setEnabled(False)
        self.run.clicked.connect(self._run)
        content.addWidget(self.run)
        content.addWidget(self.copy)
        content.addStretch()
        scroll = QScrollArea()
        scroll.setStyleSheet(f"QScrollArea {{ background: {COLORS['surface']}; border: 0; }}")
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

    def _demo_timeline(self) -> None:
        self.timeline.addWidget(
            TimelineRow("", "Plate solve is optional · manual selection is ready", "waiting")
        )

    def set_retrying(self) -> None:
        self.retry.setEnabled(False)
        self.retry.setText("Locating target…")

    def set_failure(self, failure: LEAPSError) -> None:
        """Replace the demo context with the exact typed runtime failure."""
        self.banner_title.setText(failure.title)
        self.banner_icon.setPixmap(
            icon("fa6s.triangle-exclamation", COLORS["amber"]).pixmap(24, 24)
        )
        self.banner.setStyleSheet(
            f"background: #6d4703; border-bottom: 1px solid {COLORS['amber_dark']};"
        )
        self.banner_title.setStyleSheet(
            f"color: {COLORS['amber']}; font-size: 16px; font-weight: 650;"
        )
        self.explanation.setText(failure.message)
        while self.timeline.count():
            item = self.timeline.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        current = QTime.currentTime().toString("HH:mm:ss")
        self.timeline.addWidget(TimelineRow(current, "Coordinates and pixel scale validated", "ok"))
        details = [line.strip() for line in failure.technical_details.splitlines() if line.strip()]
        for detail in details[-3:]:
            title, _, secondary = detail.partition(":")
            self.timeline.addWidget(TimelineRow(current, title or failure.code, "error", secondary.strip()))
        self.timeline.addWidget(TimelineRow(current, f"Stopped safely · {failure.diagnostic_id}", "paused"))
        self.retry.setEnabled(True)
        self.retry.setText("Retry plate solve")

    def set_project_target(self, name: str, coordinates: str, pixel_scale: float) -> None:
        self.target_name.setText(name or "Unnamed target")
        self.coordinates.setText(coordinates or "Coordinates not set")
        self.pixel_scale.setText(
            f"{pixel_scale:.2f} arcsec/pixel" if pixel_scale > 0 else "Estimated from stellar PSF"
        )

    def set_target_selected(self, x: float, y: float, verified: bool) -> None:
        suffix = "plate solved" if verified else "manual"
        self.target_selection.setText(f"Target: x {x:.1f}, y {y:.1f} · {suffix}")
        self._update_run_state()

    def set_comparisons(
        self, comparisons: list[tuple[float, float]], active: list[bool]
    ) -> None:
        while self.comparison_rows.count():
            item = self.comparison_rows.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        active_count = sum(active)
        self.comparison_selection.setText(
            f"Comparisons: {active_count} active · {len(comparisons)} selected"
        )
        for index, ((x, y), enabled) in enumerate(zip(comparisons, active, strict=True)):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(6)
            use = QCheckBox(f"C{index + 1} · x {x:.1f}, y {y:.1f}")
            use.setChecked(enabled)
            use.setToolTip("Include this comparison star in the differential-flux ensemble.")
            use.toggled.connect(
                lambda checked, row_index=index: self.comparisonActiveChanged.emit(
                    row_index, checked
                )
            )
            remove = QPushButton()
            remove.setIcon(icon("fa6s.xmark", COLORS["muted"]))
            remove.setToolTip("Remove this comparison star.")
            remove.setAccessibleName(f"Remove comparison star {index + 1}")
            remove.setFixedSize(28, 28)
            remove.clicked.connect(
                lambda _checked=False, row_index=index: self.comparisonRemoved.emit(row_index)
            )
            row_layout.addWidget(use, 1)
            row_layout.addWidget(remove)
            self.comparison_rows.addWidget(row)
        self._update_run_state()

    def _update_run_state(self) -> None:
        self.run.setEnabled(
            "not selected" not in self.target_selection.text()
            and not self.comparison_selection.text().startswith("Comparisons: 0 active")
        )

    def _run(self) -> None:
        self.runRequested.emit([], self.aperture.value())

    def photometry_config(self) -> dict[str, float | bool]:
        return {
            "aperture_radius": self.aperture.value(),
            "sky_inner_aperture": self.sky_inner.value(),
            "sky_outer_aperture": self.sky_outer.value(),
            "saturation_fraction": self.saturation.value(),
            "camera_gain": self.camera_gain.value(),
            "variable_aperture": self.variable_aperture.isChecked(),
            "geometric_center": self.geometric_center.isChecked(),
        }

    def apply_photometry_config(self, values: dict[str, Any]) -> None:
        if not values:
            return
        self.aperture.setValue(float(values.get("aperture_radius", self.aperture.value())))
        self.sky_inner.setValue(float(values.get("sky_inner_aperture", self.sky_inner.value())))
        self.sky_outer.setValue(float(values.get("sky_outer_aperture", self.sky_outer.value())))
        self.saturation.setValue(float(values.get("saturation_fraction", self.saturation.value())))
        self.camera_gain.setValue(float(values.get("camera_gain", self.camera_gain.value())))
        self.variable_aperture.setChecked(
            bool(values.get("variable_aperture", self.variable_aperture.isChecked()))
        )
        self.geometric_center.setChecked(
            bool(values.get("geometric_center", self.geometric_center.isChecked()))
        )

    def _restore_retry(self) -> None:
        self.retry.setEnabled(True)
        self.retry.setText("Locate target from coordinates")

    @staticmethod
    def _divider() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet(f"color: {COLORS['border_soft']};")
        return line


class PlateSolvePage(QWidget):
    retryRequested = Signal()
    copyDiagnosticsRequested = Signal()
    manualTargetPlaced = Signal(float, float)
    starSelectionRequested = Signal(str, float, float)
    rankRequested = Signal()
    runRequested = Signal(list, float)
    selectionChanged = Signal()

    def __init__(self, asset: Path, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Photometry",
                "Select the target and comparison stars, then run the original HOPS measurement workflow.",
            )
        )
        split = QHBoxLayout()
        split.setContentsMargins(0, 0, 0, 0)
        split.setSpacing(0)
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(16, 14, 16, 14)
        self.workspace = FITSWorkspace(asset)
        center_layout.addWidget(self.workspace)
        split.addWidget(center, 1)
        self.inspector = RecoveryInspector()
        self.target: tuple[float, float] | None = None
        self.target_label = "Target"
        self.target_verified = False
        self.comparisons: list[tuple[float, float]] = []
        self.comparison_active: list[bool] = []
        split.addWidget(self.inspector)
        outer.addLayout(split, 1)
        self.inspector.retryRequested.connect(self._retry)
        self.inspector.manualRequested.connect(self.workspace.begin_manual_target)
        self.inspector.comparisonRequested.connect(
            lambda: self.workspace.begin_selection("comparison")
        )
        self.inspector.rankRequested.connect(self.rankRequested)
        self.inspector.runRequested.connect(
            lambda _ignored, radius: self.runRequested.emit(
                [
                    comparison
                    for comparison, active in zip(
                        self.comparisons, self.comparison_active, strict=True
                    )
                    if active
                ],
                radius,
            )
        )
        self.inspector.comparisonActiveChanged.connect(self.set_comparison_active)
        self.inspector.comparisonRemoved.connect(self.remove_comparison)
        self.inspector.aperture.valueChanged.connect(self.set_aperture_radius)
        self.inspector.sky_inner.valueChanged.connect(self.sky_ring_changed)
        self.inspector.sky_outer.valueChanged.connect(self.sky_ring_changed)
        for control in (
            self.inspector.variable_aperture,
            self.inspector.geometric_center,
            self.inspector.saturation,
            self.inspector.camera_gain,
        ):
            if isinstance(control, QCheckBox):
                control.toggled.connect(lambda _checked: self.selectionChanged.emit())
            else:
                control.valueChanged.connect(lambda _value: self.selectionChanged.emit())
        self.inspector.copyRequested.connect(self.copyDiagnosticsRequested)
        self.workspace.pointSelected.connect(self.starSelectionRequested)

    def _retry(self) -> None:
        self.inspector.set_retrying()
        self.retryRequested.emit()

    def set_target(self, x: float, y: float, *, radius: float, label: str, verified: bool) -> None:
        self.target = (x, y)
        self.target_label = label
        self.target_verified = verified
        self.workspace.place_target_marker(
            x,
            y,
            radius,
            label,
            sky_inner=self.inspector.sky_inner.value(),
            sky_outer=self.inspector.sky_outer.value(),
        )
        self.inspector.aperture.setValue(radius)
        self.inspector.set_target_selected(x, y, verified)
        self.selectionChanged.emit()

    def add_comparison(
        self, x: float, y: float, *, radius: float, active: bool = True
    ) -> None:
        if any((cx - x) ** 2 + (cy - y) ** 2 < 4 for cx, cy in self.comparisons):
            return
        self.comparisons.append((x, y))
        self.comparison_active.append(active)
        self._refresh_comparisons(radius)
        self.selectionChanged.emit()

    def set_comparison_active(self, index: int, active: bool) -> None:
        if 0 <= index < len(self.comparison_active):
            self.comparison_active[index] = active
            self.workspace.image.set_marker_active(f"comparison-{index + 1}", active)
            self.inspector.set_comparisons(self.comparisons, self.comparison_active)
            self.selectionChanged.emit()

    def remove_comparison(self, index: int) -> None:
        if 0 <= index < len(self.comparisons):
            self.comparisons.pop(index)
            self.comparison_active.pop(index)
            self._refresh_comparisons(self.inspector.aperture.value())
            self.selectionChanged.emit()

    def _refresh_comparisons(self, radius: float) -> None:
        for key in list(self.workspace.image.marker_items):
            if key.startswith("comparison-"):
                self.workspace.image.remove_marker(key)
        for index, ((x, y), active) in enumerate(
            zip(self.comparisons, self.comparison_active, strict=True), start=1
        ):
            self.workspace.place_comparison_marker(
                index,
                x,
                y,
                radius,
                active=active,
                sky_inner=self.inspector.sky_inner.value(),
                sky_outer=self.inspector.sky_outer.value(),
            )
        self.inspector.set_comparisons(self.comparisons, self.comparison_active)

    def set_aperture_radius(self, radius: float) -> None:
        self.refresh_aperture_overlays()
        self.selectionChanged.emit()

    def sky_ring_changed(self, _value: float) -> None:
        self.refresh_aperture_overlays()
        self.selectionChanged.emit()

    def refresh_aperture_overlays(self) -> None:
        radius = self.inspector.aperture.value()
        if self.target:
            self.workspace.place_target_marker(
                self.target[0],
                self.target[1],
                radius,
                self.target_label,
                sky_inner=self.inspector.sky_inner.value(),
                sky_outer=self.inspector.sky_outer.value(),
            )
        self._refresh_comparisons(radius)

    def set_candidates(self, candidates: list[dict[str, float]]) -> None:
        for candidate in candidates[:5]:
            self.add_comparison(
                float(candidate["x"]),
                float(candidate["y"]),
                radius=self.inspector.aperture.value(),
            )

    def clear_selection(self) -> None:
        self.target = None
        self.target_verified = False
        self.comparisons = []
        self.comparison_active = []
        self.workspace.image.clear_markers()
        self.inspector.target_selection.setText("Target: not selected")
        self.inspector.set_comparisons([], [])


class FittingPage(QWidget):
    previewRequested = Signal(dict)
    fullFitRequested = Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Fitting",
                "Preview the transit model first, then run a full uncertainty analysis when the setup looks right.",
            )
        )
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)
        form_card = QFrame()
        form_card.setObjectName("card")
        form_layout = QVBoxLayout(form_card)
        form_layout.setContentsMargins(18, 16, 18, 18)
        title = QLabel("Planet parameters")
        title.setObjectName("sectionTitle")
        form_layout.addWidget(title)
        form = QFormLayout()
        form.setSpacing(10)
        self.planet = QLineEdit()
        self.planet.setPlaceholderText("WTS-2 b")
        self.period = QDoubleSpinBox()
        self.period.setDecimals(8)
        self.period.setRange(0.000001, 100000)
        self.period.setValue(1.0187068)
        self.mid_time = QDoubleSpinBox()
        self.mid_time.setDecimals(8)
        self.mid_time.setRange(0, 4_000_000)
        self.mid_time.setValue(2461220.42)
        self.depth = QDoubleSpinBox()
        self.depth.setDecimals(5)
        self.depth.setRange(0, 1)
        self.depth.setValue(0.03)
        form.addRow(
            LabelWithInfo(
                "Planet",
                "Resolve parameters from ExoClock, then the bundled NASA snapshot, or reuse validated manual values.",
            ),
            self.planet,
        )
        form.addRow(LabelWithInfo("Period (days)", "Time between consecutive transits."), self.period)
        form.addRow(
            LabelWithInfo("Mid-transit (BJD)", "Expected center of the transit in barycentric Julian date."),
            self.mid_time,
        )
        form.addRow(
            LabelWithInfo("Expected depth", "Approximate fractional loss of light during transit."),
            self.depth,
        )
        form_layout.addLayout(form)
        self.advanced_toggle = QPushButton("Advanced MCMC controls")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setToolTip(
            "Show sampling controls. Defaults are recommended for most observing runs."
        )
        form_layout.addWidget(self.advanced_toggle)
        self.advanced = QWidget()
        advanced_form = QFormLayout(self.advanced)
        self.walkers = QSpinBox()
        self.walkers.setRange(20, 1000)
        self.walkers.setValue(100)
        self.iterations = QSpinBox()
        self.iterations.setRange(100, 1_000_000)
        self.iterations.setValue(5000)
        self.burn = QSpinBox()
        self.burn.setRange(0, 900_000)
        self.burn.setValue(1000)
        advanced_form.addRow(
            LabelWithInfo("Walkers", "Independent MCMC chains used to explore the posterior."), self.walkers
        )
        advanced_form.addRow(
            LabelWithInfo("Iterations", "Samples generated per walker for the full fit."), self.iterations
        )
        advanced_form.addRow(
            LabelWithInfo("Burn-in", "Initial samples discarded before summarizing the posterior."), self.burn
        )
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        form_layout.addWidget(self.advanced)
        form_layout.addStretch()
        buttons = QHBoxLayout()
        preview = ActionButton(
            "Preview Fit",
            "fa6s.chart-line",
            tooltip="Run a quick deterministic fit to validate data, timing, and priors.",
        )
        preview.clicked.connect(lambda: self.previewRequested.emit(self.values()))
        full = ActionButton(
            "Run Full Fit",
            "fa6s.play",
            primary=True,
            tooltip="Run the full MCMC uncertainty analysis in the background.",
        )
        full.clicked.connect(lambda: self.fullFitRequested.emit(self.values()))
        buttons.addWidget(preview)
        buttons.addWidget(full)
        form_layout.addLayout(buttons)
        layout.addWidget(form_card, 2)
        result = QFrame()
        result.setObjectName("card")
        result_layout = QVBoxLayout(result)
        result_layout.setContentsMargins(18, 16, 18, 18)
        result_title = QLabel("Fit preview")
        result_title.setObjectName("sectionTitle")
        result_layout.addWidget(result_title)
        message = QLabel(
            "Run Preview Fit to inspect the model and residuals before committing time to the full fit."
        )
        message.setWordWrap(True)
        message.setObjectName("muted")
        result_layout.addWidget(message)
        result_layout.addStretch()
        layout.addWidget(result, 3)
        outer.addWidget(_scroll_page(body), 1)

    def values(self) -> dict[str, Any]:
        return {
            "planet": self.planet.text().strip(),
            "period": self.period.value(),
            "mid_time": self.mid_time.value(),
            "depth": self.depth.value(),
            "walkers": self.walkers.value(),
            "iterations": self.iterations.value(),
            "burn": self.burn.value(),
        }


class ComparisonStarsPage(QWidget):
    rankRequested = Signal()
    runRequested = Signal(list, float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.candidates: list[dict[str, float]] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Apertures & Comparison Stars",
                "Review ranked stars and approve the ensemble used for differential photometry.",
            )
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(18, 16, 18, 18)
        heading = QHBoxLayout()
        title = QLabel("Ranked comparison stars")
        title.setObjectName("sectionTitle")
        heading.addWidget(title)
        heading.addWidget(
            InfoButton(
                "Candidates are ranked by usable brightness, distance from the target, saturation margin, and measurement stability. You always approve the final ensemble."
            )
        )
        heading.addStretch()
        rank = ActionButton(
            "Rank candidates",
            "fa6s.wand-magic-sparkles",
            tooltip="Detect and rank comparison stars in the reference image without beginning photometry.",
        )
        rank.clicked.connect(self.rankRequested)
        heading.addWidget(rank)
        card_layout.addLayout(heading)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Use", "Rank", "X", "Y", "Peak", "Score"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setAlternatingRowColors(True)
        card_layout.addWidget(self.table, 1)
        controls = QHBoxLayout()
        controls.addWidget(
            LabelWithInfo(
                "Aperture radius",
                "Circular aperture radius in pixels. Start near 1.5–2 times the measured stellar FWHM.",
            )
        )
        self.radius = QDoubleSpinBox()
        self.radius.setRange(1, 100)
        self.radius.setDecimals(1)
        self.radius.setValue(8.0)
        self.radius.setSuffix(" px")
        controls.addWidget(self.radius)
        controls.addStretch()
        run = ActionButton(
            "Run photometry",
            "fa6s.play",
            primary=True,
            tooltip="Measure the target and approved comparisons across every accepted aligned frame in the background.",
        )
        run.clicked.connect(self._run)
        controls.addWidget(run)
        card_layout.addLayout(controls)
        layout.addWidget(card, 1)
        self.status = QLabel("Plate solve or manually place the target, then rank candidates.")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        outer.addWidget(body, 1)

    def set_candidates(self, candidates: list[dict[str, float]]) -> None:
        self.candidates = candidates
        self.table.setRowCount(len(candidates))
        for row, candidate in enumerate(candidates):
            use = QTableWidgetItem()
            use.setFlags(use.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(Qt.CheckState.Checked if row < 5 else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, use)
            values = (
                row + 1,
                f"{candidate['x']:.1f}",
                f"{candidate['y']:.1f}",
                f"{candidate['peak']:.0f}",
                f"{candidate['score']:.3f}",
            )
            for column, value in enumerate(values, start=1):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()
        self.status.setText(
            f"{len(candidates)} candidates ranked. Review the checked ensemble before running photometry."
        )

    def _run(self) -> None:
        selected = []
        for row, candidate in enumerate(self.candidates):
            if self.table.item(row, 0).checkState() == Qt.CheckState.Checked:
                selected.append((candidate["x"], candidate["y"]))
        self.runRequested.emit(selected, self.radius.value())


class ReportsPage(QWidget):
    openFolderRequested = Signal()
    exportExoClockRequested = Signal()
    exportETDRequested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Reports & Exports", "Keep familiar HOPS outputs and prepare submission-ready transit data."
            )
        )
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(14)
        for title, detail, button_text, icon_name, signal, tooltip in (
            (
                "Project outputs",
                "Open the portable outputs folder containing reduction metadata, light curves, fit summaries, and plots.",
                "Open outputs folder",
                "fa6s.folder-open",
                self.openFolderRequested,
                "Open the generated project output folder in the system file manager.",
            ),
            (
                "ExoClock export",
                "Create a normalized time, flux, and uncertainty table alongside a metadata summary for ExoClock review.",
                "Export ExoClock",
                "fa6s.file-export",
                self.exportExoClockRequested,
                "Write a fresh ExoClock-compatible export without modifying the successful photometry result.",
            ),
            (
                "ETD export",
                "Create a Julian-date, differential-magnitude, and uncertainty table for the Exoplanet Transit Database workflow.",
                "Export ETD",
                "fa6s.file-export",
                self.exportETDRequested,
                "Write an ETD-compatible magnitude export from the latest successful light curve.",
            ),
        ):
            card = QFrame()
            card.setObjectName("card")
            row = QHBoxLayout(card)
            row.setContentsMargins(18, 16, 18, 16)
            labels = QVBoxLayout()
            heading = QLabel(title)
            heading.setObjectName("sectionTitle")
            description = QLabel(detail)
            description.setWordWrap(True)
            description.setObjectName("muted")
            labels.addWidget(heading)
            labels.addWidget(description)
            row.addLayout(labels, 1)
            action = ActionButton(button_text, icon_name, tooltip=tooltip)
            action.clicked.connect(signal)
            row.addWidget(action)
            layout.addWidget(card)
        layout.addStretch()
        outer.addWidget(body, 1)


class SimpleToolPage(QWidget):
    def __init__(self, title: str, subtitle: str, icon_name: str, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(PageHeader(title, subtitle))
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(28, 28, 28, 28)
        self.graphic = QLabel()
        self.graphic.setPixmap(icon(icon_name, COLORS["cyan"]).pixmap(42, 42))
        self.graphic.setAlignment(Qt.AlignmentFlag.AlignCenter)
        heading = QLabel(title)
        heading.setObjectName("pageTitle")
        detail = QLabel(subtitle)
        detail.setWordWrap(True)
        detail.setObjectName("muted")
        layout.addWidget(self.graphic)
        layout.addWidget(heading)
        layout.addWidget(detail)
        layout.addStretch()
        shell = QVBoxLayout()
        shell.setContentsMargins(26, 24, 26, 28)
        shell.addWidget(card)
        outer.addLayout(shell, 1)

    def set_image(self, path: Path) -> None:
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return
        self.graphic.setPixmap(
            pixmap.scaled(
                980,
                620,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ObservingPlannerPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(
            PageHeader(
                "Observing Planner", "Check whether a target transit is practical from your observing site."
            )
        )
        body = QWidget()
        layout = QHBoxLayout(body)
        layout.setContentsMargins(26, 24, 26, 28)
        layout.setSpacing(18)
        form_card = QFrame()
        form_card.setObjectName("card")
        form = QFormLayout(form_card)
        form.setContentsMargins(18, 18, 18, 18)
        form.setSpacing(11)
        self.planner_ra = QLineEdit()
        self.planner_ra.setPlaceholderText("19:34:55.87")
        self.planner_dec = QLineEdit()
        self.planner_dec.setPlaceholderText("+36:48:55.79")
        self.planner_date = QLineEdit()
        self.planner_date.setPlaceholderText("2026-07-12 22:00")
        self.planner_lat = QDoubleSpinBox()
        self.planner_lat.setRange(-90, 90)
        self.planner_lat.setDecimals(5)
        self.planner_lat.setValue(42.33)
        self.planner_lon = QDoubleSpinBox()
        self.planner_lon.setRange(-180, 180)
        self.planner_lon.setDecimals(5)
        self.planner_lon.setValue(-83.05)
        form.addRow(
            LabelWithInfo("Right ascension", "ICRS target right ascension in hh:mm:ss."), self.planner_ra
        )
        form.addRow(
            LabelWithInfo("Declination", "ICRS target declination in signed dd:mm:ss."), self.planner_dec
        )
        form.addRow(
            LabelWithInfo("Local date & time", "Beginning of the observing window in local civil time."),
            self.planner_date,
        )
        form.addRow(LabelWithInfo("Latitude", "Observatory latitude; north is positive."), self.planner_lat)
        form.addRow(LabelWithInfo("Longitude", "Observatory longitude; east is positive."), self.planner_lon)
        calculate = ActionButton(
            "Evaluate observing window",
            "fa6s.moon",
            primary=True,
            tooltip="Calculate target altitude and airmass across a six-hour observing window.",
        )
        calculate.clicked.connect(self._calculate)
        form.addRow(calculate)
        layout.addWidget(form_card, 2)
        result_card = QFrame()
        result_card.setObjectName("card")
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(18, 18, 18, 18)
        title = QLabel("Visibility summary")
        title.setObjectName("sectionTitle")
        self.planner_result = QLabel("Enter a target and site to evaluate its observing window.")
        self.planner_result.setObjectName("muted")
        self.planner_result.setWordWrap(True)
        result_layout.addWidget(title)
        result_layout.addWidget(self.planner_result)
        result_layout.addStretch()
        layout.addWidget(result_card, 3)
        outer.addWidget(body, 1)

    def _calculate(self) -> None:
        try:
            import astropy.units as units
            from astropy.coordinates import AltAz, EarthLocation, SkyCoord
            from astropy.time import Time

            target = SkyCoord(
                self.planner_ra.text(), self.planner_dec.text(), unit=(units.hourangle, units.deg)
            )
            location = EarthLocation(
                lat=self.planner_lat.value() * units.deg, lon=self.planner_lon.value() * units.deg
            )
            start = Time(self.planner_date.text().strip().replace(" ", "T"), format="isot")
            times = start + [0, 1, 2, 3, 4, 5, 6] * units.hour
            altitudes = target.transform_to(AltAz(obstime=times, location=location)).alt.deg
            peak = float(max(altitudes))
            hours = sum(value >= 30 for value in altitudes)
            self.planner_result.setText(
                f"Peak altitude: {peak:.1f}°\nHours above 30° in this window: about {hours}\n"
                f"Recommendation: {'Good observing geometry.' if peak >= 45 and hours >= 3 else 'Review timing or choose another night.'}"
            )
            self.planner_result.setStyleSheet(
                f"color: {COLORS['green'] if peak >= 45 and hours >= 3 else COLORS['amber']};"
            )
        except Exception as exc:
            self.planner_result.setText(f"The observing window could not be evaluated: {exc}")
            self.planner_result.setStyleSheet(f"color: {COLORS['amber']};")
