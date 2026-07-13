from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import qtawesome as qta
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from leaps.models import StageID, StageState, StageStatus

from .theme import COLORS


def icon(name: str, color: str = COLORS["muted"], active: str | None = None):
    options = {"color": color}
    if active:
        options["color_active"] = active
    return qta.icon(name, **options)


class InfoButton(QToolButton):
    def __init__(self, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setIcon(icon("fa6s.circle-info", COLORS["muted"]))
        self.setIconSize(QSize(15, 15))
        self.setFixedSize(26, 26)
        self.setToolTip(tooltip)
        self.setAccessibleName("Information")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class LabelWithInfo(QWidget):
    def __init__(self, text: str, tooltip: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        layout.addWidget(QLabel(text))
        layout.addWidget(InfoButton(tooltip))
        layout.addStretch()


class ActionButton(QPushButton):
    def __init__(
        self,
        text: str,
        icon_name: str | None = None,
        *,
        primary: bool = False,
        tooltip: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(text, parent)
        if icon_name:
            self.setIcon(icon(icon_name, "white" if primary else COLORS["muted"]))
        if primary:
            self.setProperty("primary", True)
        if tooltip:
            self.setToolTip(tooltip)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class StageNavButton(QFrame):
    clicked = Signal(object)

    def __init__(self, stage: StageID, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stageNav")
        self.stage = stage
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedHeight(62)
        self.setAccessibleName(title)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 6, 12, 6)
        layout.setSpacing(13)
        self.status_icon = QLabel()
        self.status_icon.setFixedSize(26, 26)
        self.status_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_icon)
        labels = QVBoxLayout()
        labels.setSpacing(1)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-size: 14px; font-weight: 600;")
        self.summary = QLabel("Locked")
        self.summary.setObjectName("muted")
        labels.addWidget(self.title)
        labels.addWidget(self.summary)
        layout.addLayout(labels, 1)
        self.state = StageState()
        self.active = False
        self.update_state(self.state)

    def update_state(self, state: StageState) -> None:
        self.state = state
        self.summary.setText(state.summary)
        icons = {
            StageStatus.COMPLETE: ("fa6s.circle-check", COLORS["green"]),
            StageStatus.READY: ("fa6.circle", COLORS["cyan"]),
            StageStatus.RUNNING: ("fa6s.spinner", COLORS["cyan"]),
            StageStatus.NEEDS_ATTENTION: ("fa6s.triangle-exclamation", COLORS["amber"]),
            StageStatus.LOCKED: ("fa6s.lock", COLORS["muted_2"]),
        }
        name, color = icons[state.status]
        self.status_icon.setPixmap(icon(name, color).pixmap(23, 23))
        self.setEnabled(state.status != StageStatus.LOCKED)
        self._restyle()

    def set_active(self, active: bool) -> None:
        self.active = active
        self._restyle()

    def _restyle(self) -> None:
        if self.active:
            self.setStyleSheet(
                f"QFrame#stageNav {{background: #173049; border-left: 3px solid {COLORS['cyan']};}}"
                f"QLabel {{color: {COLORS['text']};}}"
            )
        elif not self.isEnabled():
            self.setStyleSheet(
                f"QFrame#stageNav {{background: transparent;}} QLabel {{color: {COLORS['muted_2']};}}"
            )
        else:
            self.setStyleSheet(
                f"QFrame#stageNav {{background: transparent; border-left: 3px solid transparent;}}"
                f"QFrame#stageNav:hover {{background: {COLORS['surface']};}}"
            )

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled() and event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.stage)
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.isEnabled() and event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return, Qt.Key.Key_Space):
            self.clicked.emit(self.stage)
            event.accept()
            return
        super().keyPressEvent(event)


class ToolNavButton(QPushButton):
    def __init__(self, text: str, icon_name: str, parent: QWidget | None = None) -> None:
        super().__init__(text, parent)
        self.setIcon(icon(icon_name, COLORS["muted"]))
        self.setStyleSheet(
            f"QPushButton {{text-align: left; background: transparent; border: 0; color: {COLORS['muted']};"
            "padding-left: 15px; font-weight: 500;}"
            f"QPushButton:hover {{background: {COLORS['surface']}; color: {COLORS['text']};}}"
        )


class PageHeader(QFrame):
    def __init__(self, title: str, subtitle: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("contentHeader")
        self.setFixedHeight(114)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(38, 24, 26, 20)
        texts = QVBoxLayout()
        texts.setSpacing(2)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("pageSubtitle")
        texts.addWidget(title_label)
        texts.addWidget(subtitle_label)
        layout.addLayout(texts)
        layout.addStretch()
        self.actions = QHBoxLayout()
        layout.addLayout(self.actions)


@dataclass(slots=True)
class StarOverlay:
    x: float
    y: float
    radius: float = 7.0


class FITSImageLabel(QLabel):
    targetPlaced = Signal(float, float)

    def __init__(self, asset: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.source = QPixmap(str(asset))
        self.setPixmap(self.source)
        self.setScaledContents(True)
        self.setMinimumSize(520, 520)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.manual_mode = False

    def set_image(self, pixmap: QPixmap) -> None:
        self.source = pixmap
        self.setPixmap(pixmap)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self.manual_mode and event.button() == Qt.MouseButton.LeftButton:
            self.targetPlaced.emit(
                event.position().x() / max(self.width(), 1), event.position().y() / max(self.height(), 1)
            )
            self.manual_mode = False
        super().mousePressEvent(event)


class FITSWorkspace(QFrame):
    targetPlaced = Signal(float, float)

    def __init__(self, asset: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setFixedHeight(54)
        toolbar.setStyleSheet(
            f"background: {COLORS['surface']}; border-bottom: 1px solid {COLORS['border_soft']};"
        )
        tools = QHBoxLayout(toolbar)
        tools.setContentsMargins(13, 8, 13, 8)
        tools.setSpacing(0)
        self.mode_buttons: list[QPushButton] = []
        for text, icon_name, tip in (
            ("Pan", "fa6s.hand", "Drag the image to inspect a different area."),
            ("Zoom", "fa6s.magnifying-glass", "Zoom into a region of the FITS image."),
            (
                "Contrast",
                "fa6s.circle-half-stroke",
                "Adjust image stretch and contrast without changing FITS pixels.",
            ),
            ("Reset", "fa6s.rotate-left", "Restore the original view and contrast."),
        ):
            button = ActionButton(text, icon_name, tooltip=tip)
            button.setFlat(True)
            if text == "Pan":
                button.setStyleSheet(f"background: {COLORS['surface_3']}; color: {COLORS['cyan']};")
            tools.addWidget(button)
            self.mode_buttons.append(button)
        tools.addStretch()
        fullscreen = QToolButton()
        fullscreen.setIcon(icon("fa6s.expand", COLORS["muted"]))
        fullscreen.setToolTip("Expand the FITS workspace to use the available window.")
        tools.addWidget(fullscreen)
        zoom = ActionButton("100%", "fa6s.chevron-down", tooltip="Select a zoom level.")
        zoom.setFixedWidth(90)
        tools.addWidget(zoom)
        layout.addWidget(toolbar)

        self.image = FITSImageLabel(asset)
        self.image.targetPlaced.connect(self.targetPlaced)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(self.image)
        layout.addWidget(scroll, 1)

        metadata = QFrame()
        metadata.setFixedHeight(47)
        metadata.setStyleSheet(
            f"background: {COLORS['surface']}; border-top: 1px solid {COLORS['border_soft']};"
        )
        meta_layout = QHBoxLayout(metadata)
        meta_layout.setContentsMargins(18, 0, 18, 0)
        self.filename = QLabel("FITS: light_2026-06-28T22-51-01.fits")
        self.dimensions = QLabel("2048 × 2048 px")
        self.bitdepth = QLabel("16-bit")
        self.scale = QLabel('Pixel scale: 1.20 "/pixel')
        for label in (self.filename, self.dimensions, self.bitdepth, self.scale):
            label.setObjectName("muted")
            meta_layout.addWidget(label)
            meta_layout.addStretch()
        layout.addWidget(metadata)

    def begin_manual_target(self) -> None:
        self.image.manual_mode = True
        self.image.setCursor(Qt.CursorShape.CrossCursor)

    def place_target_marker(self, x: float, y: float) -> None:
        pixmap = self.image.source.copy()
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        px, py = x * pixmap.width(), y * pixmap.height()
        pen = QPen(QColor(COLORS["amber"]), 3)
        painter.setPen(pen)
        painter.drawEllipse(int(px - 20), int(py - 20), 40, 40)
        painter.drawLine(int(px - 35), int(py), int(px - 12), int(py))
        painter.drawLine(int(px + 12), int(py), int(px + 35), int(py))
        painter.drawLine(int(px), int(py - 35), int(px), int(py - 12))
        painter.drawLine(int(px), int(py + 12), int(px), int(py + 35))
        painter.setFont(QFont("Inter", 14, QFont.Weight.DemiBold))
        painter.drawText(int(px + 26), int(py - 5), "Target")
        painter.end()
        self.image.set_image(pixmap)
        self.image.unsetCursor()


class EmptyState(QWidget):
    def __init__(self, title: str, message: str, icon_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addStretch()
        picture = QLabel()
        picture.setPixmap(icon(icon_name, COLORS["muted_2"]).pixmap(42, 42))
        picture.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 17px; font-weight: 600;")
        message_label = QLabel(message)
        message_label.setObjectName("muted")
        message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message_label.setWordWrap(True)
        layout.addWidget(picture)
        layout.addWidget(title_label)
        layout.addWidget(message_label)
        layout.addStretch()
