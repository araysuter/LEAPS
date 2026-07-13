from __future__ import annotations

import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from leaps.models import StageEvent
from leaps.platform_support import SleepInhibitor
from leaps.science import CancellationToken


class WorkerSignals(QObject):
    event = Signal(object)
    result = Signal(object)
    error = Signal(object)
    finished = Signal()


class StageWorker(QRunnable):
    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.token = CancellationToken()
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            self.kwargs.setdefault("token", self.token)
            self.kwargs.setdefault("emit", self.signals.event.emit)
            with SleepInhibitor("LEAPS is processing an observing run"):
                result = self.function(*self.args, **self.kwargs)
            self.signals.result.emit(result)
        except BaseException as exc:
            if not getattr(exc, "technical_details", None):
                try:
                    exc.technical_details = "".join(traceback.format_exception(exc))
                except Exception:
                    pass
            self.signals.error.emit(exc)
        finally:
            self.signals.finished.emit()


class TaskRunner(QObject):
    busyChanged = Signal(bool)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.current: StageWorker | None = None

    def start(
        self,
        function: Callable[..., Any],
        *args: Any,
        event: Callable[[StageEvent], None] | None = None,
        result: Callable[[Any], None] | None = None,
        error: Callable[[BaseException], None] | None = None,
        finished: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> StageWorker:
        if self.current is not None:
            raise RuntimeError("A task is already running")
        worker = StageWorker(function, *args, **kwargs)
        self.current = worker
        if event:
            worker.signals.event.connect(event)
        if result:
            worker.signals.result.connect(result)
        if error:
            worker.signals.error.connect(error)
        if finished:
            worker.signals.finished.connect(finished)
        worker.signals.finished.connect(self._finish)
        self.busyChanged.emit(True)
        self.pool.start(worker)
        return worker

    @Slot()
    def cancel(self) -> None:
        if self.current is not None:
            self.current.token.cancel()

    @Slot()
    def _finish(self) -> None:
        self.current = None
        self.busyChanged.emit(False)
