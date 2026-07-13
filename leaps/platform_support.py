from __future__ import annotations

import ctypes
import platform
import subprocess


class SleepInhibitor:
    def __init__(self, reason: str = "LEAPS is processing") -> None:
        self.reason = reason
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if platform.system() == "Darwin" and self._process is None:
            self._process = subprocess.Popen(["caffeinate", "-dimsu"])
        elif platform.system() == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            self._process = None
        elif platform.system() == "Windows":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

    def __enter__(self) -> SleepInhibitor:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
