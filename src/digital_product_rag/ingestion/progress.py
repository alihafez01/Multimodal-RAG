"""Small dependency-free progress and timing helpers for ingestion CLIs."""

from __future__ import annotations

import math
import sys
import threading
import time
from datetime import datetime
from typing import TextIO


def format_duration(seconds: float | None) -> str:
    """Format seconds as HH:MM:SS, or a placeholder when no estimate exists."""
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Render a terminal progress bar with elapsed time and an ETA estimate.

    In interactive terminals the bar refreshes once per second, including while a
    slow OCR or model request is running. When output is redirected, it emits
    periodic newline updates instead of carriage-return redraws.
    """

    def __init__(
        self,
        label: str,
        total: int,
        *,
        enabled: bool = True,
        stream: TextIO | None = None,
        heartbeat_seconds: float = 1.0,
    ) -> None:
        self.label = label
        self.total = max(0, total)
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self.heartbeat_seconds = max(0.2, heartbeat_seconds)
        self.is_terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self.current = 0
        self.detail = ""
        self.started_at = time.perf_counter()
        self._last_rendered = -1
        self._last_render_at = self.started_at
        self._last_line_length = 0
        self._line_active = False
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._redirect_interval = max(1, math.ceil(self.total / 20))

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started_at

    def log(self, message: str, *, level: str = "INFO") -> None:
        """Print a timestamped message, safely ending any live progress line."""
        with self._lock:
            if self._line_active:
                self.stream.write("\n")
                self._line_active = False
                self._last_line_length = 0
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {level} {message}", file=self.stream, flush=True)

    def start(self, message: str | None = None) -> None:
        if message:
            self.log(message)
        if not self.enabled:
            return
        with self._lock:
            self._render(force=True)
        if self.total > 0:
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"progress-{self.label[:24]}",
                daemon=True,
            )
            self._thread.start()

    def set_detail(self, detail: str, *, refresh: bool = False) -> None:
        with self._lock:
            self.detail = detail
            if refresh:
                self._render(force=True)

    def update(self, current: int, detail: str = "", *, force: bool = False) -> None:
        with self._lock:
            self.current = min(self.total, max(0, current))
            if detail:
                self.detail = detail
            self._render(force=force)

    def advance(self, detail: str = "") -> None:
        self.update(self.current + 1, detail)

    def finish(self, message: str | None = None) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=self.heartbeat_seconds + 0.2)
        with self._lock:
            if self.enabled:
                self._render(force=True)
                if self._line_active:
                    self.stream.write("\n")
                    self.stream.flush()
                    self._line_active = False
                    self._last_line_length = 0
            completion = message or self.label
            state = "complete" if self.current >= self.total else "stopped"
            self.log(
                f"{completion}: {self.current}/{self.total} items, "
                f"elapsed {format_duration(self.elapsed)} ({state})"
            )

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_seconds):
            with self._lock:
                if self.current < self.total:
                    self._render(force=True, heartbeat=True)

    def _render(self, *, force: bool, heartbeat: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if not force and self.total > 0:
            if self.is_terminal:
                if now - self._last_render_at < 0.15 and self.current == self._last_rendered:
                    return
            elif self.current < self.total:
                if self.current - max(0, self._last_rendered) < self._redirect_interval:
                    return

        if heartbeat and not self.is_terminal and now - self._last_render_at < 5.0:
            return

        done = self.current
        fraction = 1.0 if self.total == 0 else done / self.total
        percent = 100.0 if self.total == 0 else fraction * 100
        filled = int(round(20 * fraction))
        bar = "#" * filled + "-" * (20 - filled)
        elapsed = self.elapsed
        eta = elapsed * (self.total - done) / done if done > 0 and self.total else None
        detail = self.detail.replace("\r", " ").replace("\n", " ")[:54]
        suffix = f" | {detail}" if detail else ""
        line = (
            f"{self.label} [{bar}] {done}/{self.total} {percent:5.1f}%"
            f" | elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}{suffix}"
        )
        if self.is_terminal:
            padding = " " * max(0, self._last_line_length - len(line))
            self.stream.write("\r" + line + padding)
            self.stream.flush()
            self._line_active = True
            self._last_line_length = len(line)
        else:
            print(line, file=self.stream, flush=True)
            self._line_active = False
            self._last_line_length = 0
        self._last_rendered = done
        self._last_render_at = now
