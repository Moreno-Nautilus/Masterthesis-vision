"""Single-stream stdout logger used in place of the ROS node logger.

Extracted from run_pipeline_track_multicam_realsense.py.
"""
from __future__ import annotations

import time


class _UnifiedLogger:
    """Single-stream replacement for the ROS node logger.

    The ROS get_logger() writes to stderr while the pipeline's own diagnostics
    use print/_dprint on stdout; two independently-buffered streams interleave
    non-deterministically when redirected to one log file. Routing every
    get_logger() call through here puts ALL output on stdout with flush, so a
    redirected log is ordered and consistently prefixed.

    Level gating: warn/error/fatal always print (no longer silently dropped when
    debug is off); info/debug print only when verbose.
    """

    def __init__(self, verbose: bool) -> None:
        self._verbose = bool(verbose)

    @staticmethod
    def _emit(level: str, msg) -> None:
        t = time.time()
        ts = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int((t % 1) * 1000):03d}"
        print(f"[{ts}] [{level}] {msg}", flush=True)

    def info(self, msg="", *a, **kw):
        if self._verbose:
            self._emit("INFO", msg)

    def debug(self, msg="", *a, **kw):
        if self._verbose:
            self._emit("DEBUG", msg)

    def warn(self, msg="", *a, **kw):
        self._emit("WARN", msg)

    def warning(self, msg="", *a, **kw):
        self._emit("WARN", msg)

    def error(self, msg="", *a, **kw):
        self._emit("ERROR", msg)

    def fatal(self, msg="", *a, **kw):
        self._emit("FATAL", msg)
