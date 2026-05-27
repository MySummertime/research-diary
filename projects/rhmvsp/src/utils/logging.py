
from __future__ import annotations
import sys


class Logger:
    """Simple logger with file and console output."""

    def __init__(self, level: str = "INFO", log_file: str | None = None):
        self.level = getattr(__import__("logging"), level.upper(), 20)
        self.log_file = log_file
        self._file = None
        if log_file:
            self._file = open(log_file, "w")

    def _log(self, level_name: str, msg: str):
        line = f"[{level_name}] {msg}"
        print(line, file=sys.stdout)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()

    def debug(self, msg):
        if self.level <= 10:
            self._log("DEBUG", msg)

    def info(self, msg):
        if self.level <= 20:
            self._log("INFO", msg)

    def warning(self, msg):
        if self.level <= 30:
            self._log("WARNING", msg)

    def error(self, msg):
        if self.level <= 40:
            self._log("ERROR", msg)

    def close(self):
        if self._file:
            self._file.close()

class _NullLogger:
    """Logger that discards all messages."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
