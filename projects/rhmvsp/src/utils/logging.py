
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

class Tee:
    """Redirect stdout to both console and a log file simultaneously.

    Usage::

        tee = Tee("experiment.log")
        sys.stdout = tee
        print("This goes to console AND file")
        tee.close()        # restores original stdout
    """
    def __init__(self, filepath: str):
        self._file = open(filepath, "w", encoding="utf-8")
        self._stdout = sys.stdout

    def write(self, data: str):
        self._stdout.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()
        sys.stdout = self._stdout


class _NullLogger:
    """Logger that discards all messages."""
    def debug(self, msg): pass
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
