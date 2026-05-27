
import time


class Timer:
    """Timer recording both wall-clock time and real CPU process time."""

    def __init__(self):
        self._start_wall = 0.0
        self._start_cpu = 0.0
        self._elapsed_wall = 0.0
        self._elapsed_cpu = 0.0

    def start(self):
        self._start_wall = time.time()
        self._start_cpu = time.process_time()

    def stop(self) -> tuple[float, float]:
        """Stop timer and return (elapsed_wall, elapsed_cpu)."""
        self._elapsed_wall = time.time() - self._start_wall
        self._elapsed_cpu = time.process_time() - self._start_cpu
        return self._elapsed_wall, self._elapsed_cpu

    @property
    def elapsed_wall(self) -> float:
        return self._elapsed_wall

    @property
    def elapsed_cpu(self) -> float:
        return self._elapsed_cpu
