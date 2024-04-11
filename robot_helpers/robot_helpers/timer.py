# Adapted from https://realpython.com/python-timer/

from contextlib import ContextDecorator
import time


class Timer(ContextDecorator):
    timers = {}

    def __init__(
        self,
        name=None,
        logger=None,
    ):
        self._tic = None
        self.name = name
        self.logger = logger

        if name:
            self.timers.setdefault(name, 0)

    def start(self):
        if self._tic is not None:
            raise RuntimeError("Timer is already running.")

        self._tic = time.perf_counter()

    def stop(self):
        if self._tic is None:
            raise RuntimeError("Timer is not running.")

        elapsed_time = time.perf_counter() - self._tic
        self._tic = None

        if self.logger:
            self.logger(f"Elapsed time: {elapsed_time:0.4f} seconds")

        if self.name:
            self.timers[self.name] += elapsed_time

        return elapsed_time

    @classmethod
    def reset(cls):
        cls.timers = {}

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc_info):
        self.stop()
