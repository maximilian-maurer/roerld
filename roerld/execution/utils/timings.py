import time


class TimingHelper:
    def __init__(self, section_name):
        self._last_timer = time.perf_counter()
        self._results = {}
        self._last_section_name = section_name

    def time_stamp(self, section_name=None):
        self._results[self._last_section_name] = time.perf_counter() - self._last_timer
        self._last_section_name = section_name
        self._last_timer = time.perf_counter()

    def result(self):
        return self._results
