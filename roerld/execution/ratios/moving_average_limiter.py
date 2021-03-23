from collections import deque
import numpy as np

from roerld.execution.ratios.limiter import SDLimiter


class MovingAverageLimiter(SDLimiter):
    def __init__(self, minimum, maximum, window_size):
        self.counts = deque([0] * (window_size - 1), window_size - 1)
        self.min = minimum
        self.max = maximum
        self.this_step_count = 0

    def next_step(self, step_index):
        self.counts.append(self.this_step_count)
        self.this_step_count = 0

    def one_step_performed(self):
        self.this_step_count += 1

    def _value(self):
        return np.mean(np.concatenate([np.asarray(list(self.counts)),
                                       np.asarray([self.this_step_count])]))

    def _hypothetical_value(self, additional_ones):
        return np.mean(np.concatenate([np.asarray(list(self.counts)),
                                       np.asarray([self.this_step_count+additional_ones])]))

    def lower_limit_achieved(self):
        return self.min is None or self._value() >= self.min

    def upper_limit_reached(self):
        return self.max is not None and self._value() >= self.max

    def would_reach_lower_limit(self, additional_ones):
        return self.min is None or self._hypothetical_value(additional_ones) >= self.min

    def would_reach_upper_limit(self, additional_ones):
        return self.max is not None and self._hypothetical_value(additional_ones) >= self.max

    def is_upper_limit(self):
        return self.max is not None

    def is_lower_limit(self):
        return self.min is not None
