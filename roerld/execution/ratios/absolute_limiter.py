from roerld.execution.ratios.limiter import SDLimiter


class AbsoluteLimiter(SDLimiter):
    def __init__(self, minimum, maximum):
        self.count = 0
        self.min = minimum
        self.max = maximum

    def next_step(self, step_index):
        self.count = 0

    def one_step_performed(self):
        self.count += 1

    def lower_limit_achieved(self):
        return self.min is None or self.count >= self.min

    def upper_limit_reached(self):
        return self.max is not None and self.count >= self.max

    def would_reach_lower_limit(self, additional_ones):
        return self.min is None or self.count + additional_ones >= self.min

    def would_reach_upper_limit(self, additional_ones):
        return self.max is not None and self.count + additional_ones >= self.max

    def is_upper_limit(self):
        return self.max is not None

    def is_lower_limit(self):
        return self.min is not None


