
class SDLimiter:
    """
    Step driven limiter governing a process where units of work are performed, and we want to impose lower and upper
    limits on how many of them can be performed with reference to a given global step.
    """
    def next_step(self, step_index):
        """
        Notifies the limiter that the next step has happened.
        """
        raise NotImplementedError()

    def one_step_performed(self):
        """
        Notifies the limiter that one additional unit of work has been performed.
        """
        raise NotImplementedError()

    def lower_limit_achieved(self):
        """
        Has the lower limit for this quantity been reached?
        """
        raise NotImplementedError()

    def upper_limit_reached(self):
        """
        Has the upper limit for this quantity been reached?
        """
        raise NotImplementedError()

    def would_reach_lower_limit(self, additional_ones):
        raise NotImplementedError()

    def would_reach_upper_limit(self, additional_ones):
        raise NotImplementedError()

    def is_upper_limit(self):
        """
        Does this express an upper limit. If False, then upper_limit_reached may never return True.
        """
        raise NotImplementedError()

    def is_lower_limit(self):
        """
        Does this express a lower limit. If False, then lower_limit_reached must always return True.
        """
        raise NotImplementedError()

