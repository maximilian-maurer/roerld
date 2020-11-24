from typing import Tuple, List, Any


class SubsamplingEnv:
    """
    In problems where there are multiple possible control frequencies that are all multiples of a base frequency,
    e.g. the highest possible one, or the simulation interval, this environment style allows to always record the
    experience at the highest frequency independent of the control interval for the reinforcement learning, such that
    recorded experience can later be re-used. This also enables reward-rewriting for environments where the
    time-resolution of the reward calculation is higher than the interaction abstraction provided by the environment.
    """
    def subsample_step(self, action) -> List[Tuple[Any, float, bool, Any]]:
        """
        An extended version of the gym step (that replaces it).
        """
        raise NotImplementedError()

    def step(self, action):
        """The original step operation becomes a no-op with this interface."""
        raise ValueError("The subsampling environment does not support a step.")

    def max_samples_per_step(self) -> int:
        raise NotImplementedError()