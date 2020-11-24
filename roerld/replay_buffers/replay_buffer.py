from typing import Dict
import numpy as np


class ReplayBuffer:
    """
    Replay buffer that stores independent transitions and supports random sampling from them. It is not required to
    retain any data about the context of the transition (e.g. the episode it belongs to or its place in it).

    A single transition is a set of key-value pairs (str, np.ndarray). A batch of transitions for the purposes of
    transaction is represented as a set of key-value paris (str, np.ndarray), where the np.ndarray has is at least
    two dimensional and the first axis is interpreted as the index of transition.

    The buffer must advertise its maximum size and its current usage. If a transition is added while the current usage
    is not at the maximum capacity this transition and all other transitions currently in the buffer must be maintained.
    If the addition of the transition will result in the maximum capacity being exceeded, the behavior may vary across
    implementations, but all must guarantee that after the operation the transition being added will be in the buffer.
    No guarantees are made about the retention of any other transition in the buffer.

    Batch version of single operations must obey these semantics as if the store operation was performed sequentially
    on the transitions in the batch.
    """

    def store_batch(self, **kwargs) -> None:
        """
        Stores the batch of transitions obeying the rules set out in the class description.

        Args:
            **kwargs: The batch of transitions as a dictionary or a dictionary expansion. Since a transition may not
                        contain nested dictionaries, if passing a single argument and that argument is a dictionary it
                        will be interpreted as the batch.
        """
        raise NotImplementedError()

    def sample_batch(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Randomly samples transitions from the buffer. Each transition has the same probability of being chosen.

        Args:
            batch_size: The number of transitions to retrieve. It must be at least 1.

        Returns:
            The requested transitions in the batch format laid out in the class description. Note that even if only
            a single transition is requested, the returned data is still in batch format.
        """
        raise NotImplementedError()

    def sample_count(self):
        """
        Returns: the current number of transitions in the buffer.
        """
        raise NotImplementedError()

    def max_size(self):
        """
        Returns: the maximum guaranteed capacity as per the class description.
        """
        raise NotImplementedError()
