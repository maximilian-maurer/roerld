from typing import Dict, Tuple

import numpy as np

from roerld.replay_buffers.replay_buffer import ReplayBuffer


class RingReplayBuffer(ReplayBuffer):
    def __init__(self, fields: Dict[str, Tuple[Tuple, np.dtype]], size):
        assert size > 0

        self.buffers = {}
        for field, (shape, dtype) in fields.items():
            if isinstance(shape, type(None)):
                self.buffers[field] = np.zeros(size, dtype=dtype)
            elif isinstance(shape, int):
                self.buffers[field] = np.zeros((size, shape), dtype=dtype)
            else:
                self.buffers[field] = np.zeros((size, *shape), dtype=dtype)

        self.keys = list(fields.keys())

        self.current_index = 0
        self._size = 0
        self._max_size = size

    def store_single(self, **kwargs):
        # todo make this check optional, this has a significant overhead if this function is called with single rows
        assert all(key in kwargs for key in self.keys)

        for key in self.keys:
            self.buffers[key][self.current_index] = kwargs[key]

        self.current_index = (self.current_index + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def store_batch(self, **kwargs):
        # todo make these checks optional, while useful for debugging this has a significant overhead if this function
        #  is called with small batches
        all_in = all(key in kwargs for key in self.buffers.keys())
        if not all_in:
            print(f"Replay buffer expected data with keys {list(self.buffers.keys())}, but got {list(kwargs.keys())})")
            assert all_in
        assert len(kwargs) > 0

        new_elements = len(kwargs[next(iter(kwargs))])
        if not all(len(v) == new_elements for k, v in kwargs.items()):
            print(kwargs)
            assert False

        # todo this is inefficient, but will do for now, refactor as part of the replay buffer extension
        if self.current_index + new_elements < self._max_size:
            for key in self.keys:
                self.buffers[key][self.current_index:self.current_index+new_elements] = kwargs[key]

            self.current_index = (self.current_index + new_elements) % self._max_size
            self._size = min(self._size + new_elements, self._max_size)
        else:
            for i in range(new_elements):
                for key in self.keys:
                    self.buffers[key][self.current_index] = kwargs[key][i]

                self.current_index = (self.current_index + 1) % self._max_size
                self._size = min(self._size + 1, self._max_size)

    def sample_batch(self, batch_size: int):
        assert self._size > 0 or batch_size == 0

        indices = np.random.randint(0, self._size, size=batch_size)
        batch = dict.fromkeys(self.keys)
        for key in self.keys:
            batch[key] = self.buffers[key][indices]
        return batch

    def sample_count(self):
        return self._size

    def max_size(self):
        return self._max_size
