from typing import List, Any

import ray



class ResultReorderBuffer:
    """
    Given a set of futures each corresponding to a result belonging to a monotonically increasing counter, buffers
    and reorders these futures so that they fulfill the following properties when retrieved via (has_result, pop_result)
    * Reordered results will have an equal or increasing counter value
    * After the first result associated with value counter+1 has been returned, all results associated with counter
      have been output already.
    """

    def __init__(self):
        self.pending_futures_to_epoch = {}
        self.pending_result_buffer = []

        self.available_results = []

    def associate_future_with_epoch(self, future: Any, epoch):
        self.pending_futures_to_epoch[future] = epoch

    def associate_futures_with_epoch(self, futures: List[Any], epoch):
        for future in futures:
            self.pending_futures_to_epoch[future] = epoch

    def epoch(self, future):
        return self.pending_futures_to_epoch[future]

    def receive_future(self, future):
        belongs_to_epoch = self.epoch(future)
        min_epoch_that_is_missing = min(self.pending_futures_to_epoch.values())

        if belongs_to_epoch > min_epoch_that_is_missing:
            self.pending_result_buffer.append(future)
        else:
            self.pending_result_buffer.append(future)
            
        epoch = min_epoch_that_is_missing
        original_buffer_length = len(self.pending_result_buffer)
        for _ in range(original_buffer_length):
            futures_for_this_epoch = [f for f in self.pending_result_buffer
                                      if self.pending_futures_to_epoch[f] == epoch]
            if len(futures_for_this_epoch) == 0:
                break

            for f in futures_for_this_epoch:
                self.pending_result_buffer.remove(f)
                del self.pending_futures_to_epoch[f]

                self.available_results.append(ray.get(f))

            if len(self.pending_futures_to_epoch) <= 0:
                break
            epoch = min(self.pending_futures_to_epoch.values())

    def has_result(self):
        return len(self.available_results) > 0

    def pop_result(self):
        # return the results oldest-first.
        return self.available_results.pop(0)

    def pending_futures(self):
        return list(self.pending_futures_to_epoch.keys())

    def clear(self):
        self.available_results.clear()
        self.pending_futures_to_epoch.clear()
        self.pending_result_buffer.clear()
