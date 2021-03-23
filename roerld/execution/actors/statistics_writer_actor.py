from typing import List, Callable, Dict, Any

import ray

from roerld.statistics.statistics_writer import StatisticsWriter


@ray.remote
class StatisticsWriterActor:
    def __init__(self, statistics_writer_factory: Callable[[], StatisticsWriter]):
        self.writer = statistics_writer_factory()

    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        self.writer.write_statistics(epoch_index, source_tag, data, is_evaluation)

