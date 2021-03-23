import abc
from typing import Dict, Any


class StatisticsWriter(abc.ABC):
    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        raise NotImplementedError()
