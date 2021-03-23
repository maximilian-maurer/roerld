from typing import Dict, Any, Tuple, Iterable, List

from roerld.statistics.statistics_writer import StatisticsWriter


class WrapperWriter(StatisticsWriter):
    def __init__(self, wrapped: StatisticsWriter):
        self.wrapped = wrapped

    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        new_source, new_data = self.filter(source_tag, data)
        self.wrapped.write_statistics(epoch_index, new_source, new_data, is_evaluation)

    def filter(self, source_tag, data) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError()


class FilterWriter(WrapperWriter):
    def __init__(self, wrapped: StatisticsWriter, keys_to_keep: Iterable[str], invert=False):
        super().__init__(wrapped)
        self.invert = invert
        self.keys_to_keep = list(keys_to_keep)

    def filter(self, source_tag, data) -> Tuple[str, Dict[str, Any]]:
        results = {}
        for key, value in data.items():
            if (key in self.keys_to_keep and not self.invert) \
                    or (key not in self.keys_to_keep and self.invert):
                results[key] = value
        return source_tag, results


class CompoundWriter(StatisticsWriter):
    def __init__(self, writers: List[StatisticsWriter]):
        self.writers = writers

    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        for w in self.writers:
            w.write_statistics(epoch_index, source_tag, data, is_evaluation)


class BackwardsCompatibilityRenamer(StatisticsWriter):
    """
    For backwards compatibility with earlier versions, this renames keys so that they show up in the same tensorboard
    graphs as before.
    """

    def __init__(self, wrapped: StatisticsWriter):
        self.wrapped = wrapped

    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        if source_tag == "base":
            source_tag = ""
        if source_tag == "evaluation":
            source_tag = "Evaluation"

        data = {
            k.replace("Data_mean", "Mean").replace("Data_std", "STDev"): v
            for k, v in data.items()
        }

        self.wrapped.write_statistics(epoch_index, source_tag, data, is_evaluation)
