from typing import Dict, Any
import tensorflow as tf
import numpy as np

from roerld.statistics.statistics_writer import StatisticsWriter
from roerld.statistics.summaries import normalize_summary_format, aggregates_of_summary


class TensorboardStatisticsWriter(StatisticsWriter):
    def __init__(self, path):
        self.tb_writer = tf.summary.create_file_writer(logdir=path)

    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        normalized = aggregates_of_summary(normalize_summary_format(data), include_nonaggregatable=True)

        with self.tb_writer.as_default():
            for k, v in normalized.items():
                if np.isscalar(v) and not isinstance(v, str):
                    tf.summary.scalar(f"{source_tag} {k}" if len(source_tag.strip()) != 0 else k, v, step=epoch_index)



