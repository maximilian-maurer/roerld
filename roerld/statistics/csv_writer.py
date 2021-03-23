from typing import Dict, Any
import os
import numpy as np
import time

from roerld.statistics.statistics_writer import StatisticsWriter
from roerld.statistics.summaries import aggregates_of_summary, normalize_summary_format


class CSVStatisticsWriter(StatisticsWriter):
    def __init__(self, base_path):
        self.summary_file = open(os.path.join(base_path, "progress_summary.csv"), "w")
        self.full_file = open(os.path.join(base_path, "progress.csv"), "w")

        # since we support both smaller use-cases and larger deployments, there are some edge cases where
        #  the default flushing behavior can lead to excessive wait times until the evaluations appear in
        #  the summary file, which is inconvenient for live updates that watch this file.
        self.flushing_interval = 10
        self.last_update = time.monotonic()

    def _write_evaluation(self, epoch_index: int, source_tag: str, data: Dict[str, Any]):
        normalized = aggregates_of_summary(normalize_summary_format(data), include_nonaggregatable=True)
        for k, v in normalized.items():
            if isinstance(v, np.ndarray):
                v = "[" + ",".join([str(i) for i in v]) + "]"

            self.summary_file.write(f"{epoch_index},{source_tag},{k},{v}\n")

        new_time = time.monotonic()
        if new_time - self.last_update > self.flushing_interval:
            self.summary_file.flush()
            self.last_update = new_time

    def write_statistics(self, epoch_index: int, source_tag: str, data: Dict[str, Any], is_evaluation: bool):
        if is_evaluation:
            self._write_evaluation(epoch_index, source_tag, data)

        normalized = normalize_summary_format(data)
        for k, v in normalized.items():
            if isinstance(v, np.ndarray):
                v = "[" + ",".join([str(i) for i in v]) + "]"

            self.full_file.write(f"{epoch_index},{source_tag},{k},{v}\n")
