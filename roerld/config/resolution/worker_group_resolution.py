from roerld.config.experiment_config import ExperimentConfigView, ExperimentConfigError
import copy


def resolve_worker_groups(section: ExperimentConfigView):
    if section.key("name") != "worker_groups":
        raise ExperimentConfigError(f"Expected Worker Group Configuration in {section.path}")

    workers = []
    for index, group in enumerate(section.key("workers")):
        if "count" not in group or "ray_kwargs" not in group or "worker_kwargs" not in group:
            raise ExperimentConfigError(f"Worker Groups {section.path} requires at least the keys name, count, "
                                        f"ray_kwargs and worker_kwargs for each worker group.")
        if "num_gpus" not in group["ray_kwargs"] or "num_cpus" not in group["ray_kwargs"]:
            raise ExperimentConfigError(f"Worker Groups {section.path} worker group {index} is missing"
                                        f"one of the required keys: ray_kwargs.num_gpus, ray_kwargs.num_cpus")

        assert group["name"] == "worker_group"
        count = group["count"]
        assert count > 0
        assert group["ray_kwargs"]["num_cpus"] > 0

        for i in range(count):
            workers.append([copy.deepcopy(group["ray_kwargs"]),
                            copy.deepcopy(group["worker_kwargs"])])

    return workers
