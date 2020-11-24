from roerld.config.experiment_config import ExperimentConfigView, ExperimentConfigError
from roerld.utils.onpolicy_fraction_strategy.linear_rampup import LinearRampup


def resolve_linear_rampup(section: ExperimentConfigView):
    return LinearRampup(
        start_epoch=section.key("start_epoch"),
        linear_rampup_per_epoch=section.key("increase_per_epoch"),
        fraction_max=section.key("max"),
        fraction_min=section.optional_key("min", 0.)
    )


def resolve_onpolicy_fraction_strategy(config_section: ExperimentConfigView):
    kinds = {
        "linear_rampup": resolve_linear_rampup
    }

    kind = config_section.key("name")

    if kind not in kinds:
        raise ExperimentConfigError(f"{kind} is not a supported replay buffer type. "
                                    f"Supported are f{list(kinds.keys())}")

    return kinds[kind](config_section)
