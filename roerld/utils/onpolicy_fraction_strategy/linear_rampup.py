from roerld.utils.onpolicy_fraction_strategy.onpolicy_fraction_strategy import OnPolicyFractionStrategy


class LinearRampup(OnPolicyFractionStrategy):
    def __init__(self, start_epoch, linear_rampup_per_epoch, fraction_max,
                 fraction_min):
        self.start_epoch = start_epoch
        self.linear_rampup_per_epoch = linear_rampup_per_epoch
        self.max = fraction_max
        self.min = fraction_min

        assert linear_rampup_per_epoch >= 0
        assert self.max <= 1

    def process_epoch_experience(self, epoch, eval_rollout_experience):
        pass

    def onpolicy_fraction(self, epoch):
        return min(max(self.min, self.linear_rampup_per_epoch * (epoch - self.start_epoch)), self.max)
