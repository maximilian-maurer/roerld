import abc


class OnPolicyFractionStrategy(abc.ABC):
    def process_epoch_experience(self, epoch, eval_rollout_experience):
        """Processes the epochs experience. This function may be called after the first request for the on-policy
        fraction of that epoch."""
        raise NotImplementedError()

    def onpolicy_fraction(self, epoch):
        raise NotImplementedError()
