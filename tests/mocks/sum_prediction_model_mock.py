from roerld.models.model import Model

import numpy as np


class SumPredictionModelMock(Model):
    def __init__(self, observation_key, action_key):
        self.observation_key = observation_key
        self.action_key = action_key

    def predict(self, input_dict, training=False):
        targets = np.sum(input_dict[self.observation_key], axis=1)
        if input_dict[self.action_key].shape[-1] == 1:
            actions = input_dict[self.action_key].flatten()
        else:
            actions = np.sum(input_dict[self.action_key], axis=1)

        return -(actions-targets)**2+1

    def tensor_predict(self, input_dict, training):
        raise NotImplementedError()

    def get_trainable_vars(self):
        raise NotImplementedError()

    def get_regularization_losses(self):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def set_weights(self, weights):
        raise NotImplementedError()

    def save(self, path):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def print_summary(self):
        raise NotImplementedError()
