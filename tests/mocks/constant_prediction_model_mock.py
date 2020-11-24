from roerld.models.model import Model

import numpy as np


class ConstantPredictionModelMock(Model):
    def __init__(self, prediction_value: float):
        self.prediction_value = prediction_value

    def predict(self, input_dict, training=False):
        first_key = list(input_dict.keys())[0]
        return self.prediction_value * np.ones(shape=(len(input_dict[first_key])))

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
