from abc import ABC, abstractmethod
from typing import Tuple, List, Any

import tensorflow as tf

from roerld.models.model import Model


class KerasModel(Model, ABC):
    """Helper base that provides the implementation for the roerld.models.Model class for deriving classes
    that only differ in how they set up a tf keras model.
    """

    def __init__(self, input_spec, **kwargs):
        self.input_spec = input_spec
        self.kwargs = kwargs
        self.model, self.input_order = self._create_network(input_spec)
        assert all([key in input_spec for key in self.input_order])

    @abstractmethod
    def _create_network(self, input_spec) -> Tuple[tf.keras.Model, List[Any]]:
        """Creates the network in a repeatable fashion (multiple sequential calls to this function must return networks
        that are equal (and whose weights can be applied to each other) and independent of each other.

        :param input_spec: A dictionary (key: Tuple(shape, dtype)) of all the inputs which are available to the model.

        :return: A tuple (model, input_order).
        The input_order is a list of keys from a subset of input_spec that specifies in which
        order the data has to be given to the predict call of the model.
        """
        raise NotImplementedError()

    def _order_input(self, input_dict):
        return [input_dict[key] for key in self.input_order]

    @tf.function
    def _predict(self, input_dict, training=False):
        return self.model(input_dict, training=training)

    def predict(self, input_dict, training=False):
        return self._predict(self._order_input(input_dict),
                             training=tf.constant(training)).numpy()

    def tensor_predict(self, input_dict, training):
        return self.model(self._order_input(input_dict), training=training)

    def get_trainable_vars(self):
        return self.model.trainable_variables

    def get_regularization_losses(self):
        return self.model.losses

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)

    def print_summary(self):
        self.model.summary()
