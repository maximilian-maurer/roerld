from typing import Tuple, List, Any

import tensorflow as tf

from roerld.models.keras_model import KerasModel
from roerld.models.tf_net_from_config import _create_net
from roerld.models.datatype_conversion import datatype_to_tf_datatype


class MLPModel(KerasModel):
    def __init__(self,
                 input_spec,
                 config_section,
                 **kwargs):
        """A MLP model.

        All inputs from input_keys are concatenated (in the order given in that list) and then fed into a MLP"""
        self.config_section = config_section
        super().__init__(input_spec, **kwargs)


    def _create_network(self, input_spec) -> Tuple[tf.keras.Model, List[Any]]:
        input_key_order = self.config_section.key("input_key_order")
        assert input_key_order is not None

        inputs = []
        for input_key in input_key_order:
            dtype = None
            if input_spec[input_key][1] is not None:
                dtype = datatype_to_tf_datatype(input_spec[input_key][1])

            inputs.append(tf.keras.layers.Input(shape=input_spec[input_key][0],
                                                dtype=dtype))

        x = None
        if len(input_key_order) > 1:
            x = tf.keras.layers.Concatenate(axis=-1)(inputs)
        else:
            x = inputs[0]
        x = _create_net(self.config_section.key("layers"), x)
        x = tf.squeeze(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        return model, input_key_order
