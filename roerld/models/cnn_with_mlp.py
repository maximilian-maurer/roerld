from typing import Tuple, List, Any

import tensorflow as tf
from roerld.config.experiment_config import ExperimentConfigView, ExperimentConfigError
from roerld.models.keras_model import KerasModel
from roerld.models.tf_net_from_config import _create_net
from roerld.models.datatype_conversion import datatype_to_tf_datatype

import numpy as np


def resolve_activation(activation_name):
    supported_activations = {
        "none": None,
        "relu": tf.keras.activations.relu
    }
    if activation_name not in supported_activations:
        raise ExperimentConfigError(f"Activation type {activation_name} unknown.")

    return supported_activations[activation_name]


class CNNWithMLP(KerasModel):
    def __init__(self,
                 config: ExperimentConfigView,
                 input_spec,
                 input_key_order=None, **kwargs):
        self.input_key_order = input_key_order
        assert self.input_key_order is not None

        self.config = config

        super().__init__(input_spec, **kwargs)

    def _create_network(self, input_spec) -> Tuple[tf.keras.Model, List[Any]]:
        layers_image_path = self.config.key("layers_image_path")
        layers_flat_obs_path = self.config.key("layers_flat_obs_path")
        layers_combined_path = self.config.key("layers_combined_path")

        image_input_key_name = self.config.key("image_input")
        flat_obs_input_key_names = self.config.key("flat_obs_inputs")

        assert image_input_key_name in self.input_key_order
        for name in flat_obs_input_key_names:
            assert name in self.input_key_order

        inputs = {}
        for input_key in self.input_key_order:
            dtype = None
            if input_spec[input_key][1] is not None:
                dtype = datatype_to_tf_datatype(input_spec[input_key][1])

            inputs[input_key] = tf.keras.layers.Input(
                shape=input_spec[input_key][0],
                dtype=dtype,
                name=input_key)

        # if image has uint8 dtype it needs to be converted
        image_path_input = None
        if input_spec[image_input_key_name][1] == np.uint8:
            image_path_input = tf.image.convert_image_dtype(inputs[image_input_key_name], tf.float32)
        else:
            image_path_input = inputs[image_input_key_name]

        image_path = _create_net(layers_image_path, image_path_input)
        flat_path = None

        if len(flat_obs_input_key_names) == 1:
            flat_path = _create_net(layers_flat_obs_path, inputs[flat_obs_input_key_names[0]])
        elif len(flat_obs_input_key_names) > 1:
            # concatenate
            inputs_to_concat = [inputs[key] for key in flat_obs_input_key_names]
            concat_out = tf.keras.layers.Concatenate(axis=-1)(inputs_to_concat)
            flat_path = _create_net(layers_flat_obs_path, concat_out)
        else:
            assert len(layers_flat_obs_path) == 0

        if flat_path is not None:
            output_path_input = image_path + flat_path
        else:
            output_path_input = image_path
        output_path = _create_net(layers_combined_path, output_path_input)

        x = tf.squeeze(output_path)
        model = tf.keras.Model(inputs=[inputs[key] for key in self.input_key_order], outputs=x)

        return model, self.input_key_order
