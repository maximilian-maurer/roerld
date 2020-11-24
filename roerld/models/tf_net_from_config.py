from typing import List

import tensorflow as tf
from roerld.config.experiment_config import ExperimentConfigError


def resolve_activation(activation_name):
    supported_activations = {
        "none": None,
        "relu": tf.keras.activations.relu,
        "sigmoid": tf.keras.activations.sigmoid,
        "elu": tf.keras.activations.elu,
        "exponential": tf.keras.activations.exponential,
        "hard_sigmoid": tf.keras.activations.hard_sigmoid,
        "linear": tf.keras.activations.linear,
        "selu": tf.keras.activations.selu,
        "softmax": tf.keras.activations.softmax,
        "softplus": tf.keras.activations.softplus,
        "softsign": tf.keras.activations.softsign,
        "tanh": tf.keras.activations.tanh,
        "swish": tf.keras.activations.swish
    }
    if activation_name not in supported_activations:
        raise ExperimentConfigError(f"Activation type {activation_name} unknown.")

    return supported_activations[activation_name]


def _select_regularizer_if_exists(all_variables, key):
    if key not in all_variables:
        return None
    if all_variables[key]["name"] == "l1":
        return tf.keras.regularizers.l1(l=all_variables[key]["l"])
    if all_variables[key]["name"] == "l2":
        return tf.keras.regularizers.l2(l=all_variables[key]["l"])
    if all_variables[key]["name"] == "l1_l2":
        return tf.keras.regularizers.l1_l2(l1=all_variables[key]["l1"],
                                           l2=all_variables[key]["l2"])
    raise ExperimentConfigError(f"Regularizer type {all_variables[key]['name']} unknown.")


def _select_initializer_if_exists(all_variables, key, default):
    if key not in all_variables:
        return default
    if all_variables[key]["name"] == "VarianceScaling":
        return tf.keras.initializers.VarianceScaling(
            scale=all_variables[key]["scale"],
            mode=all_variables[key]["mode"],
            distribution=all_variables[key]["distribution"]
        )
    raise ExperimentConfigError(f"Initializer type {all_variables[key]['name']} unknown.")


def _create_conv2d_layer(parameters, input_var):
    conv = tf.keras.layers.Convolution2D(
        filters=parameters["filters"],
        kernel_size=parameters["kernel_size"],
        strides=parameters["strides"],
        activation=resolve_activation(parameters["activation"]),
        kernel_regularizer=_select_regularizer_if_exists(parameters, "kernel_regularizer"),
        bias_regularizer=_select_regularizer_if_exists(parameters, "bias_regularizer"),
        activity_regularizer=_select_regularizer_if_exists(parameters, "activity_regularizer"))(input_var)
    return conv


def _create_max_pool2d_layer(parameters, input_var):
    max_pool = tf.keras.layers.MaxPool2D(
        pool_size=parameters["pool_size"],
        strides=parameters["strides"])(input_var)
    return max_pool


def _create_reshape(parameters, input_var):
    reshape = tf.keras.layers.Reshape(
        target_shape=parameters["target_shape"])(input_var)
    return reshape


def _create_fully_connected_layer(parameters, input_var):
    fc = tf.keras.layers.Dense(units=parameters["units"],
                               activation=resolve_activation(parameters["activation"]),
                               kernel_regularizer=_select_regularizer_if_exists(parameters, "kernel_regularizer"),
                               bias_regularizer=_select_regularizer_if_exists(parameters, "bias_regularizer"),
                               activity_regularizer=_select_regularizer_if_exists(parameters, "activity_regularizer"),
                               kernel_initializer=_select_initializer_if_exists(parameters, "kernel_initializer",
                                                                                "glorot_uniform")
                               )(input_var)
    return fc


def _create_flatten(input_var):
    return tf.keras.layers.Flatten()(input_var)


def _create_leaky_relu(parameters, input_var):
    return tf.keras.layers.LeakyReLU(alpha=parameters["alpha"])(input_var)


def _create_net(layers: List, input_var):
    layer_in = input_var

    for layer in layers:
        if layer["name"] == "Convolution2D":
            layer_in = _create_conv2d_layer(layer, layer_in)
        elif layer["name"] == "Reshape":
            layer_in = _create_reshape(layer, layer_in)
        elif layer["name"] == "Dense":
            layer_in = _create_fully_connected_layer(layer, layer_in)
        elif layer["name"] == "MaxPool2D":
            layer_in = _create_max_pool2d_layer(layer, layer_in)
        elif layer["name"] == "Flatten":
            layer_in = _create_flatten(layer_in)
        elif layer["name"] == "LeakyReLU":
            layer_in = _create_leaky_relu(layer, layer_in)
        else:
            raise NotImplementedError(f"Network key type {layer['name']} unknown.")

    return layer_in
