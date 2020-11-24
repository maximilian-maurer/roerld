import tensorflow as tf
import numpy as np


def datatype_to_tf_datatype(datatype):
    if datatype == np.int8:
        return tf.int8
    elif datatype == np.int16:
        return tf.int16
    elif datatype == np.int32:
        return tf.int32
    elif datatype == np.int64:
        return tf.int64
    elif datatype == np.uint8:
        return tf.uint8
    elif datatype == np.uint16:
        return tf.uint16
    elif datatype == np.uint32:
        return tf.uint32
    elif datatype == np.uint64:
        return tf.uint64
    elif datatype == np.float32:
        return tf.float32
    elif datatype == np.float64:
        return tf.float64
    raise NotImplementedError()
