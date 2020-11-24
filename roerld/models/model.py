from abc import abstractmethod, ABCMeta


class Model(metaclass=ABCMeta):

    @abstractmethod
    def predict(self, input_dict, training=False):
        """Predicts.
        :param input_dict The model inputs as named key-value pairs.
        :param training: Whether this is in training mode.
        """
        raise NotImplementedError()

    @abstractmethod
    def tensor_predict(self, input_dict, training):
        """Predicts but returns the result as a TF tensor.

        This method may be not be a tf.function that wraps the __call__.

        Due to TF issue 32058, in order to be able to collect the model losses as part of the
        custom training loop (which is a tf.function), this method is required to access the predictions
        directly without there being an inner TF function (as the other predict methods may use) that would cause
         the problems described in the issue.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_trainable_vars(self):
        raise NotImplementedError()

    @abstractmethod
    def get_regularization_losses(self):
        raise NotImplementedError()

    @abstractmethod
    def get_weights(self):
        raise NotImplementedError()

    @abstractmethod
    def set_weights(self, weights):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
        raise NotImplementedError()

    @abstractmethod
    def print_summary(self):
        raise NotImplementedError()
