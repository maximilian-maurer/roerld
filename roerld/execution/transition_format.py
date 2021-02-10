import gym
import numpy as np
from typing import Tuple, Dict, List


class TransitionFormat:
    """ Format of the input data to various components. While founded in the data output by the environment and the
    other associated quantities (in particular observation, reward, dones, next_observation), but components such as
    preprocessing may arbitrarily change this."""

    def __init__(self,
                 observation_format_spec: Dict[str, Tuple[Tuple, np.dtype]],
                 action_format_spec: Tuple[Tuple, np.dtype],
                 other_format_spec: Dict[str, Tuple[Tuple, np.dtype]]
                 ):
        """
        See `from_gym_spaces` to instantiate this class from gym spaces with stronger typing.

        Args:
            observation_format_spec: A dictionary of key-value pairs that map a key name to a tuple (shape, dtype) in
                                        the observation dictionary.
            action_format_spec: A Tuple (shape, dtype) describing the format of the action space. The action space must
                                    be flat, hence the shape has to be length 1.
        """
        self.action_format_spec = action_format_spec
        self.observation_format_spec = observation_format_spec
        self.other_format_spec = other_format_spec

    @staticmethod
    def from_gym_spaces(observation_space: gym.spaces.Space, action_space: gym.spaces.Space):
        """
        Creates a transition format instance from the gym specification of the environment. Environments are required
        to have these properties (to accommodate the rest of the code for this project):
            * The observation space must be a dict space.
            * The value of the observation space must be gym.spaces.Box
            * There may not be an observation key called actions
            * The action space must be a box space
        Args:
            observation_space: the observation space of the environment.
            action_space: the action space of the environment.

        Returns:
            A transition format instance.
        """
        if type(observation_space) != gym.spaces.Dict:
            raise ValueError("Environments using this class are required to have a dict-space as their observation "
                             "space. The dict-space may not be nested. To use other environments, it is necessary to "
                             "use an adapter class.")
        input_spec = {}

        # we ensure above that this is a dict space (which has the spaces attribute)
        # noinspection PyUnresolvedReferences
        for key, subspace in observation_space.spaces.items():
            if key == "actions":
                raise ValueError("'action' is a reserved key. To use an environment which has a key"
                                 "'action' in its observation, the creation of an adapter class is necessary.")
            if type(subspace) != gym.spaces.Box:
                print("Observation space was:", str(observation_space))
                raise ValueError("This class works with an observation space of Box spaces inside"
                                 "a dict space. To use differently structured environments an adapter class "
                                 "is necessary.")
            input_spec[key] = (subspace.shape, subspace.dtype)

        action_spec = (action_space.shape, action_space.dtype)

        if len(action_spec[0]) != 1:
            raise ValueError("Action spaces used with this project must be a one-dimensional gym.spaces.Box "
                             "space.")

        return TransitionFormat(observation_format_spec=input_spec,
                                action_format_spec=action_spec,
                                other_format_spec={
                                    "rewards": (None, np.float32),
                                    # todo this is technically the wrong datatype for the dones, but should only be
                                    #   changed once proper testing is in place to validate that the change does not
                                    #   result in changed behavior due to implicit casting problems.
                                    "dones": (None, np.float32)
                                })

    def action_shape(self) -> Tuple:
        return self.action_format_spec[0]

    def action_dtype(self) -> np.dtype:
        return self.action_format_spec[1]

    def observation_base_keys(self):
        return list(self.observation_format_spec.keys())

    def observation_base_shapes(self) -> Dict[str, Tuple]:
        """
        Returns: The shapes of the observation keys by the name that they are returned on their own (as opposed to
        in the transition which stores the observation and next observation under different key names)
        """
        return {k: v[0] for k, v in self.observation_format_spec.items()}

    def observation_base_dtypes(self) -> Dict[str, Tuple]:
        """
        Returns: The dtypes of the observation keys by the name that they are returned on their own (as opposed to
        in the transition which stores the observation and next observation under different key names)
        """
        return {k: v[1] for k, v in self.observation_format_spec.items()}

    def observation_in_transition_keys(self) -> List[str]:
        """
        Returns: The key names of the (previous) observation in a transition (as opposed to those of the next
        observation, or those that the environment would return).
        """
        return ["observations_"+k for k in self.observation_format_spec.keys()]

    def next_observation_in_transition_keys(self) -> List[str]:
        """
        Returns: The key names of the next observation in a transition (as opposed to those of the previous
        observation, or those that the environment would return).
        """
        return ["next_observations_"+k for k in self.observation_format_spec.keys()]

    def transition_keys(self):
        return [
            *self.observation_in_transition_keys(),
            *self.next_observation_in_transition_keys(),
            *list(self.other_format_spec.keys()),
            "actions"
        ]

    @staticmethod
    def _normalize_key(key):
        next_observation_prefix = "next_observations_"
        observation_prefix = "observations_"

        if key.startswith(next_observation_prefix):
            key = key[len(next_observation_prefix):]
        elif key.startswith(observation_prefix):
            key = key[len(observation_prefix):]

        return key

    def key_shape(self, key) -> Tuple:
        key = TransitionFormat._normalize_key(key)

        if key == "actions":
            return self.action_shape()
        elif key in self.observation_format_spec:
            return self.observation_format_spec[key][0]
        elif key in self.other_format_spec:
            return self.other_format_spec[key][0]

        raise IndexError(f"Key {key} not in the TransitionFormat.")

    def key_dtype(self, key) -> np.dtype:
        key = TransitionFormat._normalize_key(key)

        if key == "actions":
            return self.action_dtype()
        elif key in self.observation_format_spec:
            return self.observation_format_spec[key][1]
        elif key in self.other_format_spec:
            return self.other_format_spec[key][1]

        raise IndexError(f"Key {key} not in the TransitionFormat.")

    def to_legacy_format(self):
        """
        Temporary helpers that converts this back to the representation it was in before.
        """
        format = {
            k: (self.key_shape(k), self.key_dtype(k)) for k in self.observation_base_keys()
        }
        format["actions"] = (self.action_shape(), self.action_dtype())

        return format
