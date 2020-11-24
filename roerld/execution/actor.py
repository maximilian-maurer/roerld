from abc import abstractmethod, ABC


class Actor(ABC):
    @abstractmethod
    def choose_action(self, observation):
        raise NotImplementedError()
