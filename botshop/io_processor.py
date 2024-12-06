import abc
from typing import List, Optional, Tuple, Any, Dict

from basics.base import Base

from botshop.messages import Message


class IOProcessorBase(Base, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset_state(self):
        pass

    @abc.abstractmethod
    def process_inputs(self, inputs: List[Message], conversation_start: bool) -> Dict:
        """

        :param inputs: List of conversation messages
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return: Dict with processed inputs
        """
        raise NotImplementedError("Please implement this method in a child class")

    @abc.abstractmethod
    def process_response(
        self,
        response: str,
        scores: Optional[List[float]]=None,
        stop_sequence: Optional[str]=None
    ) -> Tuple[str, Optional[List[float]]]:
        """

        :param response:
        :param scores:
        :param stop_sequence:

        :return: processed response, scores
        """
        raise NotImplementedError("Please implement this method in a child class")
