import abc
import statistics
from typing import Protocol, Optional, List, Tuple, Dict

from basics.base import Base

from botshop import ModelEvaluatorBase
from botshop.messages import Message, BotMessage, SystemMessage


class UnableToGenerateValidResponse(Exception):
    pass


class ConversationEngineInterface(Protocol):

    def reset_state(self):
        """
        Reset state of conversation engine, if any
        This allows for stateful engines

        :return:
        """
        ...

    def execute_command(self, command: str, user_name: Optional[str] = None) -> Optional[SystemMessage]:
        """

        :param command:
        :param user_name: Optional, username of the user who input the command

        :return: <system message> = None, when the conversation engine did not process any command
        """
        ...

    def respond(self, chats: List[Message], conversation_start: bool = False) -> Tuple[
        BotMessage,
        Optional[Dict]
    ]:
        """

        :param chats: List of messages representing the conversation
        :param conversation_start: Boolean

        :return: <response to input>, <auxiliary results or None>
        """
        ...


class LocalModelConversationEngine(Base, metaclass=abc.ABCMeta):

    def __init__(self, model_evaluator: ModelEvaluatorBase, debug=False, name=None):
        super().__init__(pybase_logger_name=name)

        self._model_evaluator = model_evaluator
        self._debug = debug
        self._conversation_context = {}

    def reset_state(self):
        self._conversation_context = {}
        self._model_evaluator.reset_state()

    def execute_command(self, command: str, user_name: Optional[str] = None) -> Optional[SystemMessage]:
        """

        :param command:
        :param user_name: Optional, username of the user who input the command

        :return: <system message> = None, when the conversation engine did not process any command
        """
        return None

    @abc.abstractmethod
    def respond(self, chats: List[Message], conversation_start: bool = False) -> Tuple[
        BotMessage,
        Optional[Dict]
    ]:
        """

        :param chats: List of messages representing the conversation
        :param conversation_start: Boolean

        :return: <response to input>, <auxiliary results or None>
        """
        raise NotImplementedError("Please implement this method in a child class")


class BasicLocalModelConversationEngine(LocalModelConversationEngine):

    def __init__(self,
                 io_processor,
                 model_evaluator,
                 select_token_func,
                 sequence_end_detector_func,
                 max_response_length=-1,
                 **kwargs):
        super().__init__(model_evaluator, **kwargs)

        self._io_processor = io_processor

        self._select_token_func = select_token_func
        self._sequence_end_detector_func = sequence_end_detector_func

        self._max_response_length = max_response_length

    def respond(self, chats: List[Message], conversation_start: bool = False) -> Tuple[
        BotMessage,
        Optional[Dict]
    ]:
        """

        :param chats: List of messages representing the conversation
        :param conversation_start: Boolean

        :return: <response to input>, <auxiliary results or None>
        """
        processed_inputs = self._io_processor.process_inputs(chats, conversation_start)

        self._model_evaluator.update_context(processed_inputs, self._conversation_context, conversation_start)

        return self._create_response()

    def _create_response(self, **kwargs):
        """
        Called after model context updated

        :return:
        """

        self._will_create_response()

        prediction_context = {}
        prev_token = None
        response = []
        scores = []
        while True:
            prediction_data = self._model_evaluator.predict_next_token(
                prev_token,
                prediction_context,
                self._conversation_context)

            # Obtain most likely word token and its score
            score, token = self._select_token_func(prediction_data)

            # Add new token and score
            response += [self._unwrap(token)]
            scores += [self._unwrap(score)]

            sequence_end = self._sequence_end_detector_func(response, scores)
            if sequence_end:
                self._log.debug(f"Detected sequence end: {sequence_end}")

                response = response[:-len(sequence_end)]
                scores = scores[:-len(sequence_end)]

                break

            if len(response) >= self._max_response_length > 0:
                self._log.debug("Max. response length reached.")
                break

            prev_token = token

        response, scores = self._process_response(response, scores)

        return BotMessage(text=response, score=statistics.mean(scores)), {
            "scores": scores,
        }

    def _will_create_response(self):
        pass

    def _process_response(self, response, scores):
        return self._io_processor.process_response(response, scores=scores)

    def _unwrap(self, tensor):
        """
        Unwrap scalar tensor

        :param tensor:
        :return:
        """
        return tensor
