import abc

from basics.base import Base


class ConversationEngineBase(Base, metaclass=abc.ABCMeta):

    def __init__(self, io_processor, model_evaluator, max_sequence_length=40, debug=False):
        super().__init__()

        self._io_processor = io_processor
        self._model_evaluator = model_evaluator

        self._max_sequence_length = max_sequence_length

        self._debug = debug

        self._conversation_context = {}

        self._validate()

    def reset_state(self):
        self._conversation_context = {}
        self._model_evaluator.reset_state()

    def execute_command(self, command, user_name=None):
        """

        :param command:
        :param user_name: Optional, user name of the user who input the command

        :return: <system message>, <bot response>
                 = None, None when the conversation engine did not process any command
        """
        return None, None

    @abc.abstractmethod
    def respond(self, inputs, conversation_start=False):
        """

        :param inputs: Dict with one or more different toes of inputs
        :param conversation_start: Boolean

        :return: <response to input>, <score(s)>
        """
        self._log.error("Please implement this method in a child class")

    def _validate(self):
        # TODO
        self._valid = True


class BasicConversationEngine(ConversationEngineBase):

    def __init__(self,
                 io_processor,
                 model_evaluator,
                 select_token_func,
                 sequence_end_index,
                 **kwargs):
        super().__init__(io_processor, model_evaluator, **kwargs)

        self._select_token_func = select_token_func
        self._sequence_end_index = sequence_end_index

    def respond(self, inputs, conversation_start=False):
        processed_inputs = self._io_processor.process_inputs(inputs, conversation_start)

        self._model_evaluator.update_context(processed_inputs, self._conversation_context, conversation_start)

        return self._create_response()

    def _create_response(self):
        """
        Called after model context updated

        :return:
        :rtype:
        """

        prediction_context = {}
        prev_token = None
        response = []
        scores = []
        for _ in range(self._max_sequence_length):

            prediction_data = self._model_evaluator.predict_next_token(prev_token,
                                                                       prediction_context,
                                                                       self._conversation_context)
            # Obtain most likely word token and its score
            score, token = self._select_token_func(prediction_data)

            if token == self._sequence_end_index:
                break

            # Record token and score
            response += [self._unwrap(token)]
            scores += [self._unwrap(score)]

            prev_token = token

        response, scores = self._io_processor.process_response(response, scores)

        return response, scores

    def _unwrap(self, tensor):
        """
        Unwrap scalar tensor

        :param tensor:
        :return:
        """
        return tensor
