import abc

from basics.base import Base


class IOProcessorBase(Base, metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abc.abstractmethod
    def process_inputs(self, inputs, conversation_start):
        """

        :param inputs: Dict with one or more different types of inputs
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return:
        """
        self._log.error("Please implement this method in a child class")

    @abc.abstractmethod
    def process_response(self, response, scores=None):
        """

        :param response:
        :param scores:

        :return: processed response, scores
        """
        self._log.error("Please implement this method in a child class")


class SimpleIOProcessor(IOProcessorBase):

    def __init__(self,
                 encoding_func,
                 decoding_func,
                 max_sequence_length=None,
                 preprocessing_func=None,
                 **kwargs):
        """

        :param encoding_func       : function that takes a list of sequences as inputs
                                     (not tokenized) and returns a tokenized and indexed list of inputs

        :param decoding_func     : function that takes a list of tokenized and indexed inputs
                                   and returns a detokenized and deindexed list of inputs

        :param max_sequence_length : Optional, max. sequence length of any input sequences

        :param preprocessing_func  : Optional, function that takes a list of sequences as inputs and
                                     returns a preprocessed list of inputs
        """
        super().__init__(**kwargs)

        self._encoding_func = encoding_func
        self._decoding_func = decoding_func

        self._max_sequence_length = max_sequence_length
        self._preprocessing_func = preprocessing_func

    def process_inputs(self, inputs, conversation_start):
        """

        The given inputs are split in to sequence_inputs and other_inputs.
        The sequence_inputs are further processed, before both are returned.

        :param inputs: Dict with one or more different types of inputs
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return: processed sequence_inputs, other_inputs
        """
        sequence_inputs, other_inputs = self._get_input_chats(inputs, conversation_start)

        # Preprocessing is optional
        if callable(self._preprocessing_func):
            sequence_inputs = self._preprocessing_func(sequence_inputs)

        sequence_inputs = self._encoding_func(sequence_inputs)

        # Dealing with too long sentences is optional
        if self._max_sequence_length is not None:
            sequence_inputs = self._cut_off_too_long(sequence_inputs)

        sequence_inputs = self._build_input_sequences(sequence_inputs)

        return sequence_inputs, other_inputs

    def process_response(self, response, scores=None):
        """

        :param response: Dict with one or more different types of inputs
        :param scores: data structure (e.g. list) with one or more scores related to the response
        :return:
        """

        # decoding_func assumes a list of sequences
        return self._decoding_func([response])[0], scores

    @abc.abstractmethod
    def _get_input_chats(self, inputs, conversation_start):
        """

        Get the raw input chats (and other inputs if any) from the inputs dict
        return the selected inputs split as sequence_inputs and other_inputs.

        sequence_inputs will be further processed, other_inputs not

        :param inputs: Dict with one or more different types of inputs
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return: equence_inputs, other_inputs
        """
        raise NotImplementedError("Please implement this method in a child class")

    def _cut_off_too_long(self, inputs):
        raise NotImplementedError("Please implement this method in a child class")

    @abc.abstractmethod
    def _build_input_sequences(self, inputs):
        raise NotImplementedError("Please implement this method in a child class")