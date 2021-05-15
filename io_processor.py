import abc

from basics.base import Base


class IOProcessorBase(Base, metaclass=abc.ABCMeta):

    def __init__(self):
        super().__init__()

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
    def process_response(self, response, scores):
        """

        :param response:
        :param scores:

        :return: processed response, scores
        """
        self._log.error("Please implement this method in a child class")


class SimpleIOProcessor(IOProcessorBase):

    def __init__(self,
                 max_sequence_length,
                 preprocessing_func,
                 indexing_func,
                 deindexing_func,
                 sequence_end_index):
        """

        :param max_sequence_length : Max. sequence length of any input sequences

        :param preprocessing_func  : function that takes a list of sequences as inputs and
                                     returns a preprocessed list of inputs

        :param indexing_func       : function that takes a list of sequences as inputs
                                     (not tokenized) and returns a tokenized and indexed list of inputs

        :param deindexing_func     : function that takes a list of tokenized and indexed inputs
                                     and returns a detokenized and deindexed list of inputs

        :param sequence_end_index  : index for sequence end token, to add at the end of the
                                     tokenized and index sequences
        """
        super().__init__()

        self._max_sequence_length = max_sequence_length
        self._preprocessing_func = preprocessing_func
        self._indexing_func = indexing_func
        self._deindexing_func = deindexing_func
        self._sequence_end_index = sequence_end_index

    def process_inputs(self, inputs, conversation_start):
        """

        :param inputs: Dict with one or more different types of inputs
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return:
        """
        inputs = self._get_input_chats(inputs, conversation_start)
        inputs = self._preprocessing_func(inputs)
        inputs = self._indexing_func(inputs)
        inputs = self._cut_off_too_long(inputs)
        inputs = self._add_sequence_end_index(inputs)

        return inputs

    def process_response(self, response, scores):
        """

        :param inputs: Dict with one or more different types of inputs
        :return:
        """

        # deindexing_func assumes a list of sequences
        return self._deindexing_func([response])[0], scores

    @abc.abstractmethod
    def _get_input_chats(self, inputs, conversation_start):
        """

        Get the raw input chats (and other inputs if any) from the inputs dict

        :param inputs: Dict with one or more different types of inputs
        :param conversation_start: Bool, if the given inputs are the initial inputs of a new
                                   conversation

        :return:
        """
        self._log.error("Please implement this method in a child class")

    def _cut_off_too_long(self, inputs):
        inputs_checked = []
        for input in inputs:
            if len(input) > self._max_sequence_length-1:
                input = input[:self._max_sequence_length-1]

            inputs_checked += [input]

        return inputs_checked

    def _add_sequence_end_index(self, inputs):
        inputs_stuffed = []
        for input in inputs:
            inputs_stuffed += [input + [self._sequence_end_index]]

        return inputs_stuffed
