from botshop.conversation_engine import LocalModelConversationEngine as BasicConversationEngineBase


class BasicConversationEngine(BasicConversationEngineBase):

    def _unwrap(self, tensor):
        """
        Unwrap scalar tensor

        :param tensor:
        :return:
        """
        return tensor.item()
