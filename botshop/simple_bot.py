import abc
import sys

import re
import statistics
from typing import Protocol, List, Tuple, Optional, Dict

from .conversation_engine import UnableToGenerateValidResponse

from basics.base import Base

from basics.logging import get_logger
from basics.logging_utils import log_exception

from .messages import Message, BotMessage, UserMessage


class CreateBotResponseFunc(Protocol):

    def __call__(self, conversation: List[Message]) -> Tuple[
        BotMessage,
        Optional[Dict]
    ]:
        """
        Create bot response based on given conversation messages

        :param conversation: List of conversation messages

        :return: Tuple:
            Bot Response, optional dict with auxiliary data
        """
        pass


class BotBase(Base, metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def reset_state(self):
        raise NotImplementedError("Please implement this method in your child class")

    @abc.abstractmethod
    def respond_to(self, user_chat: UserMessage, user_name: Optional[str] = None):
        raise NotImplementedError("Please implement this method in your child class")


class SimpleBot(BotBase):
    SYSTEM_COMMAND = re.compile('^##(.+)##$')

    RESET_STATE_COMMAND = re.compile('^(reset|r)$', re.I)

    QUIT_COMMAND = re.compile('^(q|quit)$', re.I)

    def __init__(self,
                 create_bot_response_func: CreateBotResponseFunc,
                 user_name: str = 'User',
                 bot_name: str = 'Bot',
                 init_chats: Optional[List[Message]] = None,
                 log_conversation: bool = True,
                 debug: bool = False,
                 name: Optional[str] = None):
        """

        :param create_bot_response_func: function that is used to create a bot response based on the conversation so far
        :param user_name: Optional, default username if no username given with respond_to() method
        :param bot_name: Optional, default bot name
        :param init_chats: Optional, list, initial chat history
                           [
                                <Message 0>
                                ...
                                <Message N>
                           ]

        :param log_conversation: If True, the conversations will be logged to stdout
        :param debug: If True, more logging is done
        :param name: Name of Bot system (class name by default)
        """

        super().__init__(pybase_logger_name=name)

        self._create_bot_response_func = create_bot_response_func

        self._user_name = user_name
        self._bot_name = bot_name

        self._log_conversation = log_conversation
        self._debug = debug

        # chat history
        if init_chats is None:
            init_chats = []
            init_is_user = []
            init_actor_name =[]
        else:
            if init_is_user is None:
                init_is_user = [bool(i % 2) for i in range(len(init_chats))]

            if init_actor_name is None:
                init_actor_name = [user_name if (i % 2) == 0 else bot_name for i in range(len(init_chats))]

            if len(init_chats) != len(init_is_user) != len(init_actor_name):
                raise Exception("Lists init_chats, init_is_user and init_actor_name nust all have the same length")

            if self._debug:
                self._log.debug("Initial chat history provided:")
                for actor_name, is_user, chat in zip(init_actor_name, init_is_user, init_chats):
                    self._log.debug(f"[{'USER' if is_user else 'BOT '}] {actor_name}: {chat}")

        self._init_chats = init_chats.copy()
        self._init_is_user = init_is_user.copy()
        self._init_actor_name = init_actor_name.copy()

        self._chats = None
        self._is_user = None
        self._actor_name = None

        self._conversation_start = True

        self.reset_state()

    def reset_state(self):
        if self._debug:
            self._log.debug("Resetting state ...")

        self._conversation_engine.reset_state()

        self._chats = self._init_chats.copy()
        self._is_user = self._init_is_user.copy()
        self._actor_name = self._init_actor_name.copy()

        self._conversation_state = None
        self._conversation_start = True

    def respond_to(self, user_chat, user_name=None):
        if user_name is None:
            user_name = self._user_name

        if self._log_conversation:
            self._log_user_input(user_chat, user_name)

        scores = None
        system_message = self._execute_command_in(user_chat, user_name=user_name)
        bot_chat = None
        other_outputs = None
        if system_message is None:  # No command executed
            self._chats.append(user_chat)
            self._is_user.append(True)
            self._actor_name.append(user_name)

            try:
                bot_chat, scores, other_outputs = self._respond()
            except UnableToGenerateValidResponse as e:
                system_message = f"Bot {self._bot_name} was unable to generate a valid response. " \
                                 f"Your chat is not recorded."
                log_exception(self._log, "Response generation failed", e)
            except Exception as e:
                system_message = f"An unexpected exception occurred, bot {self._bot_name} was " \
                                 f"unable to generate a response. Your chat is not recorded."
                log_exception(self._log, "Response generation failed", e)

            if system_message is None:  # No exception occurred
                if self._conversation_start:
                    self._conversation_start = False

                self._chats.append(bot_chat)
                self._is_user.append(False)
                self._actor_name.append(self._bot_name)

                if self._log_conversation:
                    self._log_response(self._bot_name, bot_chat, score=self._calc_final_response_score(scores))
            else:
                # An issue occurred, remove last user chat
                self._chats = self._chats[:-1]
                self._is_user = self._is_user[:-1]
                self._actor_name = self._actor_name[:-1]

        if self._log_conversation and system_message is not None:
            self._log_response("System", system_message)

        return {
            'bot': bot_chat,
            'scores': scores,
            'other_outputs': other_outputs,
            'system': system_message,
        }

    def _execute_command_in(self, user_chat, user_name=None):
        if user_name is None:
            user_name = self._user_name

        m = self.SYSTEM_COMMAND.match(user_chat)

        if m is None:
            return None

        try:
            command = m.groups()[0]

            m = self.QUIT_COMMAND.match(command)
            if m is not None:
                system_msg = "quit"
                return system_msg

            m = self.RESET_STATE_COMMAND.match(command)
            if m is not None:
                self.reset_state()
                system_msg = "Conversation model state has been reset."
                return system_msg

            return self._forward_command_to_conversation_engine(command, user_name=user_name)
        except Exception as e:
            log_exception(self._log, 'An exception occurred parsing the system command', e)
            return 'Unable to parse system command, no effect.'

    def _forward_command_to_conversation_engine(self, command, user_name=None):
        if user_name is None:
            user_name = self._user_name

        try:
            return self._conversation_engine.execute_command(command, user_name=user_name)
        except Exception as e:
            log_exception(self._log, 'Forwarding system command to evaluator failed', e)
            return 'Execution of system command by evaluator failed, no effect.', None

    def _respond(self):
        response, scores, other_outputs = self._conversation_engine.respond({
            "chats": self._chats,
            "is_user": self._is_user,
            "actor_name": self._actor_name,
            "responding_actor": self._bot_name
        }, self._conversation_start)

        return response, scores, other_outputs

    def _calc_final_response_score(self, scores):
        if scores is None:
            return None

        return statistics.mean(scores)

    def _log_user_input(self, user_chat, user_name=None):
        if user_name is None:
            user_name = "User"

        self._log.info(f'{user_name} : {user_chat}')

    def _log_response(self, actor_name, response, score=None):
        if score is not None:
            self._log.info('%s [%f] : \n%s\n' % (actor_name, score, response))
        else:
            self._log.info('%s : \n%s\n' % (actor_name, response))


def chat_with(bot: BotBase, user_name: str = "You", logger=None):
    if logger is None:
        logger = get_logger("Bot")

    while True:
        logger.info(f'{user_name} :')
        # 3) ask for input
        user_input = input()
        sys.stdout.write('\n')
        sys.stdout.flush()

        response = bot.respond_to(user_input)

        if response["system"] == "quit":
            break
