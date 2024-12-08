import abc
import logging
import sys

import re
import statistics
from typing import List, Optional, Protocol, Tuple, Dict, Callable

from botshop.conversation_engine import UnableToGenerateValidResponse, ConversationEngineInterface

from basics.base import Base

from basics.logging import get_logger
from basics.logging_utils import log_exception

from botshop.messages import Message, BotMessage, UserMessage, SystemMessage


class BotInterface(Protocol):

    def reset_state(self):
        ...

    def respond_to(self, user_chat: UserMessage) -> Tuple[Message, Optional[Dict]]:
        ...


class SimpleBot(Base):
    SYSTEM_COMMAND = re.compile('^##(.+)##$')

    RESET_STATE_COMMAND = re.compile('^(reset|r)$', re.I)

    QUIT_COMMAND = re.compile('^(q|quit)$', re.I)

    def __init__(
            self,
            conversation_engine: ConversationEngineInterface,
            user_name: Optional[str] = None,
            bot_name: Optional[str] = None,
            init_chats: Optional[List[Message]] = None,
            log_conversation: bool = True,
            debug: bool = False,
            name: Optional[str] = None):
        """

        :param conversation_engine: object implementing the ConversationEngineInterface
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

        if user_name is None:
            user_name = "User"

        if bot_name is None:
            bot_name = "Bot"

        self._conversation_engine = conversation_engine

        self._user_name = user_name
        self._bot_name = bot_name

        self._log_conversation = log_conversation
        self._debug = debug

        # chat history
        if init_chats is None:
            init_chats = []

        if self._debug:
            self._log.debug("Initial chat history provided:")
            for chat in init_chats:
                is_user = isinstance(chat, UserMessage)
                self._log.debug(f"[{'USER' if is_user else 'BOT '}] {chat.actor_name}: {chat}")

        self._init_chats = init_chats.copy()

        self._chats = None

        self._conversation_start = True

        self.reset_state()

    def reset_state(self):
        if self._debug:
            self._log.debug("Resetting state ...")

        self._conversation_engine.reset_state()

        self._chats = self._init_chats.copy()

        self._conversation_start = True

    def respond_to(self, user_chat: UserMessage) -> Tuple[Message, Optional[Dict]]:
        user_chat = UserMessage.copy(user_chat)

        if user_chat.actor_name is None:
            user_chat.actor_name = self._user_name

        if self._log_conversation:
            self._log_chat(user_chat)

        system_message = self._execute_command_in(user_chat)
        bot_chat = None
        other_outputs = None
        if system_message is None:  # No command executed
            self._chats.append(user_chat)

            try:
                bot_chat, other_outputs = self._respond()
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

                if self._log_conversation:
                    self._log_chat(bot_chat)
            else:
                # An issue occurred, remove last user chat
                self._chats = self._chats[:-1]

        if system_message is not None:
            system_message = SystemMessage(text=system_message)

        if self._log_conversation and system_message is not None:
            self._log_chat(system_message)

        response_message = system_message if system_message is not None else bot_chat

        return response_message, other_outputs

    def _execute_command_in(self, user_chat):
        user_name = self._get_user_name(user_chat)

        m = self.SYSTEM_COMMAND.match(user_chat.text)

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

    def _forward_command_to_conversation_engine(self, command, user_name):
        try:
            return self._conversation_engine.execute_command(command, user_name=user_name)
        except Exception as e:
            log_exception(self._log, 'Forwarding system command to evaluator failed', e)
            return 'Execution of system command by evaluator failed, no effect.', None

    def _respond(self):
        bot_message, other_outputs = self._conversation_engine.respond(
            self._chats,
            self._conversation_start)

        bot_message.actor_name = self._bot_name

        return bot_message, other_outputs

    def _calc_final_response_score(self, scores):
        if scores is None:
            return None

        return statistics.mean(scores)

    def _log_chat(self, chat: Message):
        if isinstance(chat, BotMessage):
            if chat.score is not None:
                self._log.info('%s [%f] : \n%s\n' % (chat.actor_name, chat.score, chat.text))
                return

        self._log.info('%s : \n%s\n' % (chat.actor_name, chat.text))

    def _get_user_name(self, user_chat: UserMessage):
        user_name = user_chat.actor_name
        user_name = user_name if user_name is not None else self._user_name
        user_name = user_name if user_name is not None else "User"

        return user_name


class LogAuxResultsFunc(Protocol):

    def __call__(self, aux_results: Optional[Dict], logger: logging.Logger):
        ...


def chat_with(
    bot: BotInterface,
    user_name: str = "You",
    log_aux_results_func: LogAuxResultsFunc = None,
    logger=None
):
    if logger is None:
        logger = get_logger("Bot")

    while True:
        logger.info(f'{user_name} :')
        # 3) ask for input
        user_input = input()
        sys.stdout.write('\n')
        sys.stdout.flush()

        response, aux_results = bot.respond_to(UserMessage(
            text=user_input,
            actor_name=user_name
        ))

        if log_aux_results_func:
            log_aux_results_func(aux_results, logger)

        if isinstance(response, SystemMessage) and response.text == "quit":
            break
