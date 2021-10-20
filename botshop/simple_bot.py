import re
import statistics

from basics.base import Base

from basics.logging import get_logger
from basics.logging_utils import log_exception


def chat_with(bot, user_name="You", logger=None):
    if logger is None:
        logger = get_logger("Bot")

    while True:
        logger.info(f'{user_name} :')
        # 3) ask for input
        user_input = input()

        response = bot.respond_to(user_input)

        if response["system"] == "quit":
            break


class SimpleBot(Base):
    SYSTEM_COMMAND = re.compile('^##(.+)##$')

    RESET_STATE_COMMAND = re.compile('^(reset|r)$', re.I)

    QUIT_COMMAND = re.compile('^(q|quit)$', re.I)

    def __init__(self,
                 conversation_engine,
                 user_name='User',
                 bot_name='Bot',
                 init_chats=None,
                 init_is_user=None,
                 init_actor_name=None,
                 debug=False):
        """

        :param conversation_engine: Conversation engine instance derived from ConversationEngineBase
        :param user_name: Optional, default user name if no user name given with respond_to() method
        :param bot_name: Optional, default bot name
        :param init_chats: Optional, list, initial chat history
                           [
                                <chat 0>
                                ...
                                <chat N>
                           ]
        :param init_is_user: Optional, list, when init_chats is given, per chat it can be provided if it was a
                             user (True) or bot (False) utterance. len(init_is_user) must be equal to len(init_chats)
                             If not provided, it is assumed that the user started, and that the user and the bot take
                             equal turns after each other.
                             [
                                <True|False>
                                ...
                                <True|False>
                            ]
        :param init_actor_name: Optional, list, when init_chats is given, per chat the actor name, who uttered the chat,
                                can be provided. If this list is not provided, it is assumed that the user started and
                                that the bot and user take equal turns after each other; in this way the default
                                user name and bot name are used as actor names.
                                [
                                    <actor name 0>
                                    ...
                                    <actor name N>
                                ]

        :param debug: If True, more logging is done
        """

        super().__init__()

        self._conversation_engine = conversation_engine

        self._user_name = user_name
        self._bot_name = bot_name

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

        self._init_chats = init_chats
        self._init_is_user = init_is_user
        self._init_actor_name = init_actor_name

        self._chats = None
        self._is_user = None
        self._actor_name = None

        self._conversation_start = True

        self.reset_state()

    def reset_state(self):
        if self._debug:
            self._log.debug("Resetting state ...")

        self._conversation_engine.reset_state()

        self._chats = self._init_chats
        self._is_user = self._init_is_user
        self._actor_name = self._init_actor_name

        self._conversation_state = None
        self._conversation_start = True

    def respond_to(self, user_chat, user_name=None):
        if user_name is None:
            user_name = self._user_name

        if self._debug:
            self._log.debug(f'{user_name} : {user_chat}')

        scores = None
        system_message = self._execute_command_in(user_chat, user_name=user_name)
        bot_chat = None
        if system_message is None:  # No command executed
            self._chats.append(user_chat)
            self._is_user.append(True)
            self._actor_name.append(user_name)

            bot_chat, scores = self._respond()

            if self._conversation_start:
                self._conversation_start = False

            self._chats.append(bot_chat)
            self._is_user.append(False)
            self._actor_name.append(self._bot_name)

            self._log_response(bot_chat, score=self._calc_final_response_score(scores))
        else:
            self._log.info('SYSTEM : \n%s' % system_message)

        return {
            'bot': bot_chat,
            'scores': scores,
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
        response, scores = self._conversation_engine.respond({
            "chats": self._chats,
            "is_user": self._is_user,
            "actor_name": self._actor_name,
            "responding_actor": self._bot_name
        }, self._conversation_start)

        return response, scores

    def _calc_final_response_score(self, scores):
        return statistics.mean(scores)

    def _log_response(self, response, score=None):
        if score is not None:
            self._log.info('%s [%f] : \n%s' % (self._bot_name, score, response))
        else:
            self._log.info('%s : \n%s' % (self._bot_name, response))

