import re
import statistics

from basics.base import Base

from basics.logging_utils import log_exception


class SimpleBot(Base):
    SYSTEM_COMMAND = re.compile('^##(.+)##$')

    SET_INITIAL_RESPONSE_COMMAND = re.compile('^INIT_RESPONSE="(.*)"$', re.I)
    NO_INITIAL_RESPONSE_COMMAND = re.compile('^NO_INIT_RESPONSE$', re.I)

    RESET_STATE_COMMAND = re.compile('^(reset|r)$', re.I)

    QUIT_COMMAND = re.compile('^(q|quit)$', re.I)

    def __init__(self,
                 conversation_engine,
                 user_name='User',
                 bot_name='Bot',
                 init_response=None,
                 debug=False):
        """

        :param conversation_engine: Conversation engine instance derived from ConversationEngineBase
        :param user_name: Optional, default user name if no user name given with respond_to() method
        :param bot_name: Optional, default bot name
        :param init_response: Optional, initial bot response

        :param debug: If True, more logging is done
        """

        super().__init__()

        self._conversation_engine = conversation_engine

        self._user_name = user_name
        self._bot_name = bot_name

        self._debug = debug

        # chat history
        self._chats = []
        self._is_user = []
        self._actor_name = []

        self._conversation_start = True

        self._init_response = None
        self.set_initial_response(init_response)

        self.reset_state()

    def reset_state(self):
        if self._debug:
            self._log.debug("Resetting state ...")

        self._conversation_engine.reset_state()

        self._chats = []
        self._is_user = []
        self._actor_name = []

        self._conversation_state = None
        self._conversation_start = True
        if self._init_response:
            if self._debug:
                self._log.debug('Adding initial response: %s' % self._init_response)

            self._chats += [self._init_response]
            self._is_user += [False]
            self._actor_name += [self._bot_name]

            return self._init_response

    def set_initial_response(self, r):
        self._init_response = r

        if self._debug:
            self._log.debug('Set initial response : %s' % self._init_response)

    def respond_to(self, user_chat, user_name=None):
        if user_name is None:
            user_name = self._user_name

        if self._debug:
            self._log.debug(f'{user_name} : {user_chat}')

        scores = None
        system_message, bot_chat = self._execute_command_in(user_chat, user_name=user_name)
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
        else:
            self._log.info('SYSTEM : \n%s' % system_message)

        if bot_chat:
            self._log.info('%s [%f] : \n%s' % (self._bot_name, statistics.mean(scores), bot_chat))

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
            return None, None

        try:
            command = m.groups()[0]

            m = self.QUIT_COMMAND.match(command)
            if m is not None:
                system_msg = "quit"
                return system_msg, None

            m = self.RESET_STATE_COMMAND.match(command)
            if m is not None:
                init_bot_response = self.reset_state()
                system_msg = "Conversation model state has been reset."
                return system_msg, init_bot_response

            m = self.SET_INITIAL_RESPONSE_COMMAND.match(command)
            if m is not None:
                try:
                    self.set_initial_response(m.groups()[0])
                except Exception as e:
                    log_exception(self._log, 'An exception occurred parsing the initial response', e)
                    return 'Unable to parse new initial response, initial response not set.', None

                return "New initial response : %s" % (self._init_response if self._init_response else
                                                      "No initial response will be used"), None

            m = self.NO_INITIAL_RESPONSE_COMMAND.match(user_chat)
            if m is not None:
                self.set_initial_response(None)
                return "New initial response : %s" % (self._init_response if self._init_response else
                                                      "No initial response will be used"), None

            return self._forward_command_to_conversation_engine(command, user_name=user_name)
        except Exception as e:
            log_exception(self._log, 'An exception occurred parsing the system command', e)
            return 'Unable to parse system command, no effect.', None

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
            "actor_name": self._actor_name
        }, self._conversation_start)

        return response, scores

