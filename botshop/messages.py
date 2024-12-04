from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:

    text: str
    actor_name: Optional[str] = None


class UserMessage(Message):
    pass


class BotMessage(Message):
    score: Optional[float] = None


class SystemMessage(Message):
    pass
