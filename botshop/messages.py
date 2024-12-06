from dataclasses import dataclass
from typing import Optional


@dataclass
class Message:

    text: str
    actor_name: Optional[str] = None

    @classmethod
    def copy(cls, message: "Message") -> "Message":
        return cls(**vars(message))


@dataclass
class UserMessage(Message):
    pass


@dataclass
class BotMessage(Message):
    score: Optional[float] = None


@dataclass
class SystemMessage(Message):
    actor_name: Optional[str] = "System"
