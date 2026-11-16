"""Brain protocol: one conversational turn in, a stream of reply text out."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from skcall.core.session import Participant


@dataclass
class Turn:
    speaker: Participant
    text: str
    history: list[dict]              # prior {role, content}; the brain must not mutate it
    mode: str = "private"            # "private" (1:1 with the operator) or "group"
    is_operator: bool = False
    participants: list[Participant] = field(default_factory=list)
    truncated: bool = False          # set by the brain if the token cap cut the reply


class Brain(Protocol):
    def stream(self, turn: Turn) -> AsyncIterator[str]:
        """Yield reply text deltas. Cancellation (barge-in) must be safe."""

    def filler(self, text: str) -> tuple[str, str]:
        """(acknowledgement phrase, bucket) for a slow turn."""

    def all_fillers(self) -> list[str]:
        """Every phrase filler() can return, so they can be pre-rendered."""
