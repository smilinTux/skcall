"""Text rules for a spoken conversation: sentence chunking of a streamed reply,
transcript filters, and the voice "leave" command.

Filters are lifted from skchat (voice_engine/stt.py is_hallucination,
transports/livekit.py is_non_speech / TranscriptDedup), GPL-3.0, same authors.
"""
from __future__ import annotations

import re
import time
from typing import Callable

# ---------------------------------------------------------------- chunking
_ABBREV = {"mr", "mrs", "ms", "dr", "st", "vs", "etc", "e.g", "i.e", "jr", "sr", "no", "approx"}
_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")


def strip_formatting(text: str) -> str:
    """Markdown and emoji out (the text gets spoken)."""
    text = re.sub(r"\*{1,3}([^*]*)\*{1,3}", r"\1", text)
    text = re.sub(r"(?<!\w)_{1,3}([^_]+)_{1,3}(?!\w)", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.M)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`+", "", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF️‍]", "", text)
    # A spoken dash is a pause; never hand a long dash to TTS.
    text = re.sub(r"\s*[\u2014\u2013]\s*", ", ", text)
    return text


class SentenceChunker:
    """Split a token stream into speakable sentences as early as possible.

    The first sentence is released the moment it is complete (that is the
    first-audio latency). Later sentences shorter than `min_chars` are joined
    with the next one, so F5 is not called for every "Okay." mid-reply. A run
    with no terminator longer than `max_chars` is cut at the last comma or
    space, never mid-word."""

    def __init__(self, min_chars: int = 24, max_chars: int = 220, first_clause_chars: int = 30):
        self.first_clause_chars = first_clause_chars
        self.min_chars = min_chars
        self.max_chars = max_chars
        self._buf = ""
        self._held = ""
        self.emitted = 0

    def _release(self, s: str) -> list[str]:
        s = s.strip()
        if not s:
            return []
        if self.emitted == 0:
            self.emitted += 1
            return [s]
        s = f"{self._held} {s}".strip() if self._held else s
        self._held = ""
        if len(s) < self.min_chars:
            self._held = s
            return []
        self.emitted += 1
        return [s]

    def push(self, delta: str) -> list[str]:
        self._buf += delta
        out: list[str] = []
        while True:
            cut = None
            for m in _END.finditer(self._buf):
                end = m.end()
                if end >= len(self._buf):
                    break  # need the following whitespace to be sure
                word = re.findall(r"[\w.]+$", self._buf[: m.start()])
                w = word[0].lower() if word else ""
                if m.group(0) == "." and w.rstrip(".") in _ABBREV:
                    continue
                cut = end
                break
            if self.emitted == 0 and self.first_clause_chars and (cut is None or cut > 2 * self.first_clause_chars):
                # First-audio lever: F5 time grows with length (51 chars ~1.0 s,
                # 150+ chars ~2 s at nfe 16) and her first sentence is often
                # long, so speak its first clause while the rest is written.
                limit = cut if cut is not None else len(self._buf)
                m = re.search(r"[,;:](?=\s)", self._buf[self.first_clause_chars:limit])
                if m and limit > self.first_clause_chars + m.end() + 1:
                    i = self.first_clause_chars + m.end()
                    out += self._release(self._buf[:i])
                    self._buf = self._buf[i:]
                    continue
            if cut is None:
                if len(self._buf) > self.max_chars:
                    i = max(self._buf.rfind(", ", 0, self.max_chars), self._buf.rfind(" ", 0, self.max_chars))
                    if i > 0:
                        out += self._release(self._buf[: i + 1])
                        self._buf = self._buf[i + 1:]
                        continue
                break
            out += self._release(self._buf[:cut])
            self._buf = self._buf[cut:]
        return out

    def flush(self, truncated: bool = False) -> list[str]:
        """End of stream. If the brain says the reply was cut by the token cap,
        a dangling unterminated tail is dropped (never end mid-word)."""
        rest = self._buf.strip()
        self._buf = ""
        if truncated and rest and not _END.search(rest):
            rest = ""
        tail = f"{self._held} {rest}".strip()
        self._held = ""
        if tail:
            self.emitted += 1
            return [tail]
        return []


# ---------------------------------------------------------------- filters
_HALLUCINATIONS = frozenset(s.lower() for s in (
    "thank you", "thank you.", "thanks.", "thank you very much.", "thank you very much",
    "thank you so much.", "thanks for watching", "thanks for watching!", "thank you for watching",
    "thank you for watching.", "bye.", "bye bye.", "goodbye.", "good bye.", "you", "you.",
    ".", "...", "..", "subscribe.", "like and subscribe.", "please subscribe.", "thanks!",
    "thank you!", "thanks for listening.", "i'll see you later.", "see you later.",
))


def is_non_speech(t: str) -> bool:
    """Whisper describing silence ("[BLANK_AUDIO]", "(silence)", "you")."""
    t = (t or "").strip()
    if not t:
        return True
    if (t[0] == "[" and t[-1] == "]") or (t[0] == "(" and t[-1] == ")") or (t[0] == "*" and t[-1] == "*"):
        return True
    bare = t.lower().strip(" .!,?-")
    return not bare or t.lower() in _HALLUCINATIONS or bare in {"you", "thank you", "bye"}


def is_whisper_repetition(text: str) -> bool:
    """The 'It's. It's. It's.' loop whisper falls into on a short context."""
    words = [w.lower().strip(".,!?\"'") for w in (text or "").split()]
    if len(words) < 6:
        return False
    top = max(set(words), key=words.count)
    return words.count(top) >= len(words) * 0.5


def too_many_words(text: str, seconds: float, max_wps: float = 5.5) -> bool:
    """More words than a human can say in the clip: a hallucinated transcript."""
    return len((text or "").split()) > max(4, max_wps * max(seconds, 0.1))


class TranscriptDedup:
    def __init__(self, window_s: float = 3.0, clock: Callable[[], float] = time.monotonic):
        self.window_s = window_s
        self._clock = clock
        self._recent: list[tuple[str, float]] = []

    def is_duplicate(self, text: str) -> bool:
        now = self._clock()
        norm = (text or "").lower().strip().rstrip(".,!?")
        self._recent = [(t, ts) for t, ts in self._recent if now - ts < self.window_s]
        if any(t == norm for t, _ in self._recent):
            return True
        self._recent.append((norm, now))
        return False


# ---------------------------------------------------------------- endpointing
# Words a finished sentence does not end on: a phrase ending in one of these
# is still going, whatever punctuation whisper put after it.
_JOINERS = frozenset("""
a an the and or but because cause if than whether um uh er erm my your our their
i'm i'd i'll we're you're they're plus
""".split())
# ...and these only when whisper closed the fragment with a period (a
# question may well end on "about?" or "with?").
_CONTINUES = _JOINERS | frozenset("""
which who whose when while where to of for from with without about into onto
at by in on as like is are was were am kind sort
""".split())


def looks_complete(text: str) -> bool:
    """Does a (partial) transcript read as a finished turn? Used to close an
    utterance before the full silence hangover: a question or a sentence that
    ends cleanly goes now; a trailing "and", "so", "the", an ellipsis or a
    comma means the speaker is mid-thought and gets the longer hangover."""
    t = (text or "").strip()
    if not t or t.endswith(("...", "…", ",", "-", ";", ":")):
        return False
    if t[-1] not in ".?!":
        return False
    words = re.findall(r"[a-z']+", t.lower())
    if not words:
        return False
    if t[-1] in "?!":
        return words[-1] not in _JOINERS
    return words[-1] not in _CONTINUES and len(words) >= 2


# ---------------------------------------------------------------- commands
_LEAVE = re.compile(
    r"\b(?:leave|exit|drop|hang up|get out of)\b[\w\s,]{0,20}\b(?:the |this )?(?:call|meeting|room)\b"
    r"|\bhang up\b|\bend the call\b",
    re.I,
)


def is_leave_command(text: str, names: tuple[str, ...] = ("lumina",), one_to_one: bool = True) -> bool:
    """'Lumina, leave the call.' In a 1:1 the name is optional; in a group it
    must be said, so nobody else's 'I have to leave the call' dismisses the
    agent. 'Goodbye <name>' always counts."""
    t = (text or "").lower()
    if any(re.search(rf"\bgoodbye,? {re.escape(n)}\b", t) for n in names):
        return True
    if not _LEAVE.search(t):
        return False
    return one_to_one or any(re.search(rf"\b{re.escape(n)}\b", t) for n in names)
