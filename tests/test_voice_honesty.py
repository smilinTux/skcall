from skcall.brain.hermes import reply_lacks_tool
from skcall.brain.voice_engine import HONESTY_NOTE


def test_honesty_phrase_hands_off_to_hermes():
    # The exact phrase the honesty rule tells the voice layer to say must trip
    # the hand-off, or the rule turns a lie into a dead end.
    assert '"I can\'t check that from here"' in HONESTY_NOTE
    assert reply_lacks_tool("I can't check that from here")


def test_honesty_note_lists_tools():
    text = HONESTY_NOTE.format(tools="search_memory, read_reflection")
    assert "search_memory, read_reflection" in text
    assert "never invent" in text.lower()
