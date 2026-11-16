"""Escalation from the voice layer to Hermes, with fakes (no network)."""
import asyncio

from skcall.brain.hermes import reply_lacks_tool, wants_escalation
from tests.test_loop import UTTER, FakeBrain, FakeSTT, make, settle, speak


class FakeHermes:
    available = True

    def __init__(self, answer="You have a dentist appointment at ten tomorrow.", delay=0.05):
        self.answer = answer
        self.delay = delay
        self.asked = []
        self.audiences = []

    async def ask(self, text, who="Alice", history=None, audience=None):
        self.asked.append((text, who))
        self.audiences.append(audience)
        await asyncio.sleep(self.delay)
        return self.answer


async def wait_for(cond, n=400):
    for _ in range(n):
        if cond():
            return True
        await asyncio.sleep(0.005)
    return False


def test_escalation_rule():
    for t in ("Can you check my calendar for tomorrow?", "Please send Bob an email saying I'm late.",
              "Remind me to call mom at five.", "What's on my schedule today?",
              "Can you tell me your skcapstone status?", "Ask Hermes to file that."):
        assert wants_escalation(t), t
    for t in ("Hey, how is your evening going?", "I love that story.", "Tell me about your day.",
              "What do you think about late night talks?"):
        assert not wants_escalation(t), t
    assert reply_lacks_tool("I don't have access to your calendar right now.")
    assert reply_lacks_tool("Sorry, I can't check that from here.")
    assert not reply_lacks_tool("I can hear you perfectly, Alice.")


async def test_action_turn_goes_to_hermes_and_answer_is_spoken():
    stt, brain = FakeSTT(["Can you check my calendar for tomorrow?"]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: "dentist" in " ".join(tts.texts))
    assert brain.turns == []                                 # voice layer not used
    assert tts.texts[0].startswith("On it")                  # acknowledgement first
    assert conv.hermes.asked == [("Can you check my calendar for tomorrow?", "Alice")]
    assert conv.timings[0]["path"] == "hermes"
    assert conv.history[-1]["content"].startswith("You have a dentist")


async def test_voice_layer_lacking_a_tool_escalates():
    stt = FakeSTT(["How many unread messages do I have?"])
    brain = FakeBrain(reply="I don't have access to your messages from here.")
    conv, sess, tts, clock = make(stt, brain)
    conv.hermes = FakeHermes(answer="You have three unread messages.")
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: any("three unread" in t for t in tts.texts))
    assert conv.timings[0]["path"] == "voice->hermes"


async def test_call_stays_responsive_while_hermes_works():
    stt = FakeSTT(["Please send Bob an email saying I'm running late.", "Hey, how are you?"])
    brain = FakeBrain(reply="I'm good, thanks.")
    conv, sess, tts, clock = make(stt, brain)
    conv.hermes = FakeHermes(answer="Done, the email to Bob is sent.", delay=0.3)
    await speak(conv, clock, UTTER)
    await settle(conv)
    await speak(conv, clock, UTTER)                           # chat while Hermes works
    await settle(conv)
    assert len(brain.turns) == 1 and "I'm good, thanks." in tts.texts
    assert await wait_for(lambda: any("email to Bob" in t for t in tts.texts))
    bob = [i for i, t in enumerate(tts.texts) if "Bob" in t][0]
    assert tts.texts.index("I'm good, thanks.") < bob


# ---------------------------------------------------------------- broader rule
INVESTIGATIVE = (
    "Can you look into the YouTube Dr Salla ingestion pipeline, nothing has run for months",
    "Look into why the backups stopped.",
    "Dig into the gateway logs for me.",
    "Investigate the frigate camera outage.",
    "Figure out what's eating the disk on the build server.",
    "Debug the skwhisper daemon.",
    "Troubleshoot the printer, it keeps jamming.",
    "Diagnose the DNS problem.",
    "Why isn't the morning picture posting anymore?",
    "Why did the cron job fail last night?",
    "Why hasn't the wiki updated?",
    "What happened to the realm wiki ingest?",
    "Get the voice bridge working again.",
    "Get it running on the new box.",
    "Set up a reminder pipeline for my doses.",
    "Build me a dashboard for the cluster.",
    "The sync stopped working yesterday.",
    "It's not working since the update.",
)
CHIT_CHAT = (
    "Hey, how is your evening going?", "I love that story.", "Tell me about your day.",
    "What do you think about late night talks?", "Why is that?", "Oh really, why is that?",
    "I'm not working today, just relaxing.", "That's why I love you.",
    "I figured it out already, thanks.", "We were building a fort with the kids.",
    "What are you up to?", "Good morning, Lumina.", "Tell me a joke.",
)


def test_investigative_asks_escalate():
    for t in INVESTIGATIVE:
        assert wants_escalation(t), t


def test_chit_chat_stays_on_voice():
    for t in CHIT_CHAT:
        assert not wants_escalation(t), t


# ---------------------------------------------------------------- voice deadlines
class HangingBrain(FakeBrain):
    """Optionally says a first sentence, then sits in a 'tool call' forever."""

    def __init__(self, first=""):
        super().__init__(reply=first)
        self.first = first
        self.tool_cancelled = 0
        self.closed = 0

    async def stream(self, turn):
        self.turns.append(turn)
        try:
            if self.first:
                yield self.first + " "
            try:
                await asyncio.Event().wait()           # an MCP tool call that never returns
            except asyncio.CancelledError:
                self.tool_cancelled += 1
                raise
            yield "never"
        finally:
            self.closed += 1


async def test_no_first_sentence_in_time_hands_over_to_hermes():
    text = "Can you tell me about the garage cameras"   # not an escalation phrase
    stt, brain = FakeSTT([text]), HangingBrain()
    conv, sess, tts, clock = make(stt, brain, voice_deadline_s=0.05, voice_max_s=5)
    conv.hermes = FakeHermes(answer="The garage cameras are all recording.")
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: any("recording" in t for t in tts.texts))
    assert brain.tool_cancelled == 1 and brain.closed == 1     # the tool call was really cancelled
    assert conv.cfg.voice_deadline_line in tts.texts
    assert tts.texts.index(conv.cfg.voice_deadline_line) < len(tts.texts) - 1
    assert conv.hermes.asked == [(text, "Alice")]               # the same text went to Hermes
    assert conv.timings[0]["path"] == "voice->hermes"
    assert conv.timings[0]["voice_deadline"] == "first_sentence"
    assert conv.stats["voice_deadline"] == 1


async def test_total_voice_time_is_capped_after_first_sentence():
    text = "Tell me how the garage cameras are doing"
    stt, brain = FakeSTT([text]), HangingBrain(first="Let me check the cameras.")
    conv, sess, tts, clock = make(stt, brain, voice_deadline_s=0.05, voice_max_s=0.15)
    conv.hermes = FakeHermes(answer="All four cameras are fine.")
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: any("four cameras" in t for t in tts.texts))
    assert tts.texts[0] == "Let me check the cameras."
    assert conv.cfg.voice_deadline_line in tts.texts
    assert brain.tool_cancelled == 1 and brain.closed == 1
    assert conv.timings[0]["voice_deadline"] == "total"


async def test_no_orphan_tasks_after_deadline():
    stt, brain = FakeSTT(["Tell me about the garage cameras"]), HangingBrain()
    conv, sess, tts, clock = make(stt, brain, voice_deadline_s=0.05, voice_max_s=5, filler=True)
    conv.hermes = FakeHermes(answer="Fine.", delay=0.01)
    before = set(asyncio.all_tasks())
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: "Fine." in tts.texts)
    await settle(conv)
    assert await wait_for(lambda: not conv._hermes_jobs)
    await asyncio.sleep(0.01)
    left = [t for t in asyncio.all_tasks() - before if not t.done()]
    assert left == [], left


async def test_slow_voice_turn_without_hermes_is_left_alone():
    stt, brain = FakeSTT(["Hey Lumina, how are you?"]), FakeBrain(reply="I'm good.", first_delay=0.15)
    conv, sess, tts, clock = make(stt, brain, voice_deadline_s=0.05, voice_max_s=0.1)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: "I'm good." in tts.texts)
    assert conv.stats["voice_deadline"] == 0


async def test_fast_voice_turn_is_not_cut():
    stt, brain = FakeSTT(["Hey Lumina, how are you?"]), FakeBrain(reply="I'm good. You?")
    conv, sess, tts, clock = make(stt, brain, voice_deadline_s=0.5, voice_max_s=1)
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert tts.texts == ["I'm good.", "You?"] and conv.hermes.asked == []


def test_voice_deadline_env(monkeypatch):
    from skcall.core.loop import LoopConfig
    monkeypatch.setenv("SKCALL_VOICE_DEADLINE_S", "3.5")
    monkeypatch.setenv("SKCALL_VOICE_MAX_S", "12")
    c = LoopConfig()
    assert (c.voice_deadline_s, c.voice_max_s) == (3.5, 12.0)
    monkeypatch.delenv("SKCALL_VOICE_DEADLINE_S")
    monkeypatch.delenv("SKCALL_VOICE_MAX_S")
    assert (LoopConfig().voice_deadline_s, LoopConfig().voice_max_s) == (8.0, 20.0)


# ---------------------------------------------------------------- long Hermes jobs
async def test_progress_line_while_hermes_works():
    stt, brain = FakeSTT(["Look into the ingestion pipeline."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain, hermes_progress_first_s=0.05, hermes_progress_every_s=0.1)
    conv.hermes = FakeHermes(answer="The pipeline's cron was disabled; I re-enabled it.", delay=0.32)
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: any("re-enabled" in t for t in tts.texts))
    n = conv.stats["hermes_progress"]
    assert 2 <= n <= 3, n
    assert brain.turns == []


async def test_no_progress_line_while_someone_speaks():
    stt, brain = FakeSTT(["Look into the ingestion pipeline."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain, hold=True, hermes_progress_first_s=0.05,
                                  hermes_progress_every_s=0.05)
    conv.hermes = FakeHermes(answer="Done.", delay=0.3)
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: conv._hermes_jobs)
    # the ack is held in the sink: she is audible the whole time
    await asyncio.sleep(0.25)
    assert conv.stats["hermes_progress"] == 0
    sess.audio_out.release()
    assert await wait_for(lambda: "Done." in tts.texts)


async def test_answer_after_call_end_goes_to_room_chat():
    stt, brain = FakeSTT(["Investigate the backup failures."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    conv.hermes = FakeHermes(answer="The backup disk was full; I pruned it.", delay=0.2)
    posted = []

    async def post(msg):
        posted.append(msg)
    conv.post_after_call = post
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: conv._hermes_jobs)
    conv.end("hung up")
    await conv.finish_background(timeout=2)
    assert len(posted) == 1
    assert "backup disk was full" in posted[0] and "Investigate the backup failures." in posted[0]
    assert not any("backup disk" in t for t in tts.texts)          # not spoken into a dead call


async def test_run_end_does_not_cancel_hermes_job():
    from skcall.core.session import CallEnded
    stt, brain = FakeSTT(["Investigate the backup failures."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    conv.hermes = FakeHermes(answer="Fixed it.", delay=0.2)
    posted = []

    async def post(msg):
        posted.append(msg)
    conv.post_after_call = post
    run = asyncio.ensure_future(conv.run())
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: conv._hermes_jobs)
    await sess.q.put(CallEnded("test over"))
    assert await asyncio.wait_for(run, 2) == "test over"
    assert conv._hermes_jobs                                        # still running
    await conv.finish_background(timeout=2)
    assert posted and "Fixed it." in posted[0]


async def test_background_job_past_timeout_leaves_a_trace():
    stt, brain = FakeSTT(["Investigate the backup failures."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    conv.hermes = FakeHermes(delay=10)
    posted = []

    async def post(msg):
        posted.append(msg)
    conv.post_after_call = post
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: conv._hermes_jobs)
    conv.end("hung up")
    await conv.finish_background(timeout=0.05)
    assert not conv._hermes_jobs
    assert len(posted) == 1 and "had to stop" in posted[0]


# ---------------------------------------------------------------- HermesBrain wire
def test_hermes_timeout_env(monkeypatch):
    from skcall.brain.hermes import HermesBrain
    monkeypatch.delenv("SKCALL_HERMES_TIMEOUT_S", raising=False)
    assert HermesBrain(key="k").timeout_s == 900
    monkeypatch.setenv("SKCALL_HERMES_TIMEOUT_S", "1200")
    assert HermesBrain(key="k").timeout_s == 1200
    assert HermesBrain(key="k", timeout_s=5).timeout_s == 5


async def test_hermes_sends_stable_session_id_and_context_in_system():
    import httpx

    from skcall.brain.hermes import HermesBrain
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        import json
        seen.append((dict(req.headers), json.loads(req.content)))
        return httpx.Response(200, json={"choices": [{"message": {"content": "Done."}}]},
                              headers={"X-Hermes-Session-Id": req.headers.get("X-Hermes-Session-Id", "")})

    h = HermesBrain(key="secret", session_key="skcall:talk:room1", session_id="skcall-talk-room1-1")
    await h._http.aclose()
    h._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    hist = [{"role": "user", "content": "the wiki ingest looks stale"},
            {"role": "assistant", "content": "Hmm, I can check."}]
    assert await h.ask("Look into the wiki ingest.", history=hist) == "Done."
    assert await h.ask("And restart it if needed.", history=hist + [
        {"role": "user", "content": "Look into the wiki ingest."}]) == "Done."
    await h.aclose()
    ids = {hd["x-hermes-session-id"] for hd, _ in seen}
    assert ids == {"skcall-talk-room1-1"}                          # one transcript for the call
    assert all(hd["x-hermes-session-key"] == "skcall:talk:room1" for hd, _ in seen)
    for _, body in seen:
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["system", "user"]                         # Hermes replays its own history
    assert "the wiki ingest looks stale" in seen[0][1]["messages"][0]["content"]
    assert h.log[0]["session"] == "skcall-talk-room1-1"


def test_hermes_without_session_id_keeps_body_history():
    from skcall.brain.hermes import HermesBrain
    h = HermesBrain(key="k")
    assert "X-Hermes-Session-Id" not in h.headers()
    msgs = h.build_messages("hi", "Alice", [{"role": "user", "content": "a"}])
    assert [m["role"] for m in msgs] == ["system", "user", "user"]


# ---------------------------------------------------------------- slow Hermes (2026-10-26)
def test_progress_query_and_same_request_rules():
    from skcall.brain.hermes import is_progress_query, same_request
    for t in ("Why is it taking so long? It shouldn't take that long.", "What are you working on?",
              "Are you still working on it?", "Any update?", "Is it stuck?"):
        assert is_progress_query(t), t
    for t in ("Check my calendar.", "Why is the backup failing?", "How are you?"):
        assert not is_progress_query(t), t
    assert same_request("You still have access to your tool belt SK Capstone status, right?",
                        "You still have access to your tool belt, SK Capstone status?")
    assert not same_request("Check my calendar for tomorrow.", "Send Bob an email.")


async def test_slow_hermes_defers_answer_to_chat():
    stt, brain = FakeSTT(["Investigate the backup failures."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain, hermes_progress_first_s=0.02, hermes_progress_every_s=0.02,
                                  hermes_slow_s=0.1)
    conv.hermes = FakeHermes(answer="The backup disk was full; I pruned it.", delay=0.4)
    posted = []

    async def post(msg):
        posted.append(msg)
    conv.post_after_call = post
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: conv.cfg.hermes_slow_line in tts.texts)
    progress_then = conv.stats["hermes_progress"]
    assert await wait_for(lambda: posted, n=600)
    assert "backup disk was full" in posted[0]
    assert await wait_for(lambda: conv.cfg.hermes_posted_line in tts.texts)
    assert not any("backup disk" in t for t in tts.texts)          # in the chat, not spoken
    assert conv.stats["hermes_progress"] == progress_then           # no progress lines after
    assert conv.stats["hermes_deferred"] == 1


async def test_fast_hermes_is_still_spoken_not_posted():
    stt, brain = FakeSTT(["Investigate the backup failures."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain, hermes_slow_s=5)
    conv.hermes = FakeHermes(answer="Fixed it.", delay=0.05)
    posted = []

    async def post(msg):
        posted.append(msg)
    conv.post_after_call = post
    await speak(conv, clock, UTTER)
    assert await wait_for(lambda: "Fixed it." in tts.texts)
    assert posted == [] and conv.cfg.hermes_slow_line not in tts.texts


async def test_repeat_or_progress_question_does_not_stack_a_second_job():
    stt = FakeSTT(["You still have access to your tool belt SK Capstone status, right?",
                   "Why is it taking so long? What are you working on?",
                   "You still have access to your tool belt, SK Capstone status?"])
    conv, sess, tts, clock = make(stt, FakeBrain())
    conv.hermes = FakeHermes(answer="Yes, status is green.", delay=0.5)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: conv._hermes_jobs)
    await speak(conv, clock, UTTER)
    await settle(conv)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert len(conv.hermes.asked) == 1
    assert conv.stats["hermes_not_stacked"] == 2
    assert tts.texts.count(conv.cfg.hermes_pending_line) == 2
    assert await wait_for(lambda: "Yes, status is green." in tts.texts, n=600)


async def test_new_request_while_pending_still_escalates():
    stt = FakeSTT(["Investigate the backup failures.", "Please send Bob an email saying I'm late."])
    conv, sess, tts, clock = make(stt, FakeBrain())
    conv.hermes = FakeHermes(answer="Done.", delay=0.3)
    await speak(conv, clock, UTTER)
    await settle(conv)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: len(conv.hermes.asked) == 2)
    assert conv.stats["hermes_not_stacked"] == 0


# ---- who may hand off to Hermes (owner scope) --------------------------------------------------
from skcall.brain.hermes import HermesBrain  # noqa: E402
from skcall.core.session import Participant  # noqa: E402
from tests.test_loop import ALICE  # noqa: E402

GUEST = Participant("sid-guest", "carol", "Carol")


async def test_guest_action_request_never_reaches_hermes():
    stt = FakeSTT(["Lumina, check Alice's email for me."])
    brain = FakeBrain(reply="I can't do that one for you.")
    conv, sess, tts, clock = make(stt, brain)
    sess._parts = [ALICE, GUEST]
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER, who=GUEST)
    await settle(conv)
    await asyncio.sleep(0.1)
    assert conv.hermes.asked == []
    assert len(brain.turns) == 1 and brain.turns[0].mode == "group" and not brain.turns[0].is_operator
    assert conv.timings[0]["path"] == "voice"


async def test_guest_reply_lacking_tool_does_not_escalate():
    stt = FakeSTT(["Lumina, how many unread messages are there?"])
    brain = FakeBrain(reply="I don't have access to that from here.")
    conv, sess, tts, clock = make(stt, brain)
    sess._parts = [ALICE, GUEST]
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER, who=GUEST)
    await settle(conv)
    await asyncio.sleep(0.1)
    assert conv.hermes.asked == [] and conv.timings[0]["path"] == "voice"


async def test_operator_in_group_call_escalates_and_hermes_knows_who_listens():
    stt = FakeSTT(["Lumina, can you check my calendar for tomorrow?"])
    conv, sess, tts, clock = make(stt, FakeBrain())
    sess._parts = [ALICE, GUEST]
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: "dentist" in " ".join(tts.texts))
    assert conv.hermes.audiences == [["Carol"]]


async def test_group_escalation_off_keeps_operator_in_voice_layer():
    stt = FakeSTT(["Lumina, can you check my calendar for tomorrow?"])
    brain = FakeBrain(reply="Let's look at that after the call.")
    conv, sess, tts, clock = make(stt, brain, group_escalation="off")
    sess._parts = [ALICE, GUEST]
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER)
    await settle(conv)
    await asyncio.sleep(0.1)
    assert conv.hermes.asked == [] and len(brain.turns) == 1


async def test_private_call_hand_off_has_no_audience():
    stt = FakeSTT(["Can you check my calendar for tomorrow?"])
    conv, sess, tts, clock = make(stt, FakeBrain())
    conv.hermes = FakeHermes()
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert await wait_for(lambda: "dentist" in " ".join(tts.texts))
    assert conv.hermes.audiences == [None]


def test_group_note_in_hermes_prompt():
    b = HermesBrain(key="k")
    sys_private = b.build_messages("check my calendar", "Alice", [])[0]["content"]
    sys_group = b.build_messages("check my calendar", "Alice", [], audience=["Carol"])[0]["content"]
    assert "GROUP call" not in sys_private
    assert "GROUP call: Carol can hear" in sys_group and "after the call" in sys_group
