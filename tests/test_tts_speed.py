from skcall.speech import F5TTS, tts_speed


def test_tts_speed_default_and_parse():
    assert tts_speed({}) == 1.0
    assert tts_speed({"SKCALL_TTS_SPEED": "1.5"}) == 1.5
    assert tts_speed({"SKCALL_TTS_SPEED": "junk"}) == 1.0


def test_tts_speed_clamped():
    assert tts_speed({"SKCALL_TTS_SPEED": "9"}) == 2.0
    assert tts_speed({"SKCALL_TTS_SPEED": "0.1"}) == 0.5


def test_speed_only_sent_when_not_default():
    assert F5TTS(url="http://x/v1/audio/speech", model="tts-1", speed=1.0).speed == 1.0
    assert F5TTS(url="http://x/v1/audio/speech", model="tts-1", speed=1.5).speed == 1.5
