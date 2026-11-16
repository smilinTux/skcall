"""The agent's video: static portrait (P5a), idle loop + lip-sync compositor (P5c).

Backend-agnostic (numpy, OpenCV, aiohttp for the service). A backend takes a
core VideoSource from `Avatar.source` and wraps its AudioSink with
`Avatar.wrap_sink()`.

Config (environment, see VideoConfig.from_env):
  SKCALL_VIDEO            1/0: publish video at all (default 1)
  SKCALL_VIDEO_MODE       static | animated (default animated: idle loop, plus
                          lips when the service is configured)
  SKCALL_VIDEO_SIZE       WxH, default 560x720
  SKCALL_VIDEO_FPS        animated fps (25); SKCALL_VIDEO_STATIC_FPS (5)
  SKCALL_AVATAR_IMAGE     portrait path (default ~/.skcapstone/agents/<agent>/avatar/portrait.png)
  SKCALL_LIPSYNC_URL      ws://host:port/v1/lipsync (unset: no lips)
  SKCALL_LIPSYNC_TOKEN    bearer token for it
  SKCALL_LIPSYNC_AVATAR   avatar id on the service (default: the agent name)
  SKCALL_AUDIO_HOLD_MS    hold the first sentence up to this for frames (300)
  SKCALL_AV_OFFSET_MS     shift the mouth later (+) or earlier (-) vs the audio (0)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from skcall.video.avatar import IdleLoop, StaticPortrait, eye_boxes, load_portrait, portrait_path
from skcall.video.compositor import Compositor, LipsyncSink
from skcall.video.lipsync import LipsyncClient

LOG = logging.getLogger("skcall.video")


def _flag(v: str) -> bool:
    return v.strip().lower() not in ("0", "false", "no", "off", "")


@dataclass
class VideoConfig:
    enabled: bool = True
    mode: str = "animated"            # static | animated
    width: int = 560
    height: int = 720
    fps: float = 25.0
    static_fps: float = 5.0
    image: str = ""
    lipsync_url: str = ""
    lipsync_token: str = ""
    avatar: str = "lumina"
    audio_hold_ms: int = 300
    av_offset_ms: int = 0

    @classmethod
    def from_env(cls, agent: str = "lumina") -> "VideoConfig":
        e = os.environ.get
        w, _, h = e("SKCALL_VIDEO_SIZE", "560x720").partition("x")
        return cls(enabled=_flag(e("SKCALL_VIDEO", "1")),
                   mode=e("SKCALL_VIDEO_MODE", "animated").strip().lower(),
                   width=int(w), height=int(h or 720),
                   fps=float(e("SKCALL_VIDEO_FPS", "25")),
                   static_fps=float(e("SKCALL_VIDEO_STATIC_FPS", "5")),
                   image=portrait_path(agent),
                   lipsync_url=e("SKCALL_LIPSYNC_URL", ""),
                   lipsync_token=e("SKCALL_LIPSYNC_TOKEN", ""),
                   avatar=e("SKCALL_LIPSYNC_AVATAR", agent),
                   audio_hold_ms=int(e("SKCALL_AUDIO_HOLD_MS", "300")),
                   av_offset_ms=int(e("SKCALL_AV_OFFSET_MS", "0")))


class Avatar:
    """Everything video for one call: the source the backend publishes, and
    (animated + service) the lip-sync client and audio wrapper."""

    def __init__(self, cfg: VideoConfig, source, compositor: Compositor | None = None,
                 client: LipsyncClient | None = None):
        self.cfg = cfg
        self.source = source
        self.compositor = compositor
        self.client = client
        self.sink: LipsyncSink | None = None

    @classmethod
    def build(cls, cfg: VideoConfig) -> "Avatar | None":
        """None when video is off or the portrait is unusable (voice only)."""
        if not cfg.enabled:
            return None
        img = load_portrait(cfg.image, cfg.width, cfg.height)
        if img is None:
            return None
        if cfg.mode == "static":
            LOG.info("video: static portrait %dx%d @%.0f fps", cfg.width, cfg.height, cfg.static_fps)
            return cls(cfg, StaticPortrait(img, cfg.static_fps))
        idle = None
        try:
            idle = IdleLoop(img, eye_boxes(cfg.image), fps=cfg.fps)
        except Exception as e:
            LOG.warning("idle loop unavailable (%r); still portrait at %.0f fps", e, cfg.fps)
        comp = Compositor(img, idle, fps=cfg.fps, av_offset_s=cfg.av_offset_ms / 1000)
        client = None
        if cfg.lipsync_url:
            def _ready(c: LipsyncClient) -> None:
                comp.set_crop(c.crop, c.frame_size)
            client = LipsyncClient(cfg.lipsync_url, cfg.lipsync_token, avatar=cfg.avatar,
                                   fps=int(cfg.fps), on_ready=_ready)
        LOG.info("video: animated %dx%d @%.0f fps, idle=%s blink=%s lipsync=%s", cfg.width, cfg.height,
                 cfg.fps, idle is not None, bool(idle and idle.can_blink), bool(client))
        return cls(cfg, comp, comp, client)

    def start(self) -> None:
        if self.client is not None:
            self.client.start()

    def wrap_sink(self, inner):
        """The AudioSink the conversation should write to."""
        if self.compositor is None or self.client is None:
            return inner
        self.sink = LipsyncSink(inner, self.compositor, self.client,
                                hold_s=self.cfg.audio_hold_ms / 1000)
        return self.sink

    async def prerender(self, clips) -> int:
        if self.sink is None or self.client is None:
            return 0
        if not await self.client.wait_ready(10):
            return 0
        return await self.sink.prerender(list(clips))

    def summary(self) -> dict:
        out: dict = {"mode": self.cfg.mode}
        if self.compositor is not None:
            out["compositor"] = self.compositor.stats.summary()
        if self.client is not None:
            out["lipsync"] = dict(self.client.stats, available=self.client.available)
        if self.sink is not None:
            out["sink"] = dict(self.sink.stats)
            if self.sink.holds:
                h = sorted(self.sink.holds)
                out["hold_ms_p50"] = round(h[len(h) // 2] * 1000)
                out["hold_ms_max"] = round(h[-1] * 1000)
        return out

    async def aclose(self) -> None:
        if self.sink is not None and self.sink._worker is not None:
            self.sink._worker.cancel()
        if self.client is not None:
            await self.client.aclose()


__all__ = ["Avatar", "VideoConfig", "Compositor", "LipsyncSink", "LipsyncClient", "IdleLoop",
           "StaticPortrait", "load_portrait"]
