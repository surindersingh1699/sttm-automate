"""Audio playback drivers for the eval harness.

YtDlpAudioFeeder  — headless mode: yt-dlp downloads audio once (cached), feeds
                    it into AudioCapture.push_external at 1× wall-clock speed.

PlaywrightYouTubeDriver — live mode: Playwright opens YouTube in a visible
                          Chromium window, starts playback from a given offset,
                          handles skip-ad. Audio is captured via BlackHole
                          loopback — the user hears kirtan and sees STTM update.
"""
from __future__ import annotations

import asyncio
import re
import time
from pathlib import Path

import numpy as np

_CACHE_DIR = Path(__file__).parent.parent.parent / "tests" / "eval" / "cache" / "audio"
_SR = 16_000           # target sample rate for pipeline

# YouTube video IDs are 11 alphanumeric/dash/underscore chars.
_VIDEO_ID_RE = re.compile(r'^[A-Za-z0-9_-]{11}$')


def _validate_video_id(video_id: str) -> str:
    if not _VIDEO_ID_RE.match(video_id):
        raise ValueError(f"Invalid YouTube video_id: {video_id!r}")
    return video_id


# ── yt-dlp feeder (headless) ────────────────────────────────────────────────

class YtDlpAudioFeeder:
    """Download once, feed at 1× pace via AudioCapture.push_external.

    Usage:
        feeder = YtDlpAudioFeeder(video_id, audio_t0, audio_t_end)
        audio_np = await feeder.load()           # downloads if not cached
        await feeder.feed(capture, stop_event)   # blocks until done or stopped
    """

    def __init__(
        self,
        video_id: str,
        audio_t0: float,
        audio_t_end: float,
        cache_dir: Path = _CACHE_DIR,
        chunk_duration_s: float = 0.5,
    ):
        self.video_id = _validate_video_id(video_id)
        self.audio_t0 = audio_t0
        self.audio_t_end = audio_t_end
        self.cache_dir = cache_dir
        self.chunk_s = chunk_duration_s
        self._audio: np.ndarray | None = None

    @property
    def cached_path(self) -> Path:
        return self.cache_dir / f"{self.video_id}.opus"

    async def load(self) -> np.ndarray:
        """Return the session audio slice as float32 16 kHz mono numpy array."""
        if self._audio is not None:
            return self._audio

        opus_path = await self._ensure_cached()
        self._audio = await asyncio.to_thread(self._decode_slice, opus_path)
        return self._audio

    async def _ensure_cached(self) -> Path:
        # Find any already-cached file for this video_id (yt-dlp picks the ext)
        existing = list(self.cache_dir.glob(f"{self.video_id}.*"))
        if existing:
            return existing[0]

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[YtDlpFeeder] Downloading {self.video_id} via yt-dlp…")

        url = f"https://www.youtube.com/watch?v={self.video_id}"
        out_template = str(self.cache_dir / f"{self.video_id}.%(ext)s")

        # asyncio.create_subprocess_exec avoids shell injection; video_id validated above.
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--quiet",
            "-x",
            "--audio-format", "opus",
            "--audio-quality", "0",
            "-o", out_template,
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp failed for {self.video_id}: {stderr.decode()[:500]}"
            )

        candidates = list(self.cache_dir.glob(f"{self.video_id}.*"))
        if not candidates:
            raise FileNotFoundError(f"yt-dlp produced no output for {self.video_id}")
        return candidates[0]

    def _decode_slice(self, audio_path: Path) -> np.ndarray:
        """Decode audio_path and return the [audio_t0, audio_t_end] slice at 16 kHz."""
        try:
            import soundfile as sf
            data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        except Exception:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(audio_path))
            seg = seg.set_channels(1).set_frame_rate(_SR)
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            return self._trim(samples, _SR)

        if data.ndim == 2:
            data = data.mean(axis=1)

        if sr != _SR:
            import librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=_SR)
            sr = _SR

        return self._trim(data, sr)

    def _trim(self, audio: np.ndarray, sr: int) -> np.ndarray:
        start_n = int(self.audio_t0 * sr)
        end_n = int(self.audio_t_end * sr) if self.audio_t_end > 0 else len(audio)
        end_n = min(end_n, len(audio))
        return audio[start_n:end_n].astype(np.float32)

    async def feed(
        self,
        capture,   # AudioCapture instance
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Feed audio into capture.push_external at 1× wall-clock speed."""
        audio = await self.load()
        chunk_n = int(self.chunk_s * _SR)
        pos = 0
        t0 = time.monotonic()

        while pos < len(audio):
            if stop_event and stop_event.is_set():
                break
            chunk = audio[pos:pos + chunk_n]
            capture.push_external(chunk)
            pos += chunk_n

            elapsed = time.monotonic() - t0
            expected = pos / _SR
            sleep_s = expected - elapsed
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)


# ── Playwright YouTube driver (live) ────────────────────────────────────────

class PlaywrightYouTubeDriver:
    """Open YouTube in visible Chromium and play from a given offset.

    The operator hears kirtan through speakers (routed via Multi-Output Device
    → BlackHole → AudioCapture). They see STTM Desktop update live.

    Call:
        driver = PlaywrightYouTubeDriver(video_id, audio_t0)
        await driver.open()     # opens browser, seeks, starts playing
        t0 = driver.play_start_wall  # monotonic time of confirmed playback start
        ...
        await driver.close()
    """

    def __init__(
        self,
        video_id: str,
        audio_t0: float,
        headless: bool = False,
    ):
        self.video_id = _validate_video_id(video_id)
        self.audio_t0 = audio_t0
        self.headless = headless
        self._pw = None
        self._browser = None
        self._page = None
        self.play_start_wall: float | None = None

    async def open(self):
        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.headless,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        ctx = await self._browser.new_context(permissions=["notifications"])
        self._page = await ctx.new_page()

        offset_s = max(0, int(self.audio_t0))
        # Embed URL suppresses most ads and allows autoplay without user gesture
        url = (
            f"https://www.youtube.com/embed/{self.video_id}"
            f"?autoplay=1&start={offset_s}&mute=0"
        )
        print(f"[PlaywrightYT] Opening {url}")
        await self._page.goto(url)
        await self._page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(2)

        # Dismiss consent / skip-ad overlays
        for selector in [
            'button[aria-label*="Accept"]',
            'button:has-text("Accept all")',
            'button:has-text("I agree")',
            ".ytp-ad-skip-button",
        ]:
            try:
                btn = self._page.locator(selector).first
                if await btn.is_visible(timeout=800):
                    await btn.click()
                    await asyncio.sleep(0.5)
            except Exception:
                pass

        # Ensure video is playing
        try:
            is_playing = await self._page.evaluate("""() => {
                const v = document.querySelector('video');
                return v && !v.paused && !v.ended && v.readyState > 2;
            }""")
            if not is_playing:
                await self._page.evaluate("""() => {
                    const v = document.querySelector('video');
                    if (v) v.play();
                }""")
        except Exception:
            pass

        self.play_start_wall = time.monotonic()
        print(f"[PlaywrightYT] Playback confirmed (video offset {self.audio_t0:.1f}s)")

    async def skip_ads(self):
        """Click skip-ad if visible. Call periodically during long sessions."""
        if not self._page:
            return
        for selector in [".ytp-ad-skip-button", ".ytp-skip-ad-button",
                          'button[aria-label*="Skip"]']:
            try:
                btn = self._page.locator(selector).first
                if await btn.is_visible(timeout=300):
                    await btn.click()
                    print("[PlaywrightYT] Skipped ad")
                    return
            except Exception:
                pass

    async def close(self):
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
        self._browser = None
        self._page = None
        self._pw = None
