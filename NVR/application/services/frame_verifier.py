"""Confirm a stream really delivers video by decoding one frame with ffmpeg.

An ffmpeg subprocess is isolated (a hung decoder cannot stall the event loop)
and killable. Two bounds keep it polite: a global concurrency cap, and a
per-host cap because a recorder serves every channel from one box and limits
concurrent RTSP sessions.
"""

import asyncio
import contextlib
import shutil
from typing import Dict, NamedTuple, Optional
from urllib.parse import urlsplit

from .discovery_adapter import redact


class VerifyResult(NamedTuple):
    ok: bool
    error: Optional[str] = None


class FrameVerifier:
    def __init__(self, ffmpeg_bin: str = "ffmpeg", timeout_s: float = 12.0,
                 max_concurrency: int = 4, per_host: int = 2):
        self._ffmpeg = ffmpeg_bin
        self._timeout_s = float(timeout_s)
        self._global = asyncio.Semaphore(max_concurrency)
        self._per_host_limit = per_host
        self._per_host: Dict[str, asyncio.Semaphore] = {}

    def available(self) -> bool:
        return shutil.which(self._ffmpeg) is not None

    def _command(self, url: str):
        cmd = [self._ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
               "-nostats", "-progress", "pipe:1"]
        if url.lower().startswith(("rtsp://", "rtsps://")):
            # Socket I/O timeout (microseconds, ffmpeg >= 5). The asyncio
            # deadline below is the hard guarantee; this just fails faster.
            cmd += ["-rtsp_transport", "tcp", "-timeout", str(int(self._timeout_s * 1_000_000))]
        cmd += ["-i", url, "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"]
        return cmd

    async def verify(self, url: str) -> VerifyResult:
        host = (urlsplit(url).hostname or "").lower()
        host_sem = self._per_host.setdefault(host, asyncio.Semaphore(self._per_host_limit))
        async with self._global, host_sem:
            return await self._run(url)

    async def _run(self, url: str) -> VerifyResult:
        """Success requires exit code 0 AND a positive frame count on stdout."""
        try:
            proc = await asyncio.create_subprocess_exec(
                *self._command(url),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return VerifyResult(False, "ffmpeg not found: {}".format(self._ffmpeg))

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout_s)
        except asyncio.TimeoutError:
            await _kill(proc)
            return VerifyResult(False, "no frame within {:.0f}s".format(self._timeout_s))
        except asyncio.CancelledError:
            await _kill(proc)
            raise

        frames = [line.partition(b"=")[2].strip() for line in stdout.splitlines()
                  if line.startswith(b"frame=")]
        if proc.returncode == 0 and any(n.isdigit() and int(n) > 0 for n in frames):
            return VerifyResult(True)
        if proc.returncode == 0:
            return VerifyResult(False, "stream ended without a decoded frame")
        return VerifyResult(False, _last_line(stderr, url) or "ffmpeg exit {}".format(proc.returncode))


async def _kill(proc) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.kill()
    # Reap it: an unwaited child would linger as a zombie.
    with contextlib.suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=5.0)


def _last_line(stderr: bytes, url: str) -> str:
    """Last ffmpeg error line, with the URL's credentials removed."""
    lines = [line.strip() for line in (stderr or b"").decode("utf-8", "replace").splitlines() if line.strip()]
    return lines[-1].replace(url, redact(url))[:300] if lines else ""
