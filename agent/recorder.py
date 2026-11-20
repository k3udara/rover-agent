from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

FRAMES_PER_SECOND = 10
GIF_WIDTH = 360
FFMPEG_TIMEOUT_SECONDS = 120


class TripRecorder:
    def __init__(self, folder: Path, fps: int = FRAMES_PER_SECOND):
        self.folder = Path(folder)
        self.frames_dir = self.folder / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_count = 0

    def add(self, image: Image.Image, hold_seconds: float = 0.0) -> None:
        repeats = max(1, round(hold_seconds * self.fps))
        for _ in range(repeats):
            self.frame_count += 1
            image.save(self.frames_dir / f"{self.frame_count:05d}.jpg", quality=88)

    def save_plan(self, call: int, image_sent: bytes, record: dict) -> None:
        (self.folder / f"plan_{call}.jpg").write_bytes(image_sent)
        (self.folder / f"plan_{call}.json").write_text(json.dumps(record, indent=2))

    def finish(self, outcome: dict, rename_to: Path | None = None) -> Path | None:
        (self.folder / "trip.json").write_text(json.dumps(outcome, indent=2))
        video = self._encode() if self.frame_count else None
        shutil.rmtree(self.frames_dir, ignore_errors=True)
        if rename_to and rename_to != self.folder and not rename_to.exists():
            self.folder.rename(rename_to)
            video = rename_to / video.name if video else None
            self.folder = rename_to
        return video

    def _encode(self) -> Path:
        ffmpeg = shutil.which("ffmpeg") or next(
            (p for p in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg") if Path(p).exists()), None)
        if ffmpeg:
            video = self.folder / "trip.mp4"
            for codec in (["-c:v", "libx264", "-pix_fmt", "yuv420p"], ["-c:v", "mpeg4", "-q:v", "3"]):
                done = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(self.fps),
                     "-i", str(self.frames_dir / "%05d.jpg"), *codec,
                     "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(video)],
                    capture_output=True, timeout=FFMPEG_TIMEOUT_SECONDS)
                if done.returncode == 0:
                    return video
        return self._encode_gif()

    def _encode_gif(self) -> Path:
        frames = []
        for path in sorted(self.frames_dir.glob("*.jpg")):
            img = Image.open(path)
            frames.append(img.resize((GIF_WIDTH, round(GIF_WIDTH * img.height / img.width))))
        gif = self.folder / "trip.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=1000 // self.fps, loop=0)
        return gif
