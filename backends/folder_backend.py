from __future__ import annotations

import io
import time
from pathlib import Path

from PIL import Image

from agent.geometry import CameraModel
from backends.base import MoveResult


class FolderBackend:
    def __init__(self, folder: str, camera: CameraModel):
        self.camera = camera
        self.photos = sorted(p for p in Path(folder).iterdir()
                             if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not self.photos:
            raise SystemExit(f"No .jpg or .png photos in {folder}")
        self.index = 0

    def capture_jpeg(self) -> bytes:
        path = self.photos[self.index % len(self.photos)]
        if path.suffix.lower() == ".png":
            buf = io.BytesIO()
            Image.open(path).convert("RGB").save(buf, format="JPEG")
            return buf.getvalue()
        return path.read_bytes()

    def turn(self, degrees: float) -> MoveResult:
        print(f"   [motor] turn {degrees:+.0f}°")
        self.index += 1
        return MoveResult(done=degrees)

    def forward(self, cm: float) -> MoveResult:
        print(f"   [motor] forward {cm:.1f} cm")
        return MoveResult(done=cm)

    def halt(self) -> None:
        print("   [motor] halt")

    def idle(self, seconds: float) -> bool:
        time.sleep(seconds)
        return True
