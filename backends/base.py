from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent.geometry import CameraModel


@dataclass
class MoveResult:
    done: float
    blocked: bool = False
    front_cm: float | None = None


class RoverBackend(Protocol):
    camera: CameraModel

    def capture_jpeg(self) -> bytes: ...
    def turn(self, degrees: float) -> MoveResult: ...
    def forward(self, cm: float) -> MoveResult: ...
    def halt(self) -> None: ...
    def idle(self, seconds: float) -> bool:
        ...
