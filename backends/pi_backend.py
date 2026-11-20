"""Raspberry Pi rover: Pi Camera, plus a microcontroller over USB serial that drives and owns safety.

Serial protocol (115200 baud, one line each way):
  T <deg>   ->  OK T <deg_done>                        turn, positive = right
  F <cm>    ->  OK F <cm_done> <front_cm>  or  BLOCKED F <cm_done> <front_cm>
  D         ->  D <front_cm>
  S         ->  OK S                                    stop now
"""
from __future__ import annotations

import io
import time

import serial
from picamera2 import Picamera2

from agent.geometry import CameraModel
from backends.base import MoveResult

DEFAULT_CAMERA = CameraModel(hfov_deg=102, vfov_deg=67, height_cm=9.0, tilt_down_deg=15.0)


class PiBackend:
    def __init__(self, port: str = "/dev/ttyACM0", camera: CameraModel = DEFAULT_CAMERA,
                 size: tuple[int, int] = (1024, 768)):

        self.camera = camera
        self.ser = serial.Serial(port, 115200, timeout=10)
        time.sleep(2)
        self.ser.reset_input_buffer()
        self.cam = Picamera2()
        self.cam.configure(self.cam.create_still_configuration(main={"size": size}))
        self.cam.start()
        time.sleep(1)

    def _cmd(self, line: str) -> list[str]:
        self.ser.write((line + "\n").encode())
        reply = self.ser.readline().decode().strip()
        if not reply:
            self.halt()
            raise RuntimeError(f"No reply from motor board to '{line}'")
        return reply.split()

    def capture_jpeg(self) -> bytes:
        buf = io.BytesIO()
        self.cam.capture_file(buf, format="jpeg")
        return buf.getvalue()

    def turn(self, degrees: float) -> MoveResult:
        parts = self._cmd(f"T {degrees:.1f}")
        return MoveResult(done=float(parts[2]))

    def forward(self, cm: float) -> MoveResult:
        parts = self._cmd(f"F {cm:.1f}")
        front = float(parts[3]) if len(parts) > 3 else None
        return MoveResult(done=float(parts[2]), blocked=parts[0] == "BLOCKED", front_cm=front)

    def halt(self) -> None:
        try:
            self.ser.write(b"S\n")
            self.ser.readline()
        except Exception:
            pass

    def idle(self, seconds: float) -> bool:
        time.sleep(seconds)
        return True
