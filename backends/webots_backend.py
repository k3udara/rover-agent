from __future__ import annotations

import io
import math
from pathlib import Path

from PIL import Image

from agent.geometry import CameraModel, vfov_from_hfov
from agent.maze import CELL, build
from backends.base import MoveResult

WHEEL_RADIUS_M = 0.0205
AXLE_M = 0.052
SPEED = 4.0
IR_BLOCKED = 150.0
CAMERA_HEIGHT_CM = 2.8

OVERHEAD_IMAGE_FILE = Path(__file__).resolve().parents[1] / "runs" / "webots_overhead.jpg"
OVERHEAD_IMAGE_PX = 640
OVERHEAD_CAMERA_FOV = 0.6
OVERHEAD_MARGIN_M = 0.1
OVERHEAD_SETTLE_SECONDS = 0.15
OVERHEAD_FIRST_PICTURE_TRIES = 15
OVERHEAD_DRIVE_HOP_CM = 4.0
PATH_LINE_WIDTH_M = 0.008
PATH_LINE_Z = 0.002
PATH_DEF = "PLANNED_PATH"


def overhead_span_m(lay) -> float:
    return max(lay.rows, lay.cols) * CELL + OVERHEAD_MARGIN_M


def overhead_camera_height(span_m: float) -> float:
    return (span_m / 2) / math.tan(OVERHEAD_CAMERA_FOV / 2)


class WebotsBackend:
    def __init__(self, robot):
        self.robot = robot
        self.ts = int(robot.getBasicTimeStep())
        self.left = robot.getDevice("left wheel motor")
        self.right = robot.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.lenc = robot.getDevice("left wheel sensor")
        self.renc = robot.getDevice("right wheel sensor")
        self.lenc.enable(self.ts)
        self.renc.enable(self.ts)
        self.front_ir = [robot.getDevice(n) for n in ("ps0", "ps7")]
        for s in self.front_ir:
            s.enable(self.ts)
        self.cam = robot.getDevice("camera")
        self.cam.enable(self.ts)
        robot.step(self.ts)

        node = robot.getSelf() if hasattr(robot, "getSelf") else None
        if node is not None:
            self._node = node
            self.true_pose = self._true_pose

        self.lay = build()
        self.overhead_span_m = overhead_span_m(self.lay)
        self.overhead_drive_hop_cm = OVERHEAD_DRIVE_HOP_CM
        self.overhead_scene = ("A camera looks straight down on a 3D model. The light grey raised strips are "
                               "walls. The rover is the small round robot.")

        w, h = self.cam.getWidth(), self.cam.getHeight()
        hfov = math.degrees(self.cam.getFov())
        self.camera = CameraModel(hfov_deg=hfov, vfov_deg=vfov_from_hfov(hfov, w, h),
                                  height_cm=CAMERA_HEIGHT_CM, tilt_down_deg=0.0)

    def _true_pose(self) -> tuple[float, float, float]:
        x, y, _ = self._node.getPosition()
        R = self._node.getOrientation()
        return x, y, math.degrees(math.atan2(R[3], R[0]))

    def pose(self) -> tuple[float, float, float]:
        return self._true_pose()

    def overhead_jpeg(self, size: int = 420, trail: bool = True) -> bytes:
        img = None
        for _ in range(OVERHEAD_FIRST_PICTURE_TRIES):
            self.idle(OVERHEAD_SETTLE_SECONDS)
            try:
                img = Image.open(io.BytesIO(OVERHEAD_IMAGE_FILE.read_bytes())).convert("RGB")
                break
            except (OSError, ValueError):
                continue
        if img is None:
            raise RuntimeError(f"No overhead camera picture at {OVERHEAD_IMAGE_FILE}; regenerate the world "
                               "(python webots/make_world.py) and reload it in Webots")
        buf = io.BytesIO()
        img.resize((size, size)).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def show_path(self, points: list[tuple[float, float]]) -> None:
        self.clear_path()
        strips = []
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            length = math.hypot(x1 - x0, y1 - y0)
            if length < 1e-3:
                continue
            angle = math.atan2(y1 - y0, x1 - x0)
            strips.append(f"""Pose {{ translation {(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {PATH_LINE_Z}
  rotation 0 0 1 {angle:.4f}
  children [ Shape {{ appearance PBRAppearance {{ baseColor 0.12 0.43 0.92 emissiveColor 0.12 0.43 0.92
                                                roughness 1 metalness 0 }}
                     geometry Box {{ size {length + PATH_LINE_WIDTH_M:.4f} {PATH_LINE_WIDTH_M} 0.001 }} }} ] }}""")
        if strips:
            root = self.robot.getRoot().getField("children")
            root.importMFNodeFromString(-1, f"DEF {PATH_DEF} Pose {{ children [ {' '.join(strips)} ] }}")

    def clear_path(self) -> None:
        node = self.robot.getFromDef(PATH_DEF)
        if node is not None:
            node.remove()

    def capture_jpeg(self) -> bytes:
        self.robot.step(self.ts)
        w, h = self.cam.getWidth(), self.cam.getHeight()
        img = Image.frombytes("RGBA", (w, h), self.cam.getImage())
        b, g, r, _ = img.split()
        buf = io.BytesIO()
        Image.merge("RGB", (r, g, b)).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _blocked(self) -> bool:
        return max(s.getValue() for s in self.front_ir) > IR_BLOCKED

    def _drive(self, left_rad: float, right_rad: float, check_front: bool) -> tuple[float, bool]:
        l0, r0 = self.lenc.getValue(), self.renc.getValue()
        self.left.setVelocity(math.copysign(SPEED, left_rad) if left_rad else 0.0)
        self.right.setVelocity(math.copysign(SPEED, right_rad) if right_rad else 0.0)
        blocked = False
        target = max(abs(left_rad), abs(right_rad))
        while self.robot.step(self.ts) != -1:
            done = max(abs(self.lenc.getValue() - l0), abs(self.renc.getValue() - r0))
            if done >= target:
                break
            if check_front and self._blocked():
                blocked = True
                break
        self.halt()
        return max(abs(self.lenc.getValue() - l0), abs(self.renc.getValue() - r0)), blocked

    def turn(self, degrees: float) -> MoveResult:
        wheel = math.radians(abs(degrees)) * (AXLE_M / 2) / WHEEL_RADIUS_M
        sign = 1 if degrees > 0 else -1
        done, _ = self._drive(sign * wheel, -sign * wheel, check_front=False)
        return MoveResult(done=sign * math.degrees(done * WHEEL_RADIUS_M / (AXLE_M / 2)))

    def forward(self, cm: float) -> MoveResult:
        if cm <= 0:
            return MoveResult(done=0.0)
        if self._blocked():
            return MoveResult(done=0.0, blocked=True)
        wheel = (cm / 100) / WHEEL_RADIUS_M
        done, blocked = self._drive(wheel, wheel, check_front=True)
        return MoveResult(done=done * WHEEL_RADIUS_M * 100, blocked=blocked)

    def halt(self) -> None:
        self.left.setVelocity(0.0)
        self.right.setVelocity(0.0)

    def idle(self, seconds: float) -> bool:
        steps = max(1, int(seconds * 1000 / self.ts))
        for _ in range(steps):
            if self.robot.step(self.ts) == -1:
                return False
        return True
