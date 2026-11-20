from __future__ import annotations

import io
import math
import time

import numpy as np
from PIL import Image, ImageDraw

from agent.geometry import CameraModel, vfov_from_hfov
from agent.maze import (CELL, LABEL_H, LABEL_W, ROUTE_GREEN, SIGN_Z, WALL_H, WALL_T, Layout, build,
                        floor_label_image)
from backends.base import MoveResult

RADIUS = 0.037

OVERHEAD_MARGIN_M = 0.1
OVERHEAD_BACKGROUND = (235, 235, 235)
OVERHEAD_FLOOR = (150, 150, 155)
OVERHEAD_WALL = (30, 30, 30)
ROVER_GREEN = (52, 168, 83)


class SimBackend:
    def __init__(self, layout: Layout | None = None, width: int = 480, height: int = 360,
                 hfov_deg: float = 80.0, camera_height_cm: float = 2.8, realtime: bool = True):
        self.lay = layout or build()
        self.realtime = realtime
        self.W, self.H = width, height
        self.camera = CameraModel(hfov_deg, vfov_from_hfov(hfov_deg, width, height), camera_height_cm, 0.0)
        self.x, self.y, self.heading = self.lay.start
        self.trail = [(self.x, self.y)]
        self.walls = np.array(self.lay.wall_segments(), dtype=np.float64)
        self.wall_shade = np.where(abs(self.walls[:, 1] - self.walls[:, 3]) < 1e-9, 0.93, 0.80)
        self.sign_list = self.lay.all_signs()
        segs, tex = [], []
        for s in self.sign_list:
            t = math.radians(s.facing_deg + 90)
            ux, uy = math.cos(t) * s.w / 2, math.sin(t) * s.w / 2
            segs.append((s.x - ux, s.y - uy, s.x + ux, s.y + uy))
            tex.append(np.asarray(s.image().resize((512, round(512 * s.h / s.w))), dtype=np.float32) / 255)
        self.signs = np.array(segs, dtype=np.float64).reshape(-1, 4)
        self.sign_tex = tex
        self.sign_h = np.array([s.h for s in self.sign_list])
        self.f = (width / 2) / math.tan(math.radians(hfov_deg / 2))
        self._label_cache: dict = {}

    def true_pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.heading

    def reset(self) -> None:
        self.x, self.y, self.heading = self.lay.start
        self.trail = [(self.x, self.y)]

    def set_pose(self, x: float, y: float, heading_deg: float) -> None:
        self.x, self.y, self.heading = x, y, heading_deg % 360
        self.trail = [(self.x, self.y)]

    def _sleep(self, s: float) -> None:
        if self.realtime:
            time.sleep(s)

    def capture_jpeg(self) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.render()).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def turn(self, degrees: float) -> MoveResult:
        self.heading = (self.heading - degrees) % 360
        self._sleep(min(abs(degrees) / 180, 1.0) * 0.3)
        return MoveResult(done=degrees)

    def forward(self, cm: float) -> MoveResult:
        h = math.radians(self.heading)
        moved, blocked = 0.0, False
        step = 0.005
        while moved < cm / 100 - 1e-9:
            d = min(step, cm / 100 - moved)
            nx, ny = self.x + d * math.cos(h), self.y + d * math.sin(h)
            if self._min_wall_dist(nx, ny) < RADIUS + WALL_T / 2:
                blocked = True
                break
            self.x, self.y, moved = nx, ny, moved + d
        self.trail.append((self.x, self.y))
        self._sleep(moved * 3)
        return MoveResult(done=moved * 100, blocked=blocked, front_cm=self._front_cm())

    def halt(self) -> None:
        pass

    def idle(self, seconds: float) -> bool:
        self._sleep(seconds)
        return True

    def _min_wall_dist(self, x, y) -> float:
        a, b = self.walls[:, :2], self.walls[:, 2:]
        ab = b - a
        t = np.clip(((np.array([x, y]) - a) * ab).sum(1) / (ab * ab).sum(1), 0, 1)
        return float(np.min(np.hypot(*(a + ab * t[:, None] - np.array([x, y])).T)))

    def _hits(self, ox, oy, dirs, segs):
        a, e = segs[:, :2], segs[:, 2:] - segs[:, :2]
        dx, dy = dirs[:, 0:1], dirs[:, 1:2]
        denom = dx * e[:, 1] - dy * e[:, 0]
        ax, ay = a[:, 0] - ox, a[:, 1] - oy
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (ax * e[:, 1] - ay * e[:, 0]) / denom
            s = (ax * dy - ay * dx) / denom
        ok = (np.abs(denom) > 1e-12) & (t > 1e-6) & (s >= 0) & (s <= 1)
        return np.where(ok, t, np.inf), s

    def _front_cm(self) -> float:
        h = math.radians(self.heading)
        t, _ = self._hits(self.x, self.y, np.array([[math.cos(h), math.sin(h)]]), self.walls)
        return float((t.min() - RADIUS) * 100)

    def render(self) -> np.ndarray:
        W, H, f = self.W, self.H, self.f
        cam_h = self.camera.height_cm / 100
        h = math.radians(self.heading)
        cols = (np.arange(W) + 0.5 - W / 2)
        angles = h - np.arctan(cols / f)
        dirs = np.stack([np.cos(angles), np.sin(angles)], 1)
        cos_off = np.cos(np.arctan(cols / f))

        img = np.empty((H, W, 3), dtype=np.float32)
        img[:] = (0.80, 0.80, 0.82)

        rows = np.arange(H) + 0.5
        below = rows > H / 2 + 0.5
        depth = np.where(below, f * cam_h / np.maximum(rows - H / 2, 1e-6), 0.0)[:, None]
        fwd = np.array([math.cos(h), math.sin(h)])
        right = np.array([math.sin(h), -math.cos(h)])
        gx = self.x + depth * fwd[0] + depth * cols[None] / f * right[0]
        gy = self.y + depth * fwd[1] + depth * cols[None] / f * right[1]
        tiles = (np.floor(gx / (CELL * 2)) + np.floor(gy / (CELL * 2))) % 2
        floor = np.where(tiles[..., None] > 0, 0.40, 0.44) * np.ones(3)
        img[below] = floor[below]

        tw, _ = self._hits(self.x, self.y, dirs, self.walls)
        nearest = tw.argmin(1)
        dist = tw[np.arange(W), nearest] * cos_off
        shade = self.wall_shade[nearest]
        top = H / 2 - f * (WALL_H - cam_h) / dist
        bot = H / 2 + f * cam_h / dist
        rr = rows[:, None]
        wall = (rr >= top[None]) & (rr <= bot[None]) & np.isfinite(dist)[None]
        fog = np.clip(1 - dist / 3, 0.75, 1)
        img[wall] = (np.ones((H, W)) * (shade * fog)[None])[wall][:, None] * np.array([1, 1, 0.97])

        for col, k, d, u_frac in self._sign_columns(dirs, cos_off, tw[np.arange(W), nearest]):
            sh = self.sign_h[k]
            r0 = int(max(0, H / 2 - f * (SIGN_Z + sh / 2 - cam_h) / d))
            r1 = int(min(H, H / 2 - f * (SIGN_Z - sh / 2 - cam_h) / d))
            if r1 <= r0:
                continue
            z = cam_h + (H / 2 - (np.arange(r0, r1) + 0.5)) * d / f
            tex = self.sign_tex[k]
            v = np.clip(((SIGN_Z + sh / 2 - z) / sh * tex.shape[0]).astype(int), 0, tex.shape[0] - 1)
            u = min(int(u_frac * tex.shape[1]), tex.shape[1] - 1)
            img[r0:r1, col] = tex[v, u]
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def _rays(self):
        h = math.radians(self.heading)
        cols = (np.arange(self.W) + 0.5 - self.W / 2)
        off = np.arctan(cols / self.f)
        angles = h - off
        return np.stack([np.cos(angles), np.sin(angles)], 1), np.cos(off)

    def _sign_columns(self, dirs, cos_off, wall_t):
        if not len(self.signs):
            return []
        ts, ss = self._hits(self.x, self.y, dirs, self.signs)
        out = []
        for col in range(len(dirs)):
            k = int(ts[col].argmin())
            if not np.isfinite(ts[col, k]) or ts[col, k] > wall_t[col] + 1e-4:
                continue
            s = self.sign_list[k]
            fx, fy = math.cos(math.radians(s.facing_deg)), math.sin(math.radians(s.facing_deg))
            if dirs[col, 0] * fx + dirs[col, 1] * fy >= -1e-3:
                continue
            out.append((col, k, ts[col, k] * cos_off[col], ss[col, k]))
        return out

    @property
    def overhead_span_m(self) -> float:
        return max(self.lay.rows, self.lay.cols) * CELL + OVERHEAD_MARGIN_M

    def pose(self) -> tuple[float, float, float]:
        return self.true_pose()

    def overhead_jpeg(self, size: int = 420, trail: bool = True) -> bytes:
        buf = io.BytesIO()
        self.overhead_image(size, trail).save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def overhead_image(self, size: int = 420, trail: bool = True) -> Image.Image:
        lay = self.lay
        span = self.overhead_span_m
        pixels_per_m = size / span
        P = lambda x, y: ((x + span / 2) * pixels_per_m, (span / 2 - y) * pixels_per_m)
        img = Image.new("RGB", (size, size), OVERHEAD_BACKGROUND)
        d = ImageDraw.Draw(img)
        x0, y0 = P(lay.cx(0) - CELL / 2, lay.cy(0) + CELL / 2)
        x1, y1 = P(lay.cx(lay.cols - 1) + CELL / 2, lay.cy(lay.rows - 1) - CELL / 2)
        d.rectangle([x0, y0, x1, y1], fill=OVERHEAD_FLOOR)
        for name in lay.towns:
            lx, ly = lay.label_centre(name)
            w, h = round(LABEL_W * pixels_per_m), round(LABEL_H * pixels_per_m)
            if (name, w, h) not in self._label_cache:
                self._label_cache[(name, w, h)] = floor_label_image(name).resize((w, h), Image.LANCZOS)
            label = self._label_cache[(name, w, h)]
            px, py = P(lx, ly)
            img.paste(label, (round(px - w / 2), round(py - h / 2)))
        wall_px = max(3, round(WALL_T * 1.6 * pixels_per_m))
        for wx0, wy0, wx1, wy1 in self.walls:
            d.line([P(wx0, wy0), P(wx1, wy1)], fill=OVERHEAD_WALL, width=wall_px)
        for s in lay.all_signs():
            t = math.radians(s.facing_deg + 90)
            half = s.w / 2 * 0.6
            colour = ROUTE_GREEN if s.kind == "route" else s.colour
            d.line([P(s.x - half * math.cos(t), s.y - half * math.sin(t)),
                    P(s.x + half * math.cos(t), s.y + half * math.sin(t))], fill=colour, width=max(2, wall_px - 1))
        if trail and len(self.trail) > 1:
            d.line([P(*p) for p in self.trail], fill=(66, 133, 244), width=2)
        rx, ry = P(self.x, self.y)
        r = RADIUS * pixels_per_m
        d.ellipse([rx - r, ry - r, rx + r, ry + r], fill=ROVER_GREEN, outline=OVERHEAD_WALL)
        h = math.radians(self.heading)
        nose = P(self.x + RADIUS * 1.6 * math.cos(h), self.y + RADIUS * 1.6 * math.sin(h))
        d.line([(rx, ry), nose], fill=OVERHEAD_WALL, width=max(2, wall_px - 1))
        return img
