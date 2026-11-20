from __future__ import annotations

import io
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from agent.keys import RATE_LIMITED
from agent.maze import Layout, build, compass
from agent.recorder import TripRecorder

BRAIN_NAME = "ER 2"
MAX_PLAN_CALLS = 3
SECONDS_BETWEEN_CALLS_BY_TIER = {"free": 13.0, "paid": 0.0}
DEFAULT_API_TIER = "paid"
MAX_CALL_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 20.0
POINT_SCALE = 1000

DRIVE_HOP_CM = 2.5
REACHED_POINT_CM = 2.0
ARRIVED_CM = 6.0
TURN_TOLERANCE_DEG = 4.0
MAX_HOPS_PER_POINT = 200

MODEL_IMAGE_PX = 640
RECORDING_PX = 720
PLAN_CARD_SECONDS = 1.5
PLANNED_BLUE = (30, 110, 235)
TRAIL_GREEN = (20, 160, 70)
CAPTION_BAR = (0, 0, 0)
CAPTION_TEXT = (255, 255, 255)

DEFAULT_SCENE = ("The thin dark lines are walls. The rover is the green disc; the short dark line "
                 "shows which way it faces.")

PLANNER_PROMPT = """You plan driving routes for a small rover in a tabletop maze, seen from directly above.

The image: {scene} The rover can only drive along the open grey floor between the walls, never
through or over a wall. Each town's name is written flat on the floor of that town's square.
Small coloured bars on walls are signs; ignore them.

Your job:
1. Decide which town on this map the person wants to go to. They may name it, or describe it
   (use what you know about Sri Lanka). If the place they want is not written on this map, set
   "destination" to null and "points" to [].
2. Otherwise plan a path from the rover to the middle of that town's square: keep to the middle of
   the corridors, put a point at every corner, and make sure the straight line between each pair of
   consecutive points crosses no wall. Start with the rover's position. End in the town's square.

Points are [y, x], normalised to 0-1000 over the image (y = 0 at the top, x = 0 at the left).

Reply with JSON only, no other text:
{"destination": "TOWN NAME AS WRITTEN ON THE MAP" or null, "reason": "one short sentence", "points": [[y, x], ...]}"""


@dataclass
class PlannerConfig:
    model: str = os.environ.get("ROVER_MODEL", "gemini-robotics-er-2-preview")
    thinking_level: str = os.environ.get("ROVER_THINKING", "MEDIUM")
    api_tier: str = os.environ.get("ROVER_API_TIER", DEFAULT_API_TIER)
    max_plan_calls: int = MAX_PLAN_CALLS
    max_seconds: float = 300.0
    runs_dir: str = "runs"
    drive_hop_cm: float = DRIVE_HOP_CM


def world_to_point(x: float, y: float, span_m: float) -> tuple[int, int]:
    return (round((span_m / 2 - y) / span_m * POINT_SCALE),
            round((x + span_m / 2) / span_m * POINT_SCALE))


def point_to_world(point_y: float, point_x: float, span_m: float) -> tuple[float, float]:
    return (point_x / POINT_SCALE * span_m - span_m / 2,
            span_m / 2 - point_y / POINT_SCALE * span_m)


def parse_plan(text: str) -> dict | None:
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        plan = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(plan, dict):
        return None
    points = []
    for p in plan.get("points") or []:
        if isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(v, (int, float)) for v in p):
            points.append((float(p[0]), float(p[1])))
        elif isinstance(p, dict) and "point" in p:
            points.append((float(p["point"][0]), float(p["point"][1])))
    destination = plan.get("destination")
    return {"destination": str(destination).strip().upper() if destination else None,
            "reason": str(plan.get("reason", "")), "points": points}


def _slug(text: str, limit: int = 24) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:limit] or "anonymous"


def _wrap_degrees(degrees: float) -> float:
    return (degrees + 180) % 360 - 180


class OverheadPlanner:
    def __init__(self, backend, client, config: PlannerConfig | None = None,
                 on_event: Callable[[dict], None] | None = None, layout: Layout | None = None):
        self.backend = backend
        self.cfg = config or PlannerConfig()
        if self.cfg.api_tier not in SECONDS_BETWEEN_CALLS_BY_TIER:
            raise ValueError(f"ROVER_API_TIER={self.cfg.api_tier!r}; use one of "
                             f"{', '.join(SECONDS_BETWEEN_CALLS_BY_TIER)}")
        self.client = client
        self.lay = layout or getattr(backend, "lay", None) or build()
        self.emit = on_event or (lambda e: None)
        if hasattr(client, "on_switch"):
            client.on_switch = lambda message: self.emit({"type": "notice", "text": message})
        self.span_m = backend.overhead_span_m
        self.planned: list[tuple[float, float]] = []
        self.trail: list[tuple[float, float]] = []
        self.caption = ""
        self._last_call_at = 0.0
        self._seconds_waited = 0.0

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)
            self._seconds_waited += seconds

    def _rover_point(self) -> tuple[int, int]:
        x, y, _ = self.backend.pose()
        return world_to_point(x, y, self.span_m)

    def view(self, size: int = RECORDING_PX) -> Image.Image:
        img = Image.open(io.BytesIO(self.backend.overhead_jpeg(size, trail=False))).convert("RGB")
        d = ImageDraw.Draw(img)
        scale = size / POINT_SCALE
        to_px = lambda x, y: tuple(v * scale for v in reversed(world_to_point(x, y, self.span_m)))
        line_px = max(3, size // 160)
        if len(self.planned) > 1:
            d.line([to_px(*p) for p in self.planned], fill=PLANNED_BLUE, width=line_px, joint="curve")
        for p in self.planned:
            px, py = to_px(*p)
            d.ellipse([px - line_px * 1.5, py - line_px * 1.5, px + line_px * 1.5, py + line_px * 1.5],
                      fill=PLANNED_BLUE)
        if len(self.trail) > 1:
            d.line([to_px(*p) for p in self.trail], fill=TRAIL_GREEN, width=max(2, line_px - 1))
        if self.caption:
            font_size = max(12, size // 40)
            font = ImageFont.load_default(size=font_size)
            d.rectangle([0, 0, size, font_size + 12], fill=CAPTION_BAR)
            d.text((8, 5), self.caption[:110], fill=CAPTION_TEXT, font=font)
        return img

    def preview_jpeg(self) -> bytes:
        buf = io.BytesIO()
        self.view().save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def _frame(self, recorder: TripRecorder, hold_seconds: float = 0.0) -> None:
        img = self.view()
        recorder.add(img, hold_seconds)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        self.emit({"type": "frame", "image": buf.getvalue()})

    def _ask(self, image: bytes, prompt_text: str, call: int) -> dict:
        contents = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=image, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt_text)])]
        config = types.GenerateContentConfig(
            system_instruction=PLANNER_PROMPT.replace(
                "{scene}", getattr(self.backend, "overhead_scene", DEFAULT_SCENE)),
            thinking_config=types.ThinkingConfig(thinking_level=self.cfg.thinking_level, include_thoughts=True))
        attempt = 0
        while True:
            client = self.client
            pause = SECONDS_BETWEEN_CALLS_BY_TIER[self.cfg.api_tier]
            self._wait(self._last_call_at + pause - time.monotonic())
            self._last_call_at = time.monotonic()
            self.emit({"type": "plan_start", "call": call, "brain": BRAIN_NAME})
            started = time.monotonic()
            answer, thoughts, usage = [], [], None
            try:
                for chunk in client.models.generate_content_stream(
                        model=self.cfg.model, contents=contents, config=config):
                    usage = getattr(chunk, "usage_metadata", None) or usage
                    for part in (chunk.candidates[0].content.parts if chunk.candidates and
                                 chunk.candidates[0].content else []) or []:
                        if not part.text:
                            continue
                        kind = "thought" if part.thought else "answer"
                        (thoughts if part.thought else answer).append(part.text)
                        self.emit({"type": "stream", "call": call, "kind": kind, "text": part.text})
            except Exception as exc:
                attempt += 1
                if getattr(exc, "code", None) != RATE_LIMITED or attempt >= MAX_CALL_ATTEMPTS:
                    raise
                self.emit({"type": "notice", "text": f"API rate limit reached; waiting "
                                                     f"{RETRY_PAUSE_SECONDS:.0f} s and trying again."})
                self._wait(RETRY_PAUSE_SECONDS)
                continue
            slot = getattr(client, "slot_in_use", None)
            pool = getattr(client, "pool", None)
            return {"call": call, "brain": BRAIN_NAME, "model": self.cfg.model,
                    "key": pool.describe(slot) if (pool is not None and slot) else None,
                    "latency_s": round(time.monotonic() - started, 2),
                    "prompt": prompt_text, "thoughts": "".join(thoughts), "raw_reply": "".join(answer),
                    "tokens": {"input": getattr(usage, "prompt_token_count", None),
                               "output": getattr(usage, "candidates_token_count", None),
                               "thinking": getattr(usage, "thoughts_token_count", None)} if usage else None}

    def _at_destination(self, town: str, strict: bool) -> bool:
        x, y, _ = self.backend.pose()
        if strict:
            cx, cy = self.lay.centre(self.lay.towns[town])
            return math.hypot(x - cx, y - cy) * 100 <= ARRIVED_CM
        return self.lay.cell_of(x, y) == self.lay.towns[town]

    def _follow(self, points: list[tuple[float, float]], town: str, recorder: TripRecorder,
                out_of_time: Callable[[], bool]) -> str:
        hop_cm = getattr(self.backend, "overhead_drive_hop_cm", self.cfg.drive_hop_cm)
        for target_x, target_y in points:
            for _ in range(MAX_HOPS_PER_POINT):
                if self._at_destination(town, strict=True):
                    return "arrived"
                if out_of_time():
                    return "timeout"
                x, y, heading = self.backend.pose()
                distance_cm = math.hypot(target_x - x, target_y - y) * 100
                if distance_cm <= REACHED_POINT_CM:
                    break
                bearing = math.degrees(math.atan2(target_y - y, target_x - x))
                turn_right_deg = _wrap_degrees(heading - bearing)
                if abs(turn_right_deg) > TURN_TOLERANCE_DEG:
                    self.backend.turn(turn_right_deg)
                moved = self.backend.forward(min(hop_cm, distance_cm))
                x, y, _ = self.backend.pose()
                self.trail.append((x, y))
                self._frame(recorder)
                if moved.blocked:
                    return "blocked"
            else:
                return "blocked"
        return "arrived" if self._at_destination(town, strict=False) else "path_ended"

    def run(self, instruction: str, who: str = "anonymous") -> dict:
        started_at = time.monotonic()
        self._seconds_waited = 0.0
        runs = Path(self.cfg.runs_dir)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        recorder = TripRecorder(runs / f"{stamp}-{_slug(who)}")
        x, y, _ = self.backend.pose()
        start_town = next((t for t, c in self.lay.towns.items() if c == self.lay.cell_of(x, y)), None)
        self.planned, self.trail, self.caption = [], [(x, y)], f"{who}: {instruction}"
        self.emit({"type": "start", "who": who, "instruction": instruction})
        self._frame(recorder, hold_seconds=1.0)

        outcome = {"reached_goal": False, "summary": "No plan reached the destination.", "steps": 0}
        town, plans, note = None, [], ""
        out_of_time = lambda: time.monotonic() - started_at - self._seconds_waited > self.cfg.max_seconds
        try:
            for call in range(1, self.cfg.max_plan_calls + 1):
                if out_of_time():
                    outcome["summary"] = "Ran out of time."
                    break
                if hasattr(self.backend, "clear_path"):
                    self.backend.clear_path()
                image = self.backend.overhead_jpeg(MODEL_IMAGE_PX, trail=False)
                ry, rx = self._rover_point()
                _, _, heading = self.backend.pose()
                prompt_text = (f"Instruction from {who}: {instruction}\n"
                               f"The rover is at [y, x] = [{ry}, {rx}], facing {compass(round(heading / 90) * 90)}."
                               + (f"\n{note}" if note else ""))
                record = self._ask(image, prompt_text, call)
                plan = parse_plan(record["raw_reply"])
                record["parsed"] = plan
                recorder.save_plan(call, image, record)
                outcome["steps"] = call
                plans.append(record)

                if plan is None:
                    note = "Your last reply could not be read. Reply with the JSON object only."
                    self.emit({"type": "notice", "text": "The plan could not be read; asking again."})
                    continue
                wanted = plan["destination"]
                if town is None:
                    if wanted not in self.lay.towns:
                        places = ", ".join(t.title() for t in self.lay.towns)
                        outcome["summary"] = (f"{wanted.title() if wanted else 'That place'} isn't on this map. "
                                              f"You can go to: {places}.")
                        self.emit({"type": "plan", "call": call, "destination": wanted, "reason": plan["reason"],
                                   "points": 0, "latency_s": record["latency_s"], "brain": record["brain"]})
                        break
                    town = wanted
                    recorder_name = f"{stamp}-{_slug(who)}-{_slug(town)}"
                if self._at_destination(town, strict=False) and call == 1:
                    outcome.update(reached_goal=True, summary=f"Already at {town.title()}.")
                    break

                self.planned = [point_to_world(py, px, self.span_m) for py, px in plan["points"]]
                self.emit({"type": "plan", "call": call, "destination": town, "reason": plan["reason"],
                           "points": len(self.planned), "latency_s": record["latency_s"], "brain": record["brain"]})
                if hasattr(self.backend, "show_path"):
                    self.backend.show_path(self.planned)
                self.caption = f"Plan {call} ({record['brain']}): to {town.title()}. {plan['reason']}"
                self._frame(recorder, hold_seconds=PLAN_CARD_SECONDS)
                if not self.planned:
                    note = "Your last reply had no points. Plan the path again."
                    continue

                result = self._follow(self.planned, town, recorder, out_of_time)
                if result == "arrived":
                    outcome.update(reached_goal=True,
                                   summary=f"Arrived at {town.title()} ({call} plan{'s' if call > 1 else ''}).")
                    break
                if result == "timeout":
                    outcome["summary"] = "Ran out of time."
                    break
                where = self._rover_point()
                note = (f"The rover followed your previous path but was stopped by a wall at [y, x] = "
                        f"[{where[0]}, {where[1]}]. Plan a new path from where it is now."
                        if result == "blocked" else
                        f"The rover reached the end of your previous path at [y, x] = [{where[0]}, {where[1]}], "
                        f"which is not inside {town}. Plan a new path from where it is now.")
                self.emit({"type": "notice", "text": ("Blocked by a wall" if result == "blocked"
                                                      else "Path ended short of the destination")
                                                     + ("; asking for a new plan." if call < self.cfg.max_plan_calls
                                                        else ".")})
            else:
                if town:
                    outcome["summary"] = f"Didn't reach {town.title()} after {self.cfg.max_plan_calls} plans."
        except Exception as exc:
            outcome["summary"] = f"Stopped because of an error: {exc}"
            self.emit({"type": "error", "text": str(exc)})
        finally:
            self.backend.halt()
        if town is None:
            recorder_name = f"{stamp}-{_slug(who)}-refused"

        mark = "Arrived" if outcome["reached_goal"] else "Stopped"
        self.caption = f"{mark}: {outcome['summary']}"
        self._frame(recorder, hold_seconds=1.5)
        if hasattr(self.backend, "clear_path"):
            self.backend.clear_path()
        x, y, _ = self.backend.pose()
        trip = {**outcome, "who": who, "instruction": instruction, "from": start_town, "to": town,
                "plans": len(plans), "brains": sorted({p["brain"] for p in plans}),
                "end_position_m": [round(x, 3), round(y, 3)],
                "seconds": round(time.monotonic() - started_at, 1)}
        video = recorder.finish(trip, rename_to=runs / recorder_name)
        outcome["recording"] = str(video) if video else None
        self.emit({"type": "done", **outcome})
        return outcome
