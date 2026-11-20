from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from google.genai import errors, types
from PIL import Image, ImageDraw, ImageFont

from agent.keys import DailyQuotaUsedUp, gemini_client
from agent.tools import TOOLS, system_prompt
from backends.base import RoverBackend

SECONDS_BETWEEN_CALLS_BY_TIER = {"free": 13.0, "paid": 0.0}
DEFAULT_API_TIER = "paid"
RATE_LIMITED = 429
MAX_CALL_ATTEMPTS = 3
RETRY_PAUSE_SECONDS = 20.0
DAILY_QUOTA_MARKER = "PerDay"

MAX_TURN_DEGREES = 180.0

PATH_BLUE = (66, 133, 244)
TARGET_RED = (234, 67, 53)
LABEL_TEXT_WHITE = (255, 255, 255)
LABEL_BAR_BLACK = (0, 0, 0)
POINT_SCALE = 1000
MAX_LABEL_CHARS = 90


@dataclass
class AgentConfig:
    model: str = os.environ.get("ROVER_MODEL", "gemini-robotics-er-2-preview")
    thinking_level: str = os.environ.get("ROVER_THINKING", "MEDIUM")
    max_steps: int = 6
    max_seconds: float = 60.0
    max_step_cm: float = 30.0
    images_in_context: int = 3
    notes_kept: int = 8
    runs_dir: str = "runs"
    api_tier: str = os.environ.get("ROVER_API_TIER", DEFAULT_API_TIER)


Event = dict
EventSink = Callable[[Event], None]


class RoverAgent:
    def __init__(self, backend: RoverBackend, config: AgentConfig | None = None,
                 client=None, on_event: EventSink | None = None):
        self.backend = backend
        self.cfg = config or AgentConfig()
        if self.cfg.api_tier not in SECONDS_BETWEEN_CALLS_BY_TIER:
            raise ValueError(f"ROVER_API_TIER={self.cfg.api_tier!r}; use one of "
                             f"{', '.join(SECONDS_BETWEEN_CALLS_BY_TIER)}")
        if client is None:
            client = gemini_client()
        self.client = client
        self.uses_real_api = getattr(client, "uses_real_api", type(client).__module__.startswith("google.genai"))
        self.seconds_between_calls = (SECONDS_BETWEEN_CALLS_BY_TIER[self.cfg.api_tier]
                                      if self.uses_real_api else 0.0)
        self._last_call_at = 0.0
        self._seconds_waited = 0.0
        self.emit = on_event or (lambda e: None)
        if hasattr(client, "on_switch"):
            client.on_switch = lambda message: self.emit({"type": "thought", "step": 0, "latency_s": 0,
                                                          "text": f"({message})"})
        self.gen_config = types.GenerateContentConfig(
            system_instruction=system_prompt(self.cfg.max_step_cm, self.cfg.max_steps),
            tools=TOOLS,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")),
            thinking_config=types.ThinkingConfig(
                thinking_level=self.cfg.thinking_level, include_thoughts=True),
        )

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)
            self._seconds_waited += seconds

    def _ask(self, history: list[types.Content]):
        for attempt in range(1, MAX_CALL_ATTEMPTS + 1):
            next_call_allowed_at = self._last_call_at + self.seconds_between_calls
            self._wait(next_call_allowed_at - time.monotonic())

            call_started_at = time.monotonic()
            self._last_call_at = call_started_at
            try:
                response = self.client.models.generate_content(
                    model=self.cfg.model, contents=self._trimmed(history), config=self.gen_config)
            except errors.ClientError as e:
                is_rate_limit = self.uses_real_api and e.code == RATE_LIMITED
                if not is_rate_limit:
                    raise
                if DAILY_QUOTA_MARKER in str(e):
                    raise DailyQuotaUsedUp(
                        "The API's daily request quota for this model is used up (free tier: 20 per day). "
                        "Switch the key's project to the paid tier, or try again tomorrow.") from e
                if attempt == MAX_CALL_ATTEMPTS:
                    raise
                pause_seconds = max(self.seconds_between_calls, RETRY_PAUSE_SECONDS)
                self.emit({"type": "thought", "step": 0, "latency_s": 0,
                           "text": f"(API rate limit reached; waiting {pause_seconds:.0f} s and trying again)"})
                self._wait(pause_seconds)
                continue
            latency_s = time.monotonic() - call_started_at
            return response, latency_s

    def run(self, instruction: str, who: str = "anonymous") -> dict:
        run_dir = Path(self.cfg.runs_dir) / time.strftime("%Y%m%d-%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        started_at = time.monotonic()
        self._seconds_waited = 0.0
        self.emit({"type": "start", "who": who, "instruction": instruction})

        frame = self.backend.capture_jpeg()
        history: list[types.Content] = [types.Content(role="user", parts=[
            types.Part.from_bytes(data=frame, mime_type="image/jpeg"),
            types.Part.from_text(text=f"Instruction from {who}: {instruction}\n"
                                      "This is your current camera view."),
        ])]
        outcome = {"reached_goal": False, "summary": "Ran out of actions.", "steps": 0}
        notes: list[str] = []

        try:
            for step in range(1, self.cfg.max_steps + 1):
                seconds_used = time.monotonic() - started_at - self._seconds_waited
                if seconds_used > self.cfg.max_seconds:
                    outcome["summary"] = "Ran out of time."
                    break

                response, latency_s = self._ask(history)

                content = response.candidates[0].content
                history.append(content)
                thoughts = [p.text for p in content.parts or [] if p.thought and p.text]
                calls = response.function_calls or []
                self.emit({"type": "thought", "step": step, "text": "\n".join(thoughts),
                           "latency_s": round(latency_s, 2)})

                if not calls:
                    outcome["summary"] = "The model did not choose an action."
                    break

                replies = []
                finished = False
                for i, call in enumerate(calls):
                    is_first_call = i == 0
                    if is_first_call:
                        args = dict(call.args or {})
                        result = self._execute(call.name, args, frame, run_dir, step)
                        notes.append(_note(step, call.name, args, result))
                    else:
                        result = {"status": "skipped", "reason": "Only one action per turn."}
                    replies.append(types.Part(function_response=types.FunctionResponse(
                        id=call.id, name=call.name, response=result)))
                    if call.name == "stop" and is_first_call:
                        finished = True
                        outcome.update(reached_goal=bool(call.args.get("reached_goal")),
                                       summary=str(call.args.get("summary", "")))

                outcome["steps"] = step
                if finished:
                    break

                frame = self.backend.capture_jpeg()
                recent_notes = notes[-self.cfg.notes_kept:]
                history.append(types.Content(role="user", parts=replies + [
                    types.Part.from_bytes(data=frame, mime_type="image/jpeg"),
                    types.Part.from_text(text="New camera view after your action.\nYour notes so far:\n"
                                              + "\n".join(recent_notes)),
                ]))
        except Exception as exc:
            outcome["summary"] = f"Stopped because of an error: {exc}"
            self.emit({"type": "error", "text": str(exc)})
        finally:
            self.backend.halt()

        self.emit({"type": "done", **outcome})
        return outcome

    def _execute(self, name: str, args: dict, frame: bytes, run_dir: Path, step: int) -> dict:
        camera = self.backend.camera
        point = None
        if name == "go_to_point":
            y, x = float(args["y"]), float(args["x"])
            point = (y, x)
            turn_degrees, floor_distance_cm = camera.point_to_motion(y, x)
            turn_done = self.backend.turn(turn_degrees)
            if floor_distance_cm is None:
                result = {"status": "turned_only", "turned_deg": round(turn_done.done),
                          "note": "That point is not on the floor, so the rover only turned toward it."}
            else:
                drive_cm = min(floor_distance_cm, self.cfg.max_step_cm)
                drive_done = self.backend.forward(drive_cm)
                result = {"status": "blocked" if drive_done.blocked else "ok",
                          "turned_deg": round(turn_done.done),
                          "moved_cm": round(drive_done.done, 1), "blocked": drive_done.blocked}
                if drive_done.front_cm is not None:
                    result["front_distance_cm"] = round(drive_done.front_cm, 1)
        elif name == "turn":
            degrees = _clamp(float(args["degrees"]), -MAX_TURN_DEGREES, MAX_TURN_DEGREES)
            turn_done = self.backend.turn(degrees)
            result = {"status": "ok", "turned_deg": round(turn_done.done)}
        elif name == "move_forward":
            drive_cm = _clamp(float(args["cm"]), 0.0, self.cfg.max_step_cm)
            drive_done = self.backend.forward(drive_cm)
            result = {"status": "blocked" if drive_done.blocked else "ok",
                      "moved_cm": round(drive_done.done, 1), "blocked": drive_done.blocked}
        elif name == "stop":
            result = {"status": "stopped"}
        else:
            result = {"status": "error", "reason": f"Unknown tool {name}"}

        args_to_show = {k: v for k, v in args.items() if k != "observation"}
        annotated = annotate(frame, name, args_to_show, point)
        path = run_dir / f"step{step:02d}_{name}.jpg"
        path.write_bytes(annotated)
        self.emit({"type": "action", "step": step, "tool": name, "args": args,
                   "result": result, "image": annotated})
        return result

    def _trimmed(self, history: list[types.Content]) -> list[types.Content]:
        frames_to_keep = self.cfg.images_in_context
        out, frames_seen = [], 0
        for content in reversed(history):
            has_frame = content.role == "user" and any(p.inline_data for p in content.parts)
            if has_frame:
                frames_seen += 1
                if frames_seen > frames_to_keep:
                    content = types.Content(role="user", parts=[
                        p if not p.inline_data else types.Part.from_text(text="[older camera frame removed]")
                        for p in content.parts])
            out.append(content)
        return list(reversed(out))


def _clamp(value: float, lowest: float, highest: float) -> float:
    return max(lowest, min(highest, value))


def _note(step: int, tool: str, args: dict, result: dict) -> str:
    seen = str(args.get("observation", "")).strip() or "(no observation)"
    did = ", ".join(f"{k}={v}" for k, v in args.items() if k not in ("observation", "label", "summary"))
    status = result.get("status", "")
    return f"{step}. saw: {seen} -> did: {tool}({did}) [{status}]"


def annotate(jpeg: bytes, tool: str, args: dict, point: tuple[float, float] | None) -> bytes:
    img = Image.open(io.BytesIO(jpeg)).convert("RGB")
    width, height = img.size
    draw = ImageDraw.Draw(img)
    line_width = max(2, width // 160)
    if point:
        point_y, point_x = point
        px = point_x / POINT_SCALE * width
        py = point_y / POINT_SCALE * height
        rover_position = (width / 2, height)
        draw.line([rover_position, (px, py)], fill=PATH_BLUE, width=line_width * 2)
        ring_radius = 4 * line_width
        draw.ellipse([px - ring_radius, py - ring_radius, px + ring_radius, py + ring_radius],
                     outline=TARGET_RED, width=line_width)
    label = tool + " " + ", ".join(f"{k}={v}" for k, v in args.items())
    font_size = max(12, height // 22)
    font = ImageFont.load_default(size=font_size)
    label_bar_height = font_size + 10
    draw.rectangle([0, 0, width, label_bar_height], fill=LABEL_BAR_BLACK)
    draw.text((6, 4), label[:MAX_LABEL_CHARS], fill=LABEL_TEXT_WHITE, font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()
