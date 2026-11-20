"""Run the rover outside Webots, then open http://127.0.0.1:8765.

  python run.py sim                    built-in simulator, ER 2 plans from the overhead view
  python run.py sim --mode camera      one step at a time from the front camera
  python run.py photos --folder DIR    photos in, printed motions out
  python run.py pi --port /dev/ttyACM0 on the Raspberry Pi

Needs GOOGLE_API_KEY_VAULT in .env (see .env.example).
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from agent.core import AgentConfig, RoverAgent
from agent.geometry import CameraModel
from agent.keys import NoApiKey, gemini_client
from agent.planner import OverheadPlanner, PlannerConfig
from agent.server import Hub, main_loop, serve
from backends.folder_backend import FolderBackend
from backends.sim_backend import SimBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("backend", choices=["sim", "photos", "pi"])
    p.add_argument("--mode", choices=["overhead", "camera"], default=os.environ.get("ROVER_MODE", "overhead"),
                   help="overhead: ER 2 plans the whole path from a top-down view (sim only); "
                        "camera: one step at a time from the front camera")
    p.add_argument("--folder", default="test_photos")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--http-port", type=int, default=8765)
    p.add_argument("--cooldown", type=float, default=0, help="seconds before the same name can go again")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--hfov", type=float, default=102)
    p.add_argument("--vfov", type=float, default=67)
    p.add_argument("--cam-height", type=float, default=9.0)
    p.add_argument("--cam-tilt", type=float, default=15.0)
    a = p.parse_args()
    try:
        client = gemini_client()
    except NoApiKey as exc:
        sys.exit(str(exc))

    if a.backend == "sim":
        backend = SimBackend()
        cfg = AgentConfig(max_step_cm=20, max_steps=30, max_seconds=240)
    elif a.backend == "photos":
        backend = FolderBackend(a.folder, CameraModel(a.hfov, a.vfov, a.cam_height, a.cam_tilt))
        cfg = AgentConfig()
    else:
        from backends.pi_backend import PiBackend
        backend = PiBackend(port=a.port)
        cfg = AgentConfig()
    if a.max_steps:
        cfg.max_steps = a.max_steps

    hub = Hub(cooldown_s=a.cooldown)
    on_event = lambda e: (hub.on_event(e), print_event(e))
    mode = a.mode if hasattr(backend, "overhead_span_m") else "camera"
    if mode == "overhead":
        agent = make_overhead_agent(client, backend, on_event)
        hub.input_hint = "Where to? e.g. Take me to Kandy"
        preview = agent.preview_jpeg
    else:
        hub.overhead_fn = getattr(backend, "overhead_jpeg", None)
        agent = RoverAgent(backend, cfg, client=client, on_event=on_event)
        preview = None
    serve(hub, a.host, a.http_port)
    print(f"Mode: {mode}.  Open http://{a.host}:{a.http_port}")
    main_loop(agent, hub, backend, preview_fn=preview)


def make_overhead_agent(client, backend, on_event, **config) -> OverheadPlanner:
    return OverheadPlanner(backend, client, PlannerConfig(**config), on_event=on_event)


def print_event(e: dict) -> None:
    if e["type"] == "start":
        print(f"\n▶ {e['who']}: {e['instruction']}")
    elif e["type"] == "thought":
        print(f"  thinking ({e['latency_s']}s): {e['text'][:300]}")
    elif e["type"] == "action":
        args = dict(e["args"])
        seen = args.pop("observation", None)
        if seen:
            print(f"  saw: {seen}")
        print(f"  {e['tool']} {args} -> {e['result']}")
    elif e["type"] == "plan":
        print(f"  plan {e['call']} ({e['latency_s']}s): {e['destination']}, "
              f"{e['points']} points. {e['reason']}")
    elif e["type"] == "notice":
        print(f"  note: {e['text']}")
    elif e["type"] in ("done", "error"):
        print(f"■ {e.get('summary', e.get('text'))}")
        if e.get("recording"):
            print(f"  recording: {e['recording']}")


if __name__ == "__main__":
    main()
