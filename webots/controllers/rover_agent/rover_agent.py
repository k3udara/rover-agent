"""Webots controller for the rover. Webots starts it when maze.wbt opens; then open http://127.0.0.1:8765.

ROVER_MODE (in .env):
  overhead  (default) ER 2 plans the whole path from the top-down camera; code drives it
  camera    one step at a time from the e-puck's front camera

It restarts itself with the project's .venv, so Webots needs no Python setting.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV.resolve():
    if os.name == "nt":
        sys.exit(subprocess.call([str(VENV_PYTHON), __file__] + sys.argv[1:]))
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), __file__] + sys.argv[1:])

sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    sys.exit(f"Packages not found. Create {VENV} as in the README (step 2), then press reset in Webots.")

load_dotenv(ROOT / ".env")

from controller import Supervisor

from agent.core import AgentConfig, RoverAgent
from agent.keys import NoApiKey, gemini_client
from agent.server import Hub, main_loop, serve
from backends.webots_backend import WebotsBackend
from run import make_overhead_agent, print_event

try:
    client = gemini_client()
except NoApiKey as exc:
    sys.exit(f"{exc}. Then press reset in Webots.")
mode = os.environ.get("ROVER_MODE", "overhead")
robot = Supervisor()
backend = WebotsBackend(robot)
hub = Hub(cooldown_s=0)
on_event = lambda e: (hub.on_event(e), print_event(e))
if mode == "overhead":
    agent = make_overhead_agent(client, backend, on_event, runs_dir=str(ROOT / "runs"))
    hub.input_hint = "Where to? e.g. Take me to Kandy"
    preview = agent.preview_jpeg
else:
    agent = RoverAgent(backend, AgentConfig(max_step_cm=20, max_steps=30, max_seconds=300,
                                            runs_dir=str(ROOT / "runs")),
                       client=client, on_event=on_event)
    preview = None
serve(hub)
print(f"Rover agent ready (mode: {mode}): open http://127.0.0.1:8765")
main_loop(agent, hub, backend, preview_fn=preview)
