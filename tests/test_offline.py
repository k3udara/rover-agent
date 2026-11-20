import json
import queue
import shutil
import sys
import tempfile
import urllib.request
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.genai import errors, types

import agent.core as core
from agent.core import SECONDS_BETWEEN_CALLS_BY_TIER, AgentConfig, RoverAgent
from agent.geometry import CameraModel
from agent.keys import KeyPool, VaultClient, keys_from_env, parse_vault
from agent.maze import STEP, build
from agent.planner import OverheadPlanner, PlannerConfig, point_to_world, world_to_point
from agent.server import Hub, serve
from agent.tools import TOOLS, system_prompt
from backends.folder_backend import FolderBackend
from backends.sim_backend import SimBackend

ROOT = Path(__file__).resolve().parents[1]


class FakeModels:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append(contents)
        name, args = self.script.pop(0)
        return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
            role="model", parts=[
                types.Part(text=f"I see a corridor; calling {name}.", thought=True),
                types.Part(function_call=types.FunctionCall(id=f"c{len(self.calls)}", name=name, args=args)),
            ]))])


class FakeClient:
    def __init__(self, script):
        self.models = FakeModels(script)


def test_geometry():
    cam = CameraModel(hfov_deg=90, vfov_deg=60, height_cm=10, tilt_down_deg=20)
    assert abs(cam.bearing_deg(500)) < 1e-9
    assert cam.bearing_deg(1000) > 44.9 and cam.bearing_deg(0) < -44.9
    assert abs(cam.ground_distance_cm(500) - 10 / 0.36397) < 0.01
    assert cam.ground_distance_cm(900) < cam.ground_distance_cm(600)
    assert CameraModel(90, 60, 10, 0).ground_distance_cm(400) is None


def test_project_round_trip():
    cam = CameraModel(hfov_deg=80, vfov_deg=64, height_cm=2.8, tilt_down_deg=0)
    for fwd, right in [(20, 0), (15, 5), (30, -8)]:
        y, x = cam.project(fwd, right)
        turn, dist = cam.point_to_motion(y, x)
        assert abs(dist - (fwd ** 2 + right ** 2) ** 0.5) < 0.05
        assert abs(turn - __import__("math").degrees(__import__("math").atan2(right, fwd))) < 0.05


def test_agent_loop():
    photos = Path(tempfile.mkdtemp())
    for p in (ROOT / "webots/worlds/textures").glob("*.png"):
        shutil.copy(p, photos / p.name)
    backend = FolderBackend(str(photos), CameraModel(102, 67, 9, 15))
    client = FakeClient([
        ("turn", {"degrees": -30}),
        ("go_to_point", {"y": 800, "x": 600, "label": "corridor"}),
        ("go_to_point", {"y": 100, "x": 500, "label": "sky"}),
        ("stop", {"reached_goal": True, "summary": "Reached the Kandy sign."}),
    ])
    events = []
    runs = tempfile.mkdtemp()
    agent = RoverAgent(backend, AgentConfig(images_in_context=2, runs_dir=runs),
                       client=client, on_event=events.append)
    out = agent.run("Find the Kandy sign", who="tester")

    assert out == {"reached_goal": True, "summary": "Reached the Kandy sign.", "steps": 4}
    actions = [e for e in events if e["type"] == "action"]
    assert [a["tool"] for a in actions] == ["turn", "go_to_point", "go_to_point", "stop"]
    assert actions[1]["result"]["status"] == "ok" and actions[1]["result"]["moved_cm"] <= 30
    assert actions[2]["result"]["status"] == "turned_only"
    last = client.models.calls[-1]
    fr = [p.function_response for c in last for p in c.parts if p.function_response]
    assert fr[-1].id == "c3"
    assert sum(1 for c in last for p in c.parts if p.inline_data) == 2
    assert len(list(Path(runs).rglob("*.jpg"))) == 4


def test_server_queue():
    hub = Hub(cooldown_s=60)
    srv = serve(hub, port=0)
    port = srv.server_address[1]

    def post(name, text):
        req = urllib.request.Request(f"http://127.0.0.1:{port}/instruct", method="POST",
                                     data=json.dumps({"name": name, "text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            return json.load(urllib.request.urlopen(req))
        except urllib.error.HTTPError as e:
            return json.load(e)

    assert post("Amaya", "Go to Ella")["ok"]
    assert not post("amaya", "again")["ok"]
    assert post("Nimal", "x" * 500)["ok"]
    item = hub.next()
    assert item["name"] == "Amaya"
    assert not post("Amaya", "Go to Galle")["ok"]
    assert len(hub.status()["queue"][0]["text"]) == 200
    html = urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
    assert b"What the robot sees" in html
    srv.shutdown()


DECOYS = {"NEGOMBO", "KANDY", "RATNAPURA", "NUWARA ELIYA"}
ON_ROUTE = {"KALUTARA", "BENTOTA", "HIKKADUWA", "GALLE"}


def _route(lay):
    return lay.path(lay.cell_of(*lay.start[:2]), lay.sign("GALLE").cell)


def test_each_junction_has_one_right_road_and_one_decoy():
    lay = build()
    path = _route(lay)
    junctions = [c for c in path if len(lay.open_sides(c)) >= 3]
    assert len(junctions) == 4 and {s.junction for s in lay.route_signs} == set(junctions)
    for k, j in enumerate(junctions):
        signs = [s for s in lay.route_signs if s.junction == j]
        right = [s for s in signs if s.cell in path]
        wrong = [s for s in signs if s.cell not in path]
        assert len(right) == 1 and len(wrong) == 1, j
        assert right[0].name in ON_ROUTE and wrong[0].name in DECOYS, (j, right[0].name, wrong[0].name)
        assert (right[0].name == "GALLE") == (k == len(junctions) - 1)
        seen, todo = {j}, [wrong[0].cell]
        while todo:
            c = todo.pop()
            seen.add(c)
            todo += [n for n in lay.neighbours(c) if n not in seen]
        assert lay.sign("GALLE").cell not in seen and len(seen) <= 3, (wrong[0].name, seen)


def test_signs_are_names_only():
    lay = build()
    assert {s.name for s in lay.route_signs} == ON_ROUTE | DECOYS
    for s in lay.route_signs:
        assert s.name.replace(" ", "").isalpha() and s.name.isupper(), s.name


def test_correct_road_is_not_always_the_same_turn():
    lay = build()
    path = _route(lay)
    turns = {"right": [], "wrong": []}
    for i, c in enumerate(path):
        if len(lay.open_sides(c)) < 3:
            continue
        came = (c[0] - path[i - 1][0], c[1] - path[i - 1][1])
        for s in (s for s in lay.route_signs if s.junction == c):
            out = STEP[s.exit]
            cross = came[0] * out[1] - came[1] * out[0]
            kind = "straight" if out == came else ("left" if cross > 0 else "right")
            turns["right" if s.cell in path else "wrong"].append(kind)
    assert len(set(turns["right"])) > 1 and len(set(turns["wrong"])) > 1, turns


def test_one_sign_per_wall_face():
    lay = build()
    faces = [(s.cell, s.wall) for s in lay.all_signs()]
    assert len(faces) == len(set(faces))
    for s in lay.all_signs():
        assert s.wall not in lay.open_sides(s.cell), s.id


def test_road_sign_rules_fail_loudly():
    maze = """
+--+--+--+
|        |
+--+  +--+
|  |R |  |
+--+--+--+
"""
    lay = build(maze, {((0, 1), "W"): "KANDY", ((0, 1), "E"): "GALLE"})
    sign = next(s for s in lay.route_signs if s.name == "KANDY")
    assert (sign.cell, sign.wall, sign.junction) == ((0, 0), "W", (0, 1))
    for bad in ({((0, 1), "W"): "KANDY"},
                {((0, 1), "N"): "KANDY", ((0, 1), "E"): "GALLE"},
                {((0, 0), "E"): "KANDY"}):
        try:
            build(maze, bad)
        except ValueError:
            continue
        raise AssertionError(f"no error for {bad}")


def test_prompt_gives_no_route_hints():
    text = system_prompt(20, 30).upper()
    for town in (ON_ROUTE | DECOYS) - {"GALLE"}:
        assert town not in text, town
    assert "ARROW" not in text.replace("NO ARROWS", "")
    assert "ONLY THE ROAD TO GALLE IS OPEN" in text


def test_every_tool_requires_observation():
    for fd in TOOLS[0].function_declarations:
        schema = fd.parameters_json_schema
        assert "observation" in schema["properties"] and "observation" in schema["required"], fd.name


def test_notes_carry_observations_forward():
    photos = Path(tempfile.mkdtemp())
    for p in (ROOT / "webots/worlds/textures").glob("*.png"):
        shutil.copy(p, photos / p.name)
    backend = FolderBackend(str(photos), CameraModel(102, 67, 9, 15))
    client = FakeClient([
        ("turn", {"degrees": 90, "observation": "Green sign down the right road: KALUTARA"}),
        ("move_forward", {"cm": 10, "observation": "corridor"}),
        ("stop", {"reached_goal": False, "summary": "done", "observation": "nothing"}),
    ])
    RoverAgent(backend, AgentConfig(images_in_context=1, runs_dir=tempfile.mkdtemp()), client=client).run("Galle")
    last_text = " ".join(p.text for c in client.models.calls[-1] for p in c.parts if p.text)
    assert "Your notes so far" in last_text and "Green sign down the right road: KALUTARA" in last_text


class _PretendGoogleModels(FakeModels):
    pass


_PretendGoogleModels.__module__ = "google.genai.pretend"


class _PretendGoogleClient:
    def __init__(self, script):
        self.models = _PretendGoogleModels(script)


_PretendGoogleClient.__module__ = "google.genai.pretend"


def _record_sleeps():
    slept, real = [], core.time.sleep
    core.time.sleep = slept.append
    return slept, lambda: setattr(core.time, "sleep", real)


def test_api_tier_switch():
    script = [("turn", {"degrees": 10, "observation": "x"})] * 3 + [("stop", {"reached_goal": False, "summary": "s"})]
    for tier, expect_wait in (("free", True), ("paid", False)):
        slept, restore = _record_sleeps()
        try:
            backend = FolderBackend(str(ROOT / "webots/worlds/textures"), CameraModel(102, 67, 9, 15))
            RoverAgent(backend, AgentConfig(api_tier=tier, max_steps=4, runs_dir=tempfile.mkdtemp()),
                       client=_PretendGoogleClient(script)).run("Galle")
        finally:
            restore()
        gap = SECONDS_BETWEEN_CALLS_BY_TIER["free"]
        assert (len([s for s in slept if s > gap - 1]) == 3) == expect_wait, (tier, slept)
    try:
        RoverAgent(FolderBackend(str(ROOT / "webots/worlds/textures"), CameraModel(102, 67, 9, 15)),
                   AgentConfig(api_tier="premium"), client=FakeClient([]))
    except ValueError as e:
        assert "free" in str(e) and "paid" in str(e)
    else:
        raise AssertionError("an unknown ROVER_API_TIER should be rejected")


def test_daily_quota_stops_without_retrying():

    class DailyLimitModels:
        calls = 0

        def generate_content(self, **kw):
            DailyLimitModels.calls += 1
            raise errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message":
                                    "quota", "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}})
    DailyLimitModels.__module__ = "google.genai.pretend"
    client = type("C", (), {"__module__": "google.genai.pretend"})()
    client.models = DailyLimitModels()
    slept, restore = _record_sleeps()
    try:
        backend = FolderBackend(str(ROOT / "webots/worlds/textures"), CameraModel(102, 67, 9, 15))
        out = RoverAgent(backend, AgentConfig(api_tier="paid", runs_dir=tempfile.mkdtemp()), client=client).run("Galle")
    finally:
        restore()
    assert DailyLimitModels.calls == 1 and not slept, (DailyLimitModels.calls, slept)
    assert "daily request quota" in out["summary"] and "paid tier" in out["summary"]


def _daily_quota_error():
    return errors.ClientError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "quota",
                                              "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}})


def _chunk(text: str, thought: bool = False):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(
        role="model", parts=[types.Part(text=text, thought=thought or None)]))])


class ScriptedPlannerModels:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, 0

    def generate_content_stream(self, model, contents, config=None):
        self.calls += 1
        if self.error:
            raise self.error
        text = json.dumps(self.reply)
        return iter([_chunk("thinking...", thought=True), _chunk(text[:10]), _chunk(text[10:])])


def _planner(client, events=None):
    sim = client.models.backend if hasattr(client.models, "backend") else SimBackend(realtime=False)
    return sim, OverheadPlanner(sim, client, PlannerConfig(runs_dir=tempfile.mkdtemp(), api_tier="paid"),
                                on_event=(events.append if events is not None else None))


def test_vault_parsing():
    assert parse_vault('["AQ.a", "AQ.b"]') == ["AQ.a", "AQ.b"]
    assert parse_vault("AQ.a,AQ.b\nAQ.c") == ["AQ.a", "AQ.b", "AQ.c"]
    assert parse_vault("[\n  AQ.a,\n  AQ.b,\n  AQ.a,\n  ]") == ["AQ.a", "AQ.b"]
    assert parse_vault("") == [] and parse_vault(None) == [] and parse_vault("[ ]") == []
    assert keys_from_env({"GOOGLE_API_KEY_VAULT": "[AQ.a, AQ.b]"}) == ["AQ.a", "AQ.b"]
    assert keys_from_env({}) == []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert keys_from_env({"GOOGLE_API_KEY": "AQ.b", "GEMINI_API_KEY": "AQ.c"}) == []
    assert len(caught) == 2 and all("GOOGLE_API_KEY_VAULT" in str(w.message) for w in caught)


def test_key_rotation_and_state():
    now = [datetime(2026, 11, 18, 9, 0, tzinfo=timezone.utc)]
    state = Path(tempfile.mkdtemp()) / "state.json"
    pool = KeyPool(["AQ.secret-one", "AQ.secret-two"], state, clock=lambda: now[0])
    used = []

    class FakeGenai:
        def __init__(self, key):
            self.models = self
            self.key = key

        def generate_content(self, **kw):
            used.append(self.key)
            if self.key == "AQ.secret-one":
                raise _daily_quota_error()
            return "ok"

    notices = []
    vault = VaultClient(pool, on_switch=notices.append)
    vault._client = FakeGenai
    assert vault.models.generate_content(model="m", contents=[]) == "ok"
    assert used == ["AQ.secret-one", "AQ.secret-two"] and vault.slot_in_use == 2
    assert notices == ["key slot 1 of 2 has used up today's quota; switching to key slot 2 of 2."]
    assert "secret" not in state.read_text()
    used.clear()
    vault.models.generate_content(model="m", contents=[])
    assert used == ["AQ.secret-two"]
    now[0] += timedelta(hours=23)
    assert pool.current() == (1, "AQ.secret-one")


def test_all_keys_used_up_stops_the_trip():
    sim = SimBackend(realtime=False)
    vault = VaultClient(KeyPool(["AQ.only-key"], Path(tempfile.mkdtemp()) / "s.json"))
    vault._client = lambda key: type("C", (), {"models": ScriptedPlannerModels(error=_daily_quota_error())})()
    planner = OverheadPlanner(sim, vault, PlannerConfig(runs_dir=tempfile.mkdtemp(), api_tier="paid"))
    before = sim.true_pose()
    out = planner.run("Take me to Kandy")
    assert not out["reached_goal"] and "used up today's ER 2 quota" in out["summary"], out
    assert sim.true_pose() == before


class RouteModels:
    def __init__(self, sim):
        self.sim, self.town = sim, None

    def generate_content_stream(self, model, contents, config=None):
        lay = self.sim.lay
        cells = lay.path(lay.cell_of(self.sim.x, self.sim.y), lay.towns[self.town])
        points = [list(world_to_point(*lay.centre(c), self.sim.overhead_span_m)) for c in cells]
        return iter([_chunk(json.dumps({"destination": self.town, "reason": "shortest", "points": points}))])


def test_planner_drives_to_every_town():
    sim = SimBackend(realtime=False)
    models = RouteModels(sim)
    models.backend = sim
    _, planner = _planner(type("C", (), {"models": models})())
    for town in sim.lay.towns:
        sim.reset()
        models.town = town
        out = planner.run(f"Take me to {town.title()}")
        assert out["reached_goal"], (town, out)
        assert sim.lay.cell_of(sim.x, sim.y) == sim.lay.towns[town]
    sim.reset()
    for town in ("GALLE", "KANDY", "NUWARA ELIYA"):
        models.town = town
        out = planner.run(f"Now go to {town.title()}")
        assert out["reached_goal"] and sim.lay.cell_of(sim.x, sim.y) == sim.lay.towns[town], (town, out)


def test_overhead_point_round_trip():
    span = 1.35
    for x, y in [(0, 0), (-0.5, -0.375), (0.5, 0.375), (0.123, -0.456)]:
        back = point_to_world(*world_to_point(x, y, span), span)
        assert abs(back[0] - x) < 0.01 and abs(back[1] - y) < 0.01, (x, y, back)
    assert world_to_point(-span / 2, span / 2, span) == (0, 0)


def test_path_through_a_wall_replans_and_stops_after_three():
    sim = SimBackend(realtime=False)
    kandy = sim.lay.centre(sim.lay.towns["KANDY"])
    straight_through_walls = {"destination": "KANDY", "reason": "straight there",
                              "points": [list(world_to_point(sim.x, sim.y, sim.overhead_span_m)),
                                         list(world_to_point(*kandy, sim.overhead_span_m))]}
    models = ScriptedPlannerModels(straight_through_walls)
    models.backend = sim
    events = []
    _, planner = _planner(type("C", (), {"models": models})(), events=events)
    out = planner.run("Take me to Kandy")
    assert not out["reached_goal"] and models.calls == 3, (out, models.calls)
    assert sum(1 for e in events if e["type"] == "notice" and "Blocked" in e["text"]) == 3
    assert "after 3 plans" in out["summary"]


def test_unknown_place_is_refused_without_driving():
    sim = SimBackend(realtime=False)
    models = ScriptedPlannerModels({"destination": None, "reason": "Jaffna is not on this map.", "points": []})
    models.backend = sim
    _, planner = _planner(type("C", (), {"models": models})())
    before = sim.true_pose()
    out = planner.run("Take me to Jaffna")
    assert not out["reached_goal"] and "isn't on this map" in out["summary"] and "Kandy" in out["summary"], out
    assert sim.true_pose() == before and models.calls == 1


def test_trip_is_recorded_without_key_text():
    sim = SimBackend(realtime=False)
    route = [sim.lay.centre(c) for c in [(3, 0), (2, 0), (2, 1)]]
    plan = {"destination": "KALUTARA", "reason": "up, then right",
            "points": [list(world_to_point(*p, sim.overhead_span_m)) for p in route]}
    vault = VaultClient(KeyPool(["AQ.super-secret-key"], Path(tempfile.mkdtemp()) / "s.json"))
    vault._client = lambda key: type("C", (), {"models": ScriptedPlannerModels(plan)})()
    planner = OverheadPlanner(sim, vault, PlannerConfig(runs_dir=tempfile.mkdtemp(), api_tier="paid"))
    out = planner.run("Take me to Kalutara", who="Test Person")
    assert out["reached_goal"], out
    folder = Path(out["recording"]).parent
    assert folder.name.endswith("-test-person-kalutara"), folder.name
    names = {f.name for f in folder.iterdir()}
    assert {"plan_1.jpg", "plan_1.json", "trip.json"} <= names and names & {"trip.mp4", "trip.gif"}, names
    record = json.loads((folder / "plan_1.json").read_text())
    assert record["key"] == "key slot 1 of 1" and record["parsed"]["destination"] == "KALUTARA"
    for f in folder.iterdir():
        assert b"super-secret" not in f.read_bytes(), f.name


def test_planner_stream_reaches_event_listeners():
    hub = Hub()
    listener = queue.Queue()
    hub.listeners.append(listener)
    hub.on_event({"type": "stream", "call": 1, "kind": "answer", "text": "{\"destination\""})
    hub.on_event({"type": "frame", "image": b"jpeg"})
    assert listener.get_nowait()["text"] == "{\"destination\"" and listener.empty()
    assert hub.frame == b"jpeg"


if __name__ == "__main__":
    test_geometry(); test_project_round_trip(); test_agent_loop(); test_server_queue()
    test_each_junction_has_one_right_road_and_one_decoy(); test_signs_are_names_only()
    test_correct_road_is_not_always_the_same_turn(); test_one_sign_per_wall_face()
    test_road_sign_rules_fail_loudly(); test_prompt_gives_no_route_hints()
    test_every_tool_requires_observation(); test_notes_carry_observations_forward()
    test_api_tier_switch(); test_daily_quota_stops_without_retrying()
    test_vault_parsing(); test_key_rotation_and_state(); test_all_keys_used_up_stops_the_trip()
    test_planner_drives_to_every_town(); test_overhead_point_round_trip()
    test_path_through_a_wall_replans_and_stops_after_three(); test_unknown_place_is_refused_without_driving()
    test_trip_is_recorded_without_key_text(); test_planner_stream_reaches_event_listeners()
    print("all offline tests passed")
