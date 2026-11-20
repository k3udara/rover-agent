from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_TEXT = 200
MAX_QUEUE = 30
MAX_RECORDINGS_LISTED = 8
EVENT_KEEPALIVE_SECONDS = 15
VIDEO_TYPES = {".mp4": "video/mp4", ".gif": "image/gif"}


class Hub:
    def __init__(self, cooldown_s: float = 120.0):
        self.lock = threading.Lock()
        self.queue: deque[dict] = deque()
        self.log: deque[str] = deque(maxlen=60)
        self.frame: bytes | None = None
        self.current: dict | None = None
        self.cooldown_s = cooldown_s
        self.last_turn: dict[str, float] = {}
        self.overhead_fn = None
        self.input_hint = "Instruction, e.g. Take me to Galle"
        self.recordings: deque[Path] = deque(maxlen=MAX_RECORDINGS_LISTED)
        self.listeners: list[queue.Queue] = []

    def publish(self, message: dict) -> None:
        for listener in list(self.listeners):
            listener.put(message)

    def submit(self, name: str, text: str) -> tuple[bool, str]:
        name, text = name.strip()[:30] or "anonymous", text.strip()[:MAX_TEXT]
        if not text:
            return False, "Empty instruction."
        key = name.lower()
        with self.lock:
            if len(self.queue) >= MAX_QUEUE:
                return False, "Queue is full, try again in a minute."
            if any(item["name"].lower() == key for item in self.queue):
                return False, "You already have an instruction waiting."
            if time.time() - self.last_turn.get(key, 0) < self.cooldown_s:
                return False, "Please wait before your next turn."
            self.queue.append({"name": name, "text": text})
            return True, f"Queued at position {len(self.queue)}."

    def next(self) -> dict | None:
        with self.lock:
            if not self.queue:
                return None
            item = self.queue.popleft()
            self.current = item
            self.last_turn[item["name"].lower()] = time.time()
            return item

    def on_event(self, event: dict) -> None:
        with self.lock:
            kind = event["type"]
            if kind == "start":
                self.log.append(f"▶ {event['who']}: {event['instruction']}")
            elif kind == "thought" and event["text"]:
                self.log.append(f"💭 ({event['latency_s']}s) {event['text'][:400]}")
            elif kind == "action":
                self.frame = event["image"]
                args = dict(event["args"])
                seen = args.pop("observation", None)
                if seen:
                    self.log.append(f"👀 {str(seen)[:300]}")
                self.log.append(f"🛞 {event['tool']} {args} → {event['result']}")
            elif kind == "error":
                self.log.append(f"⚠ {event['text']}")
            elif kind == "plan_start":
                self.log.append(f"🗺 Plan {event['call']}: asking the {event['brain']}…")
            elif kind == "plan":
                where = (event["destination"] or "not on the map").title()
                self.log.append(f"🗺 Plan {event['call']} ({event['brain']}, {event['latency_s']}s): {where}, "
                                f"{event['points']} points. {event['reason']}")
            elif kind == "notice":
                self.log.append(f"ℹ {event['text']}")
            elif kind == "frame":
                self.frame = event["image"]
            elif kind == "done":
                mark = "✅" if event["reached_goal"] else "⏹"
                self.log.append(f"{mark} {event['summary']}")
                if event.get("recording"):
                    self.recordings.appendleft(Path(event["recording"]).resolve())
                self.current = None
        if kind in ("start", "plan_start", "stream", "done"):
            self.publish({k: v for k, v in event.items() if k != "image"})

    def set_frame(self, jpeg: bytes) -> None:
        with self.lock:
            self.frame = jpeg

    def status(self) -> dict:
        with self.lock:
            return {"current": self.current, "queue": list(self.queue), "log": list(self.log),
                    "recordings": [f"{r.parent.name}/{r.name}" for r in self.recordings]}

    def recording_file(self, folder: str, name: str) -> Path | None:
        with self.lock:
            return next((r for r in self.recordings if r.parent.name == folder and r.name == name), None)


PAGE = """<!doctype html><meta charset=utf-8><title>Rover</title>
<style>body{font-family:system-ui;margin:16px;background:#111;color:#eee;display:grid;grid-template-columns:2fr 1fr;gap:16px}
img{width:100%;border-radius:8px;background:#222}pre{white-space:pre-wrap;font-size:13px;max-height:60vh;overflow:auto}
#live{max-height:30vh;background:#1b1b1b;padding:8px;border-radius:8px}#live .t{color:#999;font-style:italic}
a{color:#8ab4f8}input,button{font-size:16px;padding:8px;margin:4px 0;width:100%;box-sizing:border-box}</style>
<div><h2>What the robot sees</h2><img id=f src=/frame.jpg><h3 id=cur></h3>
<div id=ov><h2>Overhead</h2><img id=o src=/overhead.jpg onerror="this.parentNode.style.display='none'" style="max-width:420px"></div></div>
<div><form id=s><input id=n placeholder="Your name"><input id=t placeholder="__HINT__">
<button>Send</button><div id=msg></div></form><h3>Planner, live</h3><pre id=live></pre>
<h3>Queue</h3><pre id=q></pre><h3>Recordings</h3><div id=rec></div><h3>Log</h3><pre id=l></pre></div>
<script>
async function tick(){document.getElementById('f').src='/frame.jpg?'+Date.now();
if(document.getElementById('ov').style.display!='none')document.getElementById('o').src='/overhead.jpg?'+Date.now();
const s=await (await fetch('/status')).json();
document.getElementById('cur').textContent=s.current?('Now: '+s.current.name+' — '+s.current.text):'Waiting for the next instruction';
document.getElementById('q').textContent=s.queue.map((x,i)=>(i+1)+'. '+x.name+': '+x.text).join('\\n');
document.getElementById('rec').innerHTML=s.recordings.map(r=>'<div><a target=_blank href="/recordings/'+r+'">'+r+'</a></div>').join('');
document.getElementById('l').textContent=s.log.slice().reverse().join('\\n\\n');}
setInterval(tick,700);tick();
const live=document.getElementById('live');
const es=new EventSource('/events');es.onmessage=e=>{const m=JSON.parse(e.data);
if(m.type=='start'){live.textContent='';}
else if(m.type=='plan_start'){live.append('\\n— plan '+m.call+' ('+m.brain+') —\\n');}
else if(m.type=='stream'){const s=document.createElement('span');if(m.kind=='thought')s.className='t';s.textContent=m.text;live.append(s);}
else if(m.type=='done'){live.append('\\n'+(m.reached_goal?'✅ ':'⏹ ')+m.summary);}
live.scrollTop=live.scrollHeight;};
document.getElementById('s').onsubmit=async e=>{e.preventDefault();
const r=await fetch('/instruct',{method:'POST',headers:{'Content-Type':'application/json'},
body:JSON.stringify({name:n.value,text:t.value})});document.getElementById('msg').textContent=(await r.json()).message;t.value='';};
</script>"""


def serve(hub: Hub, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                page = PAGE.replace("__HINT__", hub.input_hint.replace('"', "'"))
                self._send(200, page.encode(), "text/html; charset=utf-8")
            elif path == "/frame.jpg" and hub.frame:
                self._send(200, hub.frame, "image/jpeg")
            elif path == "/overhead.jpg" and hub.overhead_fn:
                self._send(200, hub.overhead_fn(), "image/jpeg")
            elif path == "/status":
                self._send(200, json.dumps(hub.status()).encode(), "application/json")
            elif path == "/events":
                self._stream_events()
            elif path.startswith("/recordings/") and path.count("/") == 3:
                _, _, folder, name = path.split("/")
                file = hub.recording_file(folder, name)
                if file and file.exists():
                    self._send(200, file.read_bytes(), VIDEO_TYPES.get(file.suffix, "application/octet-stream"))
                else:
                    self._send(404, b"not found", "text/plain")
            else:
                self._send(404, b"not found", "text/plain")

        def _stream_events(self):
            listener: queue.Queue = queue.Queue()
            hub.listeners.append(listener)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                while True:
                    try:
                        message = listener.get(timeout=EVENT_KEEPALIVE_SECONDS)
                        self.wfile.write(f"data: {json.dumps(message)}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                hub.listeners.remove(listener)

        def do_POST(self):
            if self.path != "/instruct":
                return self._send(404, b"not found", "text/plain")
            try:
                length = min(int(self.headers.get("Content-Length", 0)), 4096)
                data = json.loads(self.rfile.read(length) or b"{}")
                ok, msg = hub.submit(str(data.get("name", "")), str(data.get("text", "")))
            except (ValueError, json.JSONDecodeError):
                ok, msg = False, "Bad request."
            self._send(200 if ok else 400, json.dumps({"ok": ok, "message": msg}).encode(),
                       "application/json")

    ThreadingHTTPServer.daemon_threads = True
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main_loop(agent, hub: Hub, backend, preview_every_s: float = 0.5, preview_fn=None) -> None:
    preview_fn = preview_fn or backend.capture_jpeg
    last_preview = 0.0
    while True:
        item = hub.next()
        if item:
            agent.run(item["text"], who=item["name"])
            continue
        if time.monotonic() - last_preview > preview_every_s:
            hub.set_frame(preview_fn())
            last_preview = time.monotonic()
        if not backend.idle(0.1):
            return
