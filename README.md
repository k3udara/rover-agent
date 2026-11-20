# Maze Rover with Gemini Robotics-ER

Tell a robot where to go in plain words, like *"Take me to Kandy"* or *"I want to see the tea plantations"*.
A camera above a small Sri Lankan road maze takes one picture. **Gemini Robotics-ER** reads the town names
on the floor, works out which town you mean, and draws the whole route. Then the code drives the robot
along that route in the [Webots](https://cyberbotics.com) simulator, and every trip is saved as a video.

## What you need

- **Python 3.10 or newer**
- **Webots R2025a or newer** (free, from [cyberbotics.com](https://cyberbotics.com)). If it doesn't start on a
  recent macOS, use a newer build from [Webots releases](https://github.com/cyberbotics/webots/releases).
- **A Gemini API key**, free from [Google AI Studio](https://aistudio.google.com/apikey)
- **Optional:** [ffmpeg](https://ffmpeg.org) for MP4 videos (without it, trips are saved as GIFs)

## Setup (once)

**1. Get the code**

```bash
git clone https://github.com/stemlink-engineering/rover-agent.git
cd rover-agent
```

**2. Create the Python environment.** It must be called `.venv` and sit in this folder, because Webots looks
for it there.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
```

**3. Add your API key**

```bash
cp .env.example .env                              # Windows: copy .env.example .env
```

Open `.env` and put your key inside the brackets: `GOOGLE_API_KEY_VAULT='[AQ.your-key]'`. You can list
several keys, separated by commas. Every other setting is listed in that file with a
short explanation. The values it starts with are the right ones for a free key.

## Run it in Webots

1. Open Webots, then **File → Open World** and pick `webots/worlds/maze.wbt`.
2. Wait for the Webots console to say `Rover agent ready (mode: overhead)`.
3. Open **http://127.0.0.1:8765** in your browser, type your name and where to go, and press send.
   You can pick any of the 9 towns: Colombo, Negombo, Kalutara, Bentota, Kandy, Hikkaduwa, Ratnapura, Galle and Nuwara Eliya.
4. Watch the planned route appear in blue and the robot drive it. Press reset (⏮) in Webots to put the
   robot back at Colombo.

## Run it without Webots

The same Gemini planner and web page also run with a simple built-in simulator:

```bash
.venv/bin/python run.py sim                      # then open http://127.0.0.1:8765
```

Close Webots first: both use port 8765.

## Your recordings

Every trip is saved in `runs/<time>-<name>-<town>/` on your computer, and is listed on the web page:

| File | What's in it |
|---|---|
| `trip.mp4` | the trip seen from above: planned route in blue, the robot's actual path in green |
| `plan_1.jpg` | the exact picture Gemini was shown |
| `plan_1.json` | Gemini's answer, its thinking, and how long it took |
| `trip.json` | where the robot started, where it ended, and whether it arrived |

## How it works

- **Gemini plans, code drives.** Gemini returns the route as points on the picture. The code turns
  toward each point and drives there.
- **Safety never depends on the model.** The robot's own sensors stop it before it hits a wall.
- **If the robot gets stuck,** it takes a new picture and asks Gemini again, at most 3 times per trip.
- **The model** is `gemini-robotics-er-2-preview`. On the free tier one key gives about 20 requests a
  day, which is about 7 to 20 trips. When every key's quota is used up, trips stop with a message until
  the next day.
- **To find its way, the code reads the robot's exact position from the simulator.** On a real robot, an overhead camera would
  track it instead.

| To change... | Edit |
|---|---|
| What Gemini is asked, how the route is followed | `agent/planner.py` |
| The maze, the towns, the signs | `agent/maze.py`, then run `.venv/bin/python webots/make_world.py` |
| How the robot moves in Webots | `backends/webots_backend.py` |
| The web page | `agent/server.py` |

After editing, press reset (⏮) in Webots to run your new code. To check nothing broke, run
`.venv/bin/python tests/test_offline.py`. It uses no API calls and should end with `all offline tests passed`.

## If something goes wrong

- **The page at :8765 doesn't load:** check the Webots console (bottom panel) for an error. "Address
  already in use" means something else is using the port: close the other Webots window or the
  `run.py sim` terminal.
- **"No Gemini API key" in the Webots console:** step 3 of Setup. Add your key to `.env`, then
  press reset (⏮).
- **"Packages not found" in the Webots console:** step 2 of Setup wasn't done in this folder. Do it,
  then press reset (⏮).
- **Gemini errors (403 or 429):** check the key in `.env`. A 429 means the quota is used up. Wait a
  minute, or until the next day.

## More

- **Step-by-step mode:** set `ROVER_MODE=camera` in `.env`. The robot then decides one move at a time
  from its own front camera and the road signs, instead of planning from above.
- **Several keys:** when one key in `GOOGLE_API_KEY_VAULT` runs out of its daily quota, the next is
  used. Keys that have run out are remembered until midnight Pacific time.
- **Real hardware:** `run.py photos` (a folder of photos) and `run.py pi` (Raspberry Pi rover). See the
  notes at the top of `backends/folder_backend.py` and `backends/pi_backend.py`.

The web page has no login and only listens on your own computer (127.0.0.1). Keep it that way.
