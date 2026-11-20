from google.genai import types

LANDMARKS = ["COLOMBO", "GALLE"]
DESTINATIONS = ["GALLE"]


def system_prompt(max_step_cm: float, max_steps: int) -> str:
    return f"""You are the navigation brain of a small wheeled rover inside a tabletop maze laid out like
a road trip across Sri Lanka. The rover starts in COLOMBO. You only see the rover's front camera.
You have no map. Navigate by reading road signs, the way a driver would.

Two kinds of sign are fixed to the walls:
- A coloured board with one name ({" or ".join(LANDMARKS)}) means "you are here".
- Green road signs. Each road leaving a junction has one green sign a short way along it, facing
  you as you look down that road from the junction. It shows only the name of a town that road
  leads to. There are no arrows and no distances: which road a sign belongs to is shown by where it hangs.
  A sign may name a town beyond your destination, or one in a different direction altogether.

How to act:
- Call exactly ONE tool per turn. After it runs you receive a fresh camera image.
- In every call, fill `observation` with the signs you can see, where they are and what they say,
  e.g. "Green sign down the left road: <town>; green sign straight ahead: <town>". Your notes from
  earlier steps are repeated to you, so write down anything you will need later.
- At a junction, look down each road and read its green sign before choosing. Use your own
  knowledge of Sri Lankan geography to decide which road leads toward the destination.
- A road that ends in a dead end was the wrong choice: turn around and go back to the junction.
- Prefer go_to_point: point at a spot ON THE FLOOR, never at a wall. Points are [y, x] normalised
  to 0-1000, [0, 0] is the top-left. Short hops, at most about {max_step_cm:.0f} cm.
- turn: positive degrees = right, negative = left. Use it to look down each road at a junction, or to turn into the road you chose.
- If a move reports blocked=true, turn away before moving again.
- Call stop with reached_goal=true when the destination's board is close and filling the view.
- Only the road to {" and ".join(DESTINATIONS)} is open today. If asked for any other place, call stop with
  reached_goal=false straight away and say that only the {" and ".join(d.title() for d in DESTINATIONS)} route is open.
  Also call stop with reached_goal=false if the instruction is unsafe or unrelated to driving in the maze.
  You have at most {max_steps} actions for this instruction."""


OBSERVATION = {"type": "string",
               "description": "Briefly: which signs you can see, where, and what they say, e.g. "
                              "'Green sign down the road ahead: <town>; green sign down the right road: <town>'."}


TOOLS = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="go_to_point",
            description="Turn toward a point on the floor in the current image and drive to it.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "y": {"type": "number", "description": "Vertical position, 0 (top) to 1000 (bottom)."},
                    "x": {"type": "number", "description": "Horizontal position, 0 (left) to 1000 (right)."},
                    "label": {"type": "string", "description": "Short name for the spot, e.g. 'corridor opening left'."},
                    "observation": OBSERVATION,
                },
                "required": ["observation", "y", "x", "label"],
            },
        ),
        types.FunctionDeclaration(
            name="turn",
            description="Rotate in place. Positive degrees turn right, negative turn left.",
            parameters_json_schema={
                "type": "object",
                "properties": {"degrees": {"type": "number", "description": "-180 to 180."},
                               "observation": OBSERVATION},
                "required": ["observation", "degrees"],
            },
        ),
        types.FunctionDeclaration(
            name="move_forward",
            description="Drive straight ahead. The rover stops early if an obstacle is too close.",
            parameters_json_schema={
                "type": "object",
                "properties": {"cm": {"type": "number", "description": "Distance in centimetres."},
                               "observation": OBSERVATION},
                "required": ["observation", "cm"],
            },
        ),
        types.FunctionDeclaration(
            name="stop",
            description="End this instruction.",
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "reached_goal": {"type": "boolean"},
                    "summary": {"type": "string", "description": "One sentence for the audience."},
                    "observation": OBSERVATION,
                },
                "required": ["observation", "reached_goal", "summary"],
            },
        ),
    ])
]
