from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFont

MAZE = """
+--+--+--+--+--+
|  |           |
+--+  +--+  +  +
|  |     |  |  |
+  +  +  +--+  +
|     |  |     |
+  +--+--+  +  +
|RC|G       |  |
+--+--+--+--+--+
"""

CELL = 0.25
WALL_T = 0.012
WALL_H = 0.08
SIGN_W, SIGN_H, SIGN_Z = 0.13, 0.065, 0.04

LANDMARKS = {
    "C": ("COLOMBO", (219, 68, 55)),
    "G": ("GALLE", (156, 39, 176)),
}

ROUTE_SIGNS = {
    ((2, 0), "E"): "KALUTARA",  ((2, 0), "N"): "NEGOMBO",
    ((1, 1), "N"): "BENTOTA",   ((1, 1), "E"): "KANDY",
    ((0, 3), "E"): "HIKKADUWA", ((0, 3), "S"): "RATNAPURA",
    ((2, 4), "W"): "GALLE",     ((2, 4), "S"): "NUWARA ELIYA",
}
ROUTE_GREEN = (0, 110, 60)

LABEL_W, LABEL_H = 0.21, 0.07
LABEL_OFFSET_SOUTH = 0.07
LABEL_FLOOR = (225, 222, 205)
LABEL_INK = (25, 25, 25)

SIDES = ("N", "E", "S", "W")
STEP = {"N": (-1, 0), "S": (1, 0), "W": (0, -1), "E": (0, 1)}
OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}
HEADING_DEG = {"E": 0.0, "N": 90.0, "W": 180.0, "S": -90.0}


@dataclass
class Sign:
    name: str
    colour: tuple[int, int, int]
    cell: tuple[int, int]
    x: float
    y: float
    facing_deg: float
    wall: str = ""
    kind: str = "landmark"
    w: float = SIGN_W
    h: float = SIGN_H

    @property
    def id(self) -> str:
        return f"r{self.cell[0]}c{self.cell[1]}-{self.wall}"

    @property
    def text(self) -> str:
        return self.name

    @property
    def plain_text(self) -> str:
        return self.name

    @property
    def entries(self) -> list:
        return []

    def image(self) -> Image.Image:
        return sign_image(self.name, self.colour)


@dataclass
class RouteSign:
    name: str
    cell: tuple[int, int]
    wall: str
    x: float
    y: float
    facing_deg: float
    junction: tuple[int, int]
    exit: str
    kind: str = "route"
    w: float = SIGN_W
    h: float = SIGN_H

    @property
    def id(self) -> str:
        return f"r{self.cell[0]}c{self.cell[1]}-{self.wall}"

    @property
    def text(self) -> str:
        return self.name

    @property
    def plain_text(self) -> str:
        return self.name

    def image(self) -> Image.Image:
        return route_sign_image(self.name, self.w / self.h)

@dataclass
class Layout:
    rows: int
    cols: int
    h_walls: set
    v_walls: set
    signs: list[Sign] = field(default_factory=list)
    start: tuple[float, float, float] = (0.0, 0.0, 90.0)
    route_signs: list[RouteSign] = field(default_factory=list)
    towns: dict = field(default_factory=dict)

    def label_centre(self, name: str) -> tuple[float, float]:
        x, y = self.centre(self.towns[name])
        return x, y - LABEL_OFFSET_SOUTH

    def all_signs(self) -> list:
        return list(self.signs) + list(self.route_signs)

    def open_sides(self, cell) -> list[str]:
        r, c = cell
        return [s for s in SIDES if (r + STEP[s][0], c + STEP[s][1]) in set(self.neighbours(cell))]

    def wall_point(self, cell, side: str) -> tuple[float, float]:
        a = math.radians(HEADING_DEG[side])
        off = CELL / 2 - WALL_T / 2 - 0.003
        cx, cy = self.centre(cell)
        return cx + off * math.cos(a), cy + off * math.sin(a)

    def is_decision_cell(self, cell) -> bool:
        sides = self.open_sides(cell)
        return len(sides) >= 3 or (len(sides) == 2 and OPPOSITE[sides[0]] != sides[1])

    def cx(self, c: float) -> float:
        return (c - (self.cols - 1) / 2) * CELL

    def cy(self, r: float) -> float:
        return ((self.rows - 1) / 2 - r) * CELL

    def cell_of(self, x: float, y: float) -> tuple[int, int]:
        c = round(x / CELL + (self.cols - 1) / 2)
        r = round((self.rows - 1) / 2 - y / CELL)
        return min(max(r, 0), self.rows - 1), min(max(c, 0), self.cols - 1)

    def centre(self, cell: tuple[int, int]) -> tuple[float, float]:
        return self.cx(cell[1]), self.cy(cell[0])

    def wall_boxes(self):
        out = []
        for r, c in sorted(self.h_walls):
            y = self.cy(r) + CELL / 2 if r < self.rows else self.cy(self.rows - 1) - CELL / 2
            out.append((self.cx(c), y, CELL + WALL_T, WALL_T))
        for r, c in sorted(self.v_walls):
            x = self.cx(c) - CELL / 2 if c < self.cols else self.cx(self.cols - 1) + CELL / 2
            out.append((x, self.cy(r), WALL_T, CELL + WALL_T))
        return out

    def wall_segments(self):
        segs = []
        for x, y, sx, sy in self.wall_boxes():
            if sx > sy:
                segs.append((x - sx / 2, y, x + sx / 2, y))
            else:
                segs.append((x, y - sy / 2, x, y + sy / 2))
        return segs

    def neighbours(self, cell):
        r, c = cell
        if r > 0 and (r, c) not in self.h_walls:
            yield (r - 1, c)
        if r < self.rows - 1 and (r + 1, c) not in self.h_walls:
            yield (r + 1, c)
        if c > 0 and (r, c) not in self.v_walls:
            yield (r, c - 1)
        if c < self.cols - 1 and (r, c + 1) not in self.v_walls:
            yield (r, c + 1)

    def path(self, start, goal):
        prev, todo = {start: None}, deque([start])
        while todo:
            cur = todo.popleft()
            if cur == goal:
                break
            for n in self.neighbours(cur):
                if n not in prev:
                    prev[n] = cur
                    todo.append(n)
        if goal not in prev:
            return []
        out, cur = [], goal
        while cur is not None:
            out.append(cur)
            cur = prev[cur]
        return out[::-1]

    def sign(self, name: str) -> Sign | None:
        return next((s for s in self.signs if s.name == name.upper()), None)


def build(maze: str = MAZE, route_table: dict | None = None) -> Layout:
    lines = maze.strip("\n").splitlines()
    rows, cols = (len(lines) - 1) // 2, (len(lines[0]) - 1) // 3
    lay = Layout(rows, cols, set(), set())
    cells = {}
    for r in range(rows + 1):
        for c in range(cols):
            if lines[2 * r][3 * c + 1] == "-":
                lay.h_walls.add((r, c))
    for r in range(rows):
        line = lines[2 * r + 1].ljust(3 * cols + 1)
        for c in range(cols + 1):
            if line[3 * c] == "|":
                lay.v_walls.add((r, c))
        for c in range(cols):
            ch = line[3 * c + 1: 3 * c + 3].strip()
            if ch:
                cells[(r, c)] = ch

    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    for (r, c), ch in [(cell, ch) for cell, text in cells.items() for ch in text]:
        if ch == "R":
            lay.start = (lay.cx(c), lay.cy(r), 90.0)
            continue
        if ch not in LANDMARKS:
            continue
        sides = {"N": ((r, c) in lay.h_walls, 90), "S": ((r + 1, c) in lay.h_walls, -90),
                 "W": ((r, c) in lay.v_walls, 180), "E": ((r, c + 1) in lay.v_walls, 0)}
        choice = next((opposite[s] for s, (w, _) in sides.items() if not w and sides[opposite[s]][0]), None)
        choice = choice or next(s for s, (w, _) in sides.items() if w)
        out = math.radians(sides[choice][1])
        off = CELL / 2 - WALL_T / 2 - 0.003
        name, colour = LANDMARKS[ch]
        lay.signs.append(Sign(name, colour, (r, c), lay.cx(c) + off * math.cos(out),
                              lay.cy(r) + off * math.sin(out), sides[choice][1] + 180, wall=choice))
    lay.route_signs = route_signs(lay, ROUTE_SIGNS if route_table is None else route_table)
    lay.towns = {s.name: s.cell for s in lay.signs}
    for s in lay.route_signs:
        lay.towns.setdefault(s.name, s.cell)
    return lay


def route_signs(lay: Layout, table: dict) -> list[RouteSign]:
    out, faces = [], {(s.cell, s.wall) for s in lay.signs}
    for (junction, exit_), town in table.items():
        opens = lay.open_sides(junction)
        if len(opens) < 3 or exit_ not in opens:
            raise ValueError(f"Road sign {town}: {junction} side {exit_} is not an exit of a junction")
        cell = (junction[0] + STEP[exit_][0], junction[1] + STEP[exit_][1])
        if exit_ in lay.open_sides(cell):
            raise ValueError(f"Road sign {town}: the corridor from {junction} going {exit_} must bend or end "
                             f"after one cell, so the sign has a wall facing the junction")
        if (cell, exit_) in faces:
            raise ValueError(f"Road sign {town}: wall {exit_} of {cell} already holds a sign")
        faces.add((cell, exit_))
        x, y = lay.wall_point(cell, exit_)
        out.append(RouteSign(town, cell, exit_, x, y, HEADING_DEG[exit_] + 180, junction, exit_))
    for j in {s.junction for s in out}:
        signed = sum(1 for s in out if s.junction == j)
        if signed != len(lay.open_sides(j)) - 1:
            raise ValueError(f"Junction {j} has {signed} road signs; it needs one per exit except the way in")
    return out

def compass(deg: float) -> str:
    return {0: "east", 90: "north", 180: "west", 270: "south"}[round(deg) % 360]


def sign_table(lay: Layout | None = None) -> str:
    lay = lay or build()
    rows = [("id", "kind", "cell (row,col)", "wall", "faces", "text")]
    for s in sorted(lay.all_signs(), key=lambda s: (s.cell, s.wall)):
        rows.append((s.id, s.kind, f"{s.cell[0]},{s.cell[1]}", s.wall, compass(s.facing_deg), s.plain_text))
    widths = [max(len(r[i]) for r in rows) for i in range(5)]
    lines = ["  ".join(v.ljust(w) for v, w in zip(r[:5], widths)) + "  " + r[5] for r in rows]
    lines.insert(1, "-" * len(lines[0]))
    note = ("Rows are numbered from the top (north), columns from the left (west).\n"
            "landmark = \"you are here\" board. route = green road sign naming the town down that exit of the\n"
            "junction next to it; it faces back toward that junction. Signs carry only the town name.")
    return "\n".join(lines) + "\n\n" + note


def sign_image(name: str, colour) -> Image.Image:
    img = Image.new("RGB", (512, 256), colour)
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 501, 245], outline=(255, 255, 255), width=10)
    font = ImageFont.load_default(size=96 if len(name) <= 6 else 76)
    box = d.textbbox((0, 0), name, font=font)
    d.text(((512 - box[2]) / 2, (256 - box[3]) / 2 - 8), name, font=font, fill=(255, 255, 255))
    return img


def route_sign_image(name: str, aspect: float = SIGN_W / SIGN_H, width: int = 1024) -> Image.Image:
    W, H = width, round(width / aspect)
    border = W // 50
    pad = border * 3
    inner_w, inner_h = W - 2 * pad, H - 2 * pad
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    options = [[name]] + ([name.split(" ", 1)] if " " in name else [])
    best = (0, [name])
    gap = 0.22
    for lines in options:
        for size in range(int(inner_h / (len(lines) * (1 + gap) - gap)), 7, -2):
            font = ImageFont.load_default(size=size)
            if all(probe.textlength(t, font=font) <= inner_w for t in lines):
                if size > best[0]:
                    best = (size, lines)
                break
    size, lines = best
    img = Image.new("RGB", (W, H), ROUTE_GREEN)
    d = ImageDraw.Draw(img)
    d.rectangle([border, border, W - 1 - border, H - 1 - border], outline=(255, 255, 255), width=border)
    font = ImageFont.load_default(size=size)
    y = pad + (inner_h - (len(lines) * size * (1 + gap) - size * gap)) / 2
    for t in lines:
        b = d.textbbox((0, 0), t, font=font)
        d.text(((W - (b[2] - b[0])) / 2 - b[0], y - b[1] + (size - (b[3] - b[1])) / 2), t,
               font=font, fill=(255, 255, 255))
        y += size * (1 + gap)
    return img

def floor_label_image(name: str, width: int = 1024) -> Image.Image:
    W, H = width, round(width * LABEL_H / LABEL_W)
    pad = W // 30
    inner_w, inner_h = W - 2 * pad, H - 2 * pad
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    options = [[name]] + ([name.split(" ", 1)] if " " in name else [])
    line_gap = 0.15
    best_size, best_lines = 0, [name]
    for lines in options:
        for size in range(int(inner_h / (len(lines) * (1 + line_gap) - line_gap)), 7, -2):
            font = ImageFont.load_default(size=size)
            if all(probe.textlength(t, font=font) <= inner_w for t in lines):
                if size > best_size:
                    best_size, best_lines = size, lines
                break
    img = Image.new("RGB", (W, H), LABEL_FLOOR)
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=best_size)
    block_h = len(best_lines) * best_size * (1 + line_gap) - best_size * line_gap
    y = (H - block_h) / 2
    for t in best_lines:
        b = d.textbbox((0, 0), t, font=font)
        d.text(((W - (b[2] - b[0])) / 2 - b[0], y - b[1] + (best_size - (b[3] - b[1])) / 2), t,
               font=font, fill=LABEL_INK)
        y += best_size * (1 + line_gap)
    return img


if __name__ == "__main__":
    print(sign_table())
