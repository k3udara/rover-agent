"""Export every maze sign as a 300 dpi PNG at real size, plus a placement sheet, into print/.

  python scripts/export_print_signs.py [--out DIR]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFont

from agent.maze import CELL, ROUTE_GREEN, build, compass

DPI = 300


def px(metres: float) -> int:
    return round(metres / 0.0254 * DPI)


def export_signs(lay, out: Path) -> list[tuple[int, object, Path]]:
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.png"):
        old.unlink()
    done = []
    signs = sorted(lay.all_signs(), key=lambda s: (s.kind != "landmark", s.cell, s.wall))
    for n, s in enumerate(signs, 1):
        img = s.image().resize((px(s.w), px(s.h)), Image.LANCZOS)
        name = f"{n:02d}_{s.kind}_{s.id}.png"
        img.save(out / name, dpi=(DPI, DPI))
        done.append((n, s, out / name))
    return done


def placement_sheet(lay, numbered, out: Path) -> Path:
    W, H = 2480, 3508
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    title = ImageFont.load_default(size=64)
    font = ImageFont.load_default(size=34)
    small = ImageFont.load_default(size=28)
    d.text((120, 100), "Rover maze: sign placement (top view, north up)", font=title, fill="black")
    d.text((120, 190), f"Cell = {CELL * 100:.0f} cm. Each number sits on the wall the sign is glued to; "
                       "the tick points the way the sign faces.", font=small, fill=(60, 60, 60))

    span = max(lay.rows, lay.cols) * CELL
    size = 1700
    ox, oy = (W - size) // 2, 290
    P = lambda x, y: (ox + (x + span / 2) / span * size, oy + (span / 2 - y) / span * size)
    for x0, y0, x1, y1 in lay.wall_segments():
        d.line([P(x0, y0), P(x1, y1)], fill="black", width=10)
    for r in range(lay.rows):
        for c in range(lay.cols):
            cx, cy = P(*lay.centre((r, c)))
            d.text((cx - 40, cy - 14), f"r{r} c{c}", font=small, fill=(170, 170, 170))
    sx, sy, _ = lay.start
    rx, ry = P(sx, sy)
    d.text((rx - 20, ry + 20), "START", font=small, fill=(52, 168, 83))
    for n, s, _ in numbered:
        colour = s.colour if s.kind == "landmark" else ROUTE_GREEN
        a = math.radians(s.facing_deg)
        d.line([P(s.x, s.y), P(s.x + 0.045 * math.cos(a), s.y + 0.045 * math.sin(a))], fill=colour, width=8)
        x, y = P(s.x + 0.022 * math.cos(a), s.y + 0.022 * math.sin(a))
        d.ellipse([x - 26, y - 26, x + 26, y + 26], fill=colour)
        b = d.textbbox((0, 0), str(n), font=small)
        d.text((x - (b[2] - b[0]) / 2, y - 18), str(n), font=small, fill="white")

    y = oy + size + 70
    d.text((120, y), "#   kind        cell    wall  faces   text", font=font, fill="black")
    y += 50
    for n, s, path in numbered:
        line = f"{n:<3d} {s.kind:<10s}  r{s.cell[0]} c{s.cell[1]}   {s.wall:<4s}  {compass(s.facing_deg):<6s}  {s.plain_text}"
        d.text((120, y), line, font=small, fill="black")
        y += 40
    d.text((120, y + 20), "landmark = coloured \"you are here\" board. route = green road sign: it names the town "
                          "down that road and faces back toward the junction.", font=small, fill=(60, 60, 60))
    png = out / "placement_sheet.png"
    page.save(png, dpi=(DPI, DPI))
    page.save(out / "placement_sheet.pdf", resolution=DPI)
    return png


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "print"))
    a = p.parse_args(argv)
    out = Path(a.out)
    lay = build()
    numbered = export_signs(lay, out / "signs")
    sheet = placement_sheet(lay, numbered, out)
    print(f"{len(numbered)} signs -> {out / 'signs'} (300 dpi, real size)")
    print(f"placement sheet -> {sheet} and {sheet.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
