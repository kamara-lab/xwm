"""Build the introductory SVGs: python tools/build_world_model_diagram.py.

Editable vector schematic inspired by xevals' method diagram, using xwm's
banner blues and amber. Requires fonttools and TeX Live's Latin Modern fonts.
Labels are outlined so GitHub and browsers display the same typography without
requiring visitors to install fonts. Label text remains in accessible SVG groups.
"""

import subprocess
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

ASSETS = Path(__file__).resolve().parents[1] / "docs" / "assets"


@lru_cache(maxsize=2)
def label_font(bold):
    name = "lmsans10-bold.otf" if bold else "lmsans10-regular.otf"
    path = subprocess.check_output(["kpsewhich", name], text=True).strip()
    return TTFont(path)


def build(theme):
    dark = theme == "dark"
    bg, ink, muted, line, blue, amber, blue_soft, amber_soft = (
        ("#17232d", "#edf1f3", "#afbac2", "#64737e", "#7baed3", "#d6a465", "#263e50", "#40372e")
        if dark
        else (
            "#fafaf8",
            "#18232c",
            "#5c6871",
            "#a8b2b8",
            "#326e9a",
            "#a56e32",
            "#e8f0f6",
            "#f4ecdf",
        )
    )
    parts = [
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="360"
  viewBox="0 0 1120 360" role="img" aria-labelledby="title desc">
  <title id="title">xwm: from observation to action</title>
  <desc id="desc">An encoder turns observations into a compact internal state.
  A dynamics model predicts future states for candidate actions. A planner uses
  these predictions and a goal or reward to choose an action. The action changes
  the environment, producing the next observation. Predictions are internal
  representations, not generated images. Optional reward and value heads are omitted.</desc>
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
      markerWidth="5" markerHeight="5" orient="auto-start-reverse">
      <path d="M 1 1 L 9 5 L 1 9 L 3 5 Z" fill="{muted}"/>
    </marker>
  </defs>
  <style>
    .wire {{ fill: none; stroke-linecap: round; stroke-linejoin: round; }}
  </style>'''
    ]

    def text(x, y, label, size=20, color=ink, weight=400):
        font = label_font(weight >= 600)
        glyphs = font.getGlyphSet()
        cmap = font.getBestCmap()
        names = [cmap[ord(char)] for char in label]
        scale = size / font["head"].unitsPerEm
        advance = sum(font["hmtx"][name][0] for name in names)
        left = x - advance * scale / 2
        pen = SVGPathPen(glyphs)
        for name in names:
            glyphs[name].draw(TransformPen(pen, (scale, 0, 0, -scale, left, y)))
            left += font["hmtx"][name][0] * scale
        parts.append(
            f'<g role="img" aria-label="{escape(label)}"'
            f' data-font="Latin Modern Sans" fill="{color}">'
            f'<path d="{pen.getCommands()}"/></g>'
        )

    def rect(x, y, w, h, fill="none", stroke=line, radius=5, dashed=False):
        dash = ' stroke-dasharray="4 4"' if dashed else ""
        parts.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}"'
            f' rx="{radius}" fill="{fill}" stroke="{stroke}"'
            f' stroke-width="1.5"{dash}/>'
        )

    def path(d, color=muted, width=1.6, arrow=False, dashed=False):
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        dash = ' stroke-dasharray="5 5"' if dashed else ""
        parts.append(
            f'<path d="{d}" class="wire" stroke="{color}" stroke-width="{width}"{marker}{dash}/>'
        )

    def circle(x, y, r, fill=bg, stroke=ink, width=1.6):
        parts.append(
            f'<circle cx="{x}" cy="{y}" r="{r}"'
            f' fill="{fill}" stroke="{stroke}" stroke-width="{width}"/>'
        )

    # Shared baseline and consistent node sizes; only the learned core is grouped.
    rect(198, 103, 548, 190)
    text(472, 131, "World model", 18, ink, 600)
    for x, w, label in (
        (20, 142, "Observation"),
        (220, 132, "Encoder"),
        (410, 314, "Dynamics"),
        (790, 142, "Planner"),
        (972, 128, "Action"),
    ):
        rect(
            x,
            154,
            w,
            116,
            blue_soft if label == "Dynamics" else "none",
            blue if label == "Dynamics" else line,
        )
        text(x + w / 2, 250, label)
    for start, end in ((162, 220), (352, 410), (724, 790), (932, 972)):
        path(f"M {start} 211 H {end - 3}", arrow=True)

    # Candidate actions enter the dynamics; predictions return on the main line.
    path("M 861 154 V 62 H 567 V 151", arrow=True)
    text(714, 46, "Candidate actions", 17, muted)

    # Robot arm: simple joints, a gripper, and a blue object.
    path("M 51 223 H 128", line)
    rect(57, 216, 20, 7, bg, ink, 1)
    path("M 67 215 L 60 192 L 82 171 L 105 184", line, 7)
    path("M 67 215 L 60 192 L 82 171 L 105 184", ink, 1.6)
    for x, y in ((67, 215), (60, 192), (82, 171), (105, 184)):
        circle(x, y, 4)
    path("M 105 189 V 196 M 99 201 V 196 H 111 V 201", ink)
    rect(101, 207, 13, 15, blue_soft, blue, 1)
    path("M 101 207 l 5 -4 h 13 v 15 l -5 4 M 114 207 l 5 -4", blue)

    # Encoder: observations become a smaller set of useful features.
    for y1 in (175, 195, 215):
        for y2 in (181, 208):
            path(f"M 257 {y1} L 285 {y2}", line, 1)
    for y1 in (181, 208):
        for y2 in (185, 205):
            path(f"M 285 {y1} L 313 {y2}", line, 1)
    for x, ys in ((257, (175, 195, 215)), (285, (181, 208)), (313, (185, 205))):
        for y in ys:
            circle(x, y, 4.2, blue_soft if x == 313 else bg, blue if x == 313 else ink)

    # A rollout in feature space; tiled states deliberately avoid predicted images.
    for step, x in enumerate((437, 535, 633)):
        rect(x, 173, 62, 49, bg, blue, 4, dashed=step == 2)
        for i in range(6):
            active = (i + step) % 3 != 1
            rect(
                x + 10 + (i % 3) * 15,
                183 + (i // 3) * 15,
                11,
                11,
                blue if active else blue_soft,
                "none",
                2,
            )
        if step < 2:
            path(f"M {x + 66} 197 H {x + 93}", arrow=True)

    # Planner: score possible paths; emphasize the selected one with amber.
    path("M 815 221 H 908", line, 1)
    path("M 823 216 L 845 196 L 870 206 L 900 179", line, 1.5)
    path("M 823 216 L 845 205 L 870 183 L 900 192", blue, 1.8)
    path("M 823 216 L 845 193 L 870 185 L 900 173", amber, 2.2)
    for x, y in ((823, 216), (845, 193), (870, 185), (900, 173)):
        circle(x, y, 2.8, bg, amber, 1.5)

    # Action: a single chosen movement, illustrated by a displaced object.
    rect(994, 193, 20, 20, bg, line, 2, dashed=True)
    path("M 1021 203 H 1040", amber, 2)
    path("M 1035 199 l 5 4 l -5 4", amber, 2)
    rect(1049, 193, 20, 20, amber_soft, amber, 2)
    path("M 993 222 H 1073", line, 1)

    # Apply the chosen move in the environment, then encode the new observation.
    path("M 1036 270 V 328 H 611")
    path("M 509 328 H 91 V 273", arrow=True)
    text(560, 334, "Next step", 16, muted)
    parts.append("</svg>\n")
    (ASSETS / f"world-model-{theme}.svg").write_text("\n".join(parts))


if __name__ == "__main__":
    for theme in ("light", "dark"):
        build(theme)
