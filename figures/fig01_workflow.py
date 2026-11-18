"""fig01: Reconstruction workflow."""

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from matplotlib.font_manager import FontProperties, findfont
from PIL import Image, ImageFont

from common import figure_args, set_plot_style


SVG_NS = "http://www.w3.org/2000/svg"


WIDTH, HEIGHT, DPI = 3200, 1680, 300


def create_workflow() -> ET.Element:
    ET.register_namespace("", SVG_NS)
    root = ET.Element(f"{{{SVG_NS}}}svg", width=str(WIDTH), height=str(HEIGHT), viewBox="0 0 1600 840")

    def el(tag, parent=root, **attrs):
        attributes = {key.rstrip("_").replace("_", "-"): str(value) for key, value in attrs.items()}
        return ET.SubElement(parent, f"{{{SVG_NS}}}{tag}", attributes)

    el("title").text = "Workflow for annual winter and summer mass balance reconstruction"
    el("desc").text = (
        "Inputs are matched by RGI identifiers. Seasonal windows use reported windows, glacier history, "
        "nearby glaciers or a regional ensemble in priority order. Three predictor matrices feed separate "
        "XGBoost models and observation-first reconstructions. Dashed branches indicate validation."
    )
    colors = {
        "input": ("#EDF3F9", "#4E7799"),
        "prep": ("#EEEAF6", "#78649B"),
        "matrix": ("#EAF4F4", "#3B8586"),
        "model": ("#E9EFFA", "#506FA4"),
        "validation": ("#FCF2E2", "#A47735"),
        "product": ("#ECF4E7", "#678650"),
    }
    defs = el("defs")
    for key, (_, edge) in colors.items():
        marker = el("marker", defs, id=f"arrow-{key}", viewBox="0 0 10 10", refX=9, refY=5,
                    markerWidth=7, markerHeight=7, orient="auto-start-reverse", markerUnits="userSpaceOnUse")
        el("path", marker, d="M 0 1 L 9 5 L 0 9 Z", fill=edge)
    el("rect", x=0, y=0, width=1600, height=840, fill="white")
    bg = el("g", id="stage-backgrounds")
    lines = el("g", id="connectors", fill="none", stroke_linejoin="round")
    nodes = el("g", id="nodes", font_family="Arial, Helvetica, sans-serif", fill="#202124")
    font_files = {
        bold: findfont(FontProperties(family="Arial", weight=weight), fallback_to_default=False)
        for bold, weight in ((False, "normal"), (True, "bold"))
    }

    def text(x, y, value, size=20, bold=False):
        label = el("text", nodes, x=x, y=y, font_size=size,
                   font_weight="700" if bold else "400", text_anchor="middle")
        label.text = value

    def route(points, role="prep", dashed=False, arrow=True, width=1.8):
        attrs = {
            "points": " ".join(f"{x},{y}" for x, y in points),
            "stroke": colors[role][1],
            "stroke_width": width,
        }
        if dashed:
            attrs["stroke-dasharray"] = "5 4"
        if arrow:
            attrs["marker-end"] = f"url(#arrow-{role})"
        el("polyline", lines, **attrs)

    def box(x, y, width, height, title, body, role, title_size=20, body_size=17, dashed=False):
        fill, edge = colors[role]
        attrs = dict(x=x, y=y, width=width, height=height, rx=10, fill=fill, stroke=edge, stroke_width=1.35)
        if dashed:
            attrs["stroke-dasharray"] = "5 4"
        el("rect", nodes, **attrs)

        # Center the full visible text block using actual Arial glyph bounds.
        entries = [(title, title_size, True)] + [(value, body_size, False) for value in body]
        measured = []
        for value, size, bold in entries:
            font = ImageFont.truetype(font_files[bold], round(size * 10))
            _, top, _, bottom = font.getbbox(value, anchor="ls")
            measured.append((value, size, bold, top / 10, (bottom - top) / 10))
        gaps = [4 if index or height < 60 else 7 for index in range(len(measured) - 1)]
        cursor = y + (height - sum(line[4] for line in measured) - sum(gaps)) / 2
        for (value, size, bold, top, glyph_height), gap in zip(measured, [*gaps, 0], strict=True):
            text(x + width / 2, cursor - top, value, size, bold)
            cursor += glyph_height + gap

    stages = [
        (10, "Input data", "input"),
        (330, "Preparation", "prep"),
        (650, "Target matrices", "matrix"),
        (970, "Models and validation", "model"),
        (1290, "Reconstruction and outputs", "product"),
    ]
    for x, label, role in stages:
        fill, _ = colors[role]
        el("rect", bg, x=x, y=82, width=290, height=708, rx=12, fill=fill, opacity=".34")
        text(x + 145, 42, label, 21, True)
        el("line", bg, x1=x + 10, y1=66, x2=x + 280, y2=66, stroke="#B8BCC0", stroke_width=1)

    # Six inputs retain all record counts and predictor counts from the manuscript.
    input_specs = [
        ("WGMS annual balance", "5,465 records"),
        ("WGMS winter balance", "2,519 records"),
        ("WGMS summer balance", "2,488 records"),
        ("WGMS seasonal dates", "reported start and end dates"),
        ("RGI 6.0", "identifier and 9 attributes"),
        ("ERA5-Land", "14 monthly variables"),
    ]
    for i, (title, body) in enumerate(input_specs):
        box(20, 100 + 115 * i, 270, 100, title, [body], "input")

    box(340, 100, 270, 95, "WGMS–RGI matching", ["common RGI identifiers"], "prep")
    box(340, 210, 270, 75, "Annual climate", ["12 calendar months"], "prep")
    # The surrounding panel makes the priority-based window assignment one explicit step.
    el("rect", nodes, x=337, y=300, width=276, height=292, rx=11,
       fill="#E2DCF0", stroke="#78649B", stroke_width=1.4)
    text(475, 325, "Seasonal window assignment", 18.5, True)
    text(475, 346, "Use the first available source", 15.5)
    steps = [
        ("1  Reported window", "retain when available"),
        ("2  Glacier history", "most frequent; ≥3 reported windows"),
        ("3  Nearby glaciers", "same region; ≤5; weighted vote ≥50%"),
        ("4  Regional ensemble", "up to 3 most frequent windows"),
    ]
    for i, (title, body) in enumerate(steps):
        box(348, 358 + 56 * i, 254, 48, title, [body], "prep", title_size=17.5, body_size=14.7)
    box(340, 609, 270, 76, "Seasonal climate", ["aggregate monthly variables", "over the assigned window"], "prep")
    box(340, 695, 270, 90, "Window assignment validation",
        ["single-window withheld", "glacier-history withheld", "recovery and prediction sensitivity"],
        "validation", title_size=17.2, body_size=15.4, dashed=True)

    centers = [147.5, 365, 582.5]
    for target, cy in zip(("Annual", "Winter", "Summer"), centers, strict=True):
        if target == "Annual":
            predictors = ["9 glacier attributes", "14 variables × 12 months", "177 predictors"]
            reconstruction = ["retain annual observations", "else: winter + summer", "else: annual-model fallback"]
        else:
            predictors = ["9 glacier attributes", "14 seasonal variables", "23 predictors"]
            reconstruction = ["retain observed balances", "else: model predictions", "window source retained"]
        box(660, cy - 47.5, 270, 95, f"{target} predictor matrix", predictors, "matrix", title_size=19.7)
        box(980, cy - 47.5, 270, 95, f"{target} model", ["XGBoost"], "model")
        box(1300, cy - 47.5, 270, 95, f"{target} reconstruction", reconstruction, "product")
    box(980, 695, 270, 90, "Model evaluation", ["random 10-fold CV", "LOGO · LOYO"], "validation", dashed=True)
    box(1300, 695, 270, 90, "Mass-balance series", ["1979–2025"], "product")

    # Inputs use separate gutters, so vertical segments stay clear of the boxes.
    for cy in (150, 265, 380, 610):
        route([(290, cy), (306, cy)], "input", arrow=False, width=1.35)
    route([(306,147.5),(306,610)],'input',arrow=False,width=1.35)
    route([(306,147.5),(340,147.5)],'input')
    route([(290,495),(318,495),(318,340),(337,340)],'prep')
    route([(290,725),(328,725),(328,247.5)],'prep',arrow=False,width=1.4)
    route([(328,247.5),(340,247.5)],'prep')
    route([(328,647),(340,647)],'prep')
    route([(475,592),(475,609)],'prep')
    route([(475,685),(475,695)],'validation',dashed=True)

    # Shared attributes and climate data enter through separate, well-spaced buses.
    route([(610,147.5),(624,147.5),(624,582.5)],'input',arrow=False,width=1.4)
    for cy in centers:
        route([(624,cy),(660,cy)],'input')
    route([(610,247.5),(642,247.5),(642,170),(660,170)],'prep')
    route([(610,647),(646,647),(646,390)],'prep',arrow=False,width=1.4)
    route([(646,390),(660,390)],'prep')
    route([(646,607),(660,607)],'prep')

    # Matrices, models and reconstructions share the same centers: direct arrows.
    for cy in centers:
        route([(930,cy),(980,cy)],'model',width=2)
        route([(1250,cy),(1300,cy)],'product',width=2)
        route([(1250,cy+28),(1275,cy+28)],'validation',dashed=True,arrow=False,width=1.5)
    route([(1275,175.5),(1275,740)],'validation',dashed=True,arrow=False,width=1.5)
    route([(1275,740),(1250,740)],'validation',dashed=True)
    for cy in centers:
        route([(1570,cy),(1588,cy)],'product',arrow=False,width=1.5)
    route([(1588,147.5),(1588,740)],'product',arrow=False,width=1.5)
    route([(1588,740),(1570,740)],'product')

    return root


def draw_workflow(_root: Path, output: Path) -> None:
    svg = output / "figure01.svg"
    png = output / "figure01.png"
    diagram = ET.ElementTree(create_workflow())
    ET.indent(diagram)
    diagram.write(svg, encoding="utf-8", xml_declaration=True)

    renderer = Path(__file__).with_name("render_workflow.cjs")
    subprocess.run(["node", str(renderer), str(svg), str(png), str(DPI)], check=True)
    with Image.open(png) as image:
        image.save(output / "figure01.pdf", resolution=DPI, quality=100, subsampling=0)


if __name__ == "__main__":
    root, output, dpi = figure_args(__doc__)
    set_plot_style(dpi=dpi)
    draw_workflow(root, output)
