#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a ChArUco board (PDF + SVG) with precise metric sizing.
- Requires: opencv-contrib-python, reportlab, svgwrite, numpy
"""

import io, base64, argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import cv2
from cv2 import aruco
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.colors import black, white
import svgwrite

# ---------------------- Args ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="ChArUco board generator (PDF + SVG)")
    p.add_argument("--squares-x", type=int, default=6, help="Squares along width (columns)")
    p.add_argument("--squares-y", type=int, default=8, help="Squares along height (rows)")
    p.add_argument("--square-mm", type=float, default=23, help="Chess square size in mm")
    p.add_argument("--marker-mm", type=float, default=12, help="Marker side in mm (black border included)")
    p.add_argument("--page", type=str, default="A4", help="A4/A3 or 'WxH' in mm, e.g., 210x297")
    p.add_argument("--margin-mm", type=float, default=10.0, help="Margin on all sides in mm")
    p.add_argument("--dict", type=str, default="DICT_4X4_1000", help="ArUco dictionary name")
    p.add_argument("--start-id", type=int, default=0, help="First marker id")
    p.add_argument("--dpi", type=int, default=600, help="Marker raster DPI")
    p.add_argument("--out-base", type=str, default="/HDD/etc/outputs/charuco_board", help="Output base filename (no extension)")
    return p.parse_args()

# ---------------------- Helpers ----------------------
@dataclass
class PageSpec:
    width_mm: float
    height_mm: float

PRESETS = {
    "A4": PageSpec(210.0, 297.0),
    "A3": PageSpec(297.0, 420.0),
}

def parse_page(spec: str) -> PageSpec:
    key = spec.upper()
    if key in PRESETS: return PRESETS[key]
    if "x" in spec.lower():
        w, h = spec.lower().split("x")
        return PageSpec(float(w), float(h))
    raise ValueError(f"Unknown page spec: {spec}")

def mm2pt(v): return v * mm
def px_for_mm(mm_val: float, dpi: int) -> int:
    return int(round(mm_val / 25.4 * dpi))

def aruco_dict_by_name(name: str):
    name = name.upper()
    try:
        attr = getattr(aruco, name)
        return aruco.getPredefinedDictionary(attr)
    except Exception as e:
        raise RuntimeError(f"Unknown/unsupported dictionary: {name}") from e

def check_fit(sx, sy, sq_mm, margin_mm, page: PageSpec):
    board_w = sx * sq_mm
    board_h = sy * sq_mm
    need_w = board_w + 2*margin_mm
    need_h = board_h + 2*margin_mm
    if need_w > page.width_mm + 1e-6 or need_h > page.height_mm + 1e-6:
        raise ValueError(
            f"보드({board_w:.1f}×{board_h:.1f}mm) + 여백({margin_mm:.1f}mm)가 "
            f"페이지({page.width_mm:.1f}×{page.height_mm:.1f}mm)를 초과합니다. "
            f"필요 크기: {need_w:.1f}×{need_h:.1f}mm"
        )
    return board_w, board_h

def white_square_positions(sx, sy):
    # Top-left is black -> white if (r+c)%2==1
    pos = []
    for r in range(sy):
        for c in range(sx):
            if (r + c) % 2 == 1:
                pos.append((r, c))
    return pos

# ---------------------- Main ----------------------
def main():
    a = parse_args()
    page = parse_page(a.page)
    dict_ = aruco_dict_by_name(a.dict)
    board_w_mm, board_h_mm = check_fit(a.squares_x, a.squares_y, a.square_mm, a.margin_mm, page)

    # -------- PDF --------
    pdf_path = Path(f"{a.out_base}.pdf")
    c = canvas.Canvas(str(pdf_path), pagesize=(mm2pt(page.width_mm), mm2pt(page.height_mm)))
    c.setFillColor(white)
    c.rect(0, 0, mm2pt(page.width_mm), mm2pt(page.height_mm), fill=1, stroke=0)

    origin_x_mm = (page.width_mm - board_w_mm) / 2.0
    origin_y_mm = (page.height_mm + board_h_mm) / 2.0  # top edge in PDF coords

    # Draw chessboard
    for r in range(a.squares_y):
        for x in range(a.squares_x):
            x_mm = origin_x_mm + x * a.square_mm
            y_top_mm = origin_y_mm - r * a.square_mm
            is_black = ((r + x) % 2 == 0)
            if is_black:
                c.setFillColor(black)
                c.rect(mm2pt(x_mm), mm2pt(y_top_mm - a.square_mm), mm2pt(a.square_mm), mm2pt(a.square_mm), fill=1, stroke=0)

    # Draw markers (centered on white squares)
    tag_px = px_for_mm(a.marker_mm, a.dpi)
    tag_id = a.start_id
    for (r, x) in white_square_positions(a.squares_x, a.squares_y):
        img = aruco.generateImageMarker(dict_, tag_id, tag_px)
        ok, png_buf = cv2.imencode(".png", img)
        if not ok: raise RuntimeError("PNG encoding failed")
        from reportlab.lib.utils import ImageReader
        bio = io.BytesIO(png_buf.tobytes())

        x_mm = origin_x_mm + x * a.square_mm + (a.square_mm - a.marker_mm)/2.0
        y_top_mm = origin_y_mm - r * a.square_mm
        c.drawImage(ImageReader(bio), mm2pt(x_mm), mm2pt(y_top_mm - a.marker_mm),
                    width=mm2pt(a.marker_mm), height=mm2pt(a.marker_mm),
                    preserveAspectRatio=True, mask='auto')

        c.setFillColor(black); c.setFont("Helvetica", 7)
        c.drawString(mm2pt(x_mm), mm2pt(y_top_mm - a.marker_mm - 4), f"id={tag_id}")
        tag_id += 1

    # Scale bar + info
    c.setFillColor(black); c.setStrokeColor(black); c.setLineWidth(1)
    bar_len = 100.0
    c.rect(mm2pt(a.margin_mm), mm2pt(page.height_mm - a.margin_mm - 3), mm2pt(bar_len), mm2pt(3), fill=1, stroke=0)
    c.setFont("Helvetica", 8)
    c.drawString(mm2pt(a.margin_mm), mm2pt(page.height_mm - a.margin_mm - 8),
        f"100 mm scale bar | ChArUco {a.squares_x}x{a.squares_y}, square={a.square_mm:.1f} mm, marker={a.marker_mm:.1f} mm, dict={a.dict}, start_id={a.start_id}")

    c.showPage(); c.save()

    # -------- SVG --------
    svg_path = Path(f"{a.out_base}.svg")
    dwg = svgwrite.Drawing(str(svg_path), size=(f"{page.width_mm}mm", f"{page.height_mm}mm"))
    dwg.add(dwg.rect(insert=("0mm","0mm"), size=(f"{page.width_mm}mm", f"{page.height_mm}mm"), fill="white"))

    origin_x = (page.width_mm - board_w_mm) / 2.0
    origin_y = (page.height_mm - board_h_mm) / 2.0  # SVG y increases downward

    # board
    for r in range(a.squares_y):
        for x in range(a.squares_x):
            xx = origin_x + x * a.square_mm
            yy = origin_y + r * a.square_mm
            if ((r + x) % 2 == 0):
                dwg.add(dwg.rect(insert=(f"{xx}mm", f"{yy}mm"),
                                 size=(f"{a.square_mm}mm", f"{a.square_mm}mm"),
                                 fill="black"))

    # markers: embed PNGs as data URIs
    tag_id = a.start_id
    for (r, x) in white_square_positions(a.squares_x, a.squares_y):
        img = aruco.generateImageMarker(dict_, tag_id, tag_px)
        ok, png_buf = cv2.imencode(".png", img)
        if not ok: raise RuntimeError("PNG encoding failed")
        b64 = base64.b64encode(png_buf).decode("ascii")
        href = f"data:image/png;base64,{b64}"

        xx = origin_x + x * a.square_mm + (a.square_mm - a.marker_mm)/2.0
        yy = origin_y + r * a.square_mm + (a.square_mm - a.marker_mm)/2.0
        dwg.add(dwg.image(href=href, insert=(f"{xx}mm", f"{yy}mm"),
                          size=(f"{a.marker_mm}mm", f"{a.marker_mm}mm")))
        dwg.add(dwg.text(f"id={tag_id}", insert=(f"{xx}mm", f"{yy + a.marker_mm + 2}mm"),
                         font_size="2.5mm", fill="black"))
        tag_id += 1

    # scale bar
    dwg.add(dwg.rect(insert=(f"{a.margin_mm}mm", f"{page.height_mm - a.margin_mm - 3}mm"),
                     size=("100mm", "3mm"), fill="black"))
    dwg.add(dwg.text("100 mm scale bar",
                     insert=(f"{a.margin_mm}mm", f"{page.height_mm - a.margin_mm - 5}mm"),
                     font_size="3mm", fill="black"))

    dwg.save()

    print(f"Done:\n - {pdf_path}\n - {svg_path}\n"
          f"Board size: {board_w_mm:.1f} × {board_h_mm:.1f} mm, on page {page.width_mm:.1f} × {page.height_mm:.1f} mm")

if __name__ == "__main__":
    main()
