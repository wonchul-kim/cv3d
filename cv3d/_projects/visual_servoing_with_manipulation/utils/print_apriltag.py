#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create an AprilTag (Tag36h11) grid as a print-ready PDF (and optional PNGs).
- Requires: opencv-contrib-python, reportlab, numpy
- Prints at exact metric scale (mm). Includes 100 mm scale bar and print check box.
"""

import math
import argparse
from dataclasses import dataclass
import numpy as np

# OpenCV (aruco) / ReportLab
import cv2
from cv2 import aruco
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4, A3
from reportlab.lib.colors import black, white

# ---------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------
@dataclass
class PageSpec:
    width_mm: float
    height_mm: float

PAGE_PRESETS = {
    "A4": PageSpec(width_mm=210.0, height_mm=297.0),
    "A3": PageSpec(width_mm=297.0, height_mm=420.0),
}

def parse_args():
    p = argparse.ArgumentParser(description="Generate AprilTag grid (Tag36h11) as print-accurate PDF.")
    p.add_argument("--rows", type=int, default=4, help="Number of rows")
    p.add_argument("--cols", type=int, default=5, help="Number of cols")
    p.add_argument("--tag-size-mm", type=float, default=10, help="Tag side length (mm), black border included")
    p.add_argument("--spacing", type=float, default=0.25, help="White spacing ratio between tags (e.g., 0.25 = 25%% of tag size)")
    p.add_argument("--margin-mm", type=float, default=10.0, help="Page margin (mm) on all sides")
    p.add_argument("--page", type=str, default="A4", help="A4/A3 or 'WxH' in mm, e.g., 210x297")
    p.add_argument("--start-id", type=int, default=0, help="Starting AprilTag ID (Tag36h11)")
    p.add_argument("--dpi", type=int, default=600, help="Rasterization DPI for tags")
    p.add_argument("--outfile", type=str, default="/HDD/etc/outputs/apriltag_grid.pdf", help="Output PDF filename")
    p.add_argument("--save-individual-png", action="store_true", help="Also save individual tag PNGs")
    args = p.parse_args()
    return args

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def parse_page(page_str: str) -> PageSpec:
    key = page_str.upper()
    if key in PAGE_PRESETS:
        return PAGE_PRESETS[key]
    if "x" in page_str.lower():
        w, h = page_str.lower().split("x")
        return PageSpec(width_mm=float(w), height_mm=float(h))
    raise ValueError(f"Unknown page spec: {page_str}")

def mm2pt(mm_val: float) -> float:
    return mm_val * mm

def check_fit(rows, cols, tag_size_mm, spacing_ratio, margin_mm, page: PageSpec):
    gap_mm = tag_size_mm * spacing_ratio
    grid_w = cols * tag_size_mm + (cols - 1) * gap_mm
    grid_h = rows * tag_size_mm + (rows - 1) * gap_mm
    need_w = grid_w + 2 * margin_mm
    need_h = grid_h + 2 * margin_mm
    if need_w > page.width_mm + 1e-6 or need_h > page.height_mm + 1e-6:
        raise ValueError(
            f"보드({page.width_mm:.1f}×{page.height_mm:.1f}mm) 페이지에 그리드가 맞지 않습니다. "
            f"그리드 크기(태그+간격+여백): {need_w:.1f}×{need_h:.1f}mm, "
            f"태그 {rows}×{cols}, 태그크기 {tag_size_mm:.1f}mm, 간격비 {spacing_ratio}, 여백 {margin_mm:.1f}mm."
        )
    return grid_w, grid_h, gap_mm

def make_tag_image(tag_id: int, tag_px: int, dictionary) -> np.ndarray:
    # drawMarker creates a white image with black marker
    img = aruco.generateImageMarker(dictionary, tag_id, tag_px)
    # ensure 0/255 uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img

def draw_scale_and_info(c: canvas.Canvas, x0_mm, y0_mm, tag_size_mm, rows, cols, start_id, spacing_ratio):
    # 100 mm scale bar
    c.setFillColor(black)
    c.setStrokeColor(black)
    c.setLineWidth(1)
    bar_len_mm = 100.0
    c.rect(mm2pt(x0_mm), mm2pt(y0_mm - 8), mm2pt(bar_len_mm), mm2pt(3), fill=1, stroke=0)
    c.setFont("Helvetica", 8)
    c.drawString(mm2pt(x0_mm), mm2pt(y0_mm - 12), f"100 mm scale bar")

    # Print check box (40 mm)
    c.rect(mm2pt(x0_mm + bar_len_mm + 10), mm2pt(y0_mm - 8), mm2pt(40), mm2pt(40), fill=0, stroke=1)
    c.drawString(mm2pt(x0_mm + bar_len_mm + 10), mm2pt(y0_mm - 12), "Check: 40 mm box")

    # Info text
    info = f"AprilTag Tag36h11 grid: {rows}x{cols}, tag={tag_size_mm:.1f} mm, spacing={spacing_ratio*100:.0f}%, start_id={start_id}"
    c.drawString(mm2pt(x0_mm), mm2pt(y0_mm - 18), info)

def main():
    args = parse_args()
    page = parse_page(args.page)
    rows, cols = args.rows, args.cols
    tag_mm = args.tag_size_mm
    spacing_ratio = args.spacing
    margin_mm = args.margin_mm
    start_id = args.start_id
    dpi = args.dpi
    outfile = args.outfile

    # Select AprilTag dictionary (36h11)
    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
    except Exception as e:
        raise RuntimeError("OpenCV-contrib의 aruco 모듈이 필요합니다. 'pip install opencv-contrib-python'로 설치하세요.") from e

    # Fit check
    grid_w_mm, grid_h_mm, gap_mm = check_fit(rows, cols, tag_mm, spacing_ratio, margin_mm, page)

    # Prepare PDF
    c = canvas.Canvas(outfile, pagesize=(mm2pt(page.width_mm), mm2pt(page.height_mm)))
    c.setFillColor(white)
    c.rect(0, 0, mm2pt(page.width_mm), mm2pt(page.height_mm), fill=1, stroke=0)

    # Top-left origin (margin)
    origin_x_mm = (page.width_mm - grid_w_mm) / 2.0
    origin_y_mm = (page.height_mm + grid_h_mm) / 2.0  # we draw downward in PDF via decreasing y

    # Scale & info near top-left
    draw_scale_and_info(c, x0_mm=margin_mm, y0_mm=page.height_mm - margin_mm, tag_size_mm=tag_mm,
                        rows=rows, cols=cols, start_id=start_id, spacing_ratio=spacing_ratio)

    # Draw grid of tags
    tag_px = int(round(tag_mm / 25.4 * dpi))  # convert mm to inches then to pixels
    for r in range(rows):
        for k in range(cols):
            tag_id = start_id + (r * cols + k)
            img = make_tag_image(tag_id, tag_px, dictionary)

            # Convert to ReportLab image by saving to a numpy buffer (PNG) in memory
            # ReportLab doesn't take numpy directly; write to temp PNG in memory (BytesIO)
            import io
            ret, png_buf = cv2.imencode(".png", img)
            if not ret:
                raise RuntimeError("PNG 인코딩 실패")
            bio = io.BytesIO(png_buf.tobytes())

            # Position (mm)
            x_mm = origin_x_mm + k * (tag_mm + gap_mm)
            y_mm_top = origin_y_mm - r * (tag_mm + gap_mm)

            # Draw white background square for each tag
            c.setFillColor(white)
            c.rect(mm2pt(x_mm), mm2pt(y_mm_top - tag_mm), mm2pt(tag_mm), mm2pt(tag_mm), fill=1, stroke=0)

            # Place PNG (ReportLab coords: lower-left)
            from reportlab.lib.utils import ImageReader
            img_reader = ImageReader(bio)
            c.drawImage(img_reader, mm2pt(x_mm), mm2pt(y_mm_top - tag_mm), width=mm2pt(tag_mm), height=mm2pt(tag_mm), preserveAspectRatio=True, mask='auto')

            # Tag ID label (small)
            c.setFont("Helvetica", 7)
            c.setFillColor(black)
            c.drawString(mm2pt(x_mm), mm2pt(y_mm_top - tag_mm - 4), f"id={tag_id}")

            # Optional: save per-tag PNGs
            if args.save_individual_png:
                cv2.imwrite(f"apriltag_{tag_id:04d}.png", img)

    # Crop marks (optional)
    c.setStrokeColor(black)
    c.setLineWidth(0.5)
    mark = 5.0
    # corners
    c.line(mm2pt(0 + margin_mm), mm2pt(page.height_mm - margin_mm), mm2pt(0 + margin_mm + mark), mm2pt(page.height_mm - margin_mm))
    c.line(mm2pt(0 + margin_mm), mm2pt(page.height_mm - margin_mm), mm2pt(0 + margin_mm), mm2pt(page.height_mm - margin_mm - mark))
    c.line(mm2pt(page.width_mm - margin_mm), mm2pt(page.height_mm - margin_mm), mm2pt(page.width_mm - margin_mm - mark), mm2pt(page.height_mm - margin_mm))
    c.line(mm2pt(page.width_mm - margin_mm), mm2pt(page.height_mm - margin_mm), mm2pt(page.width_mm - margin_mm), mm2pt(page.height_mm - margin_mm - mark))
    c.line(mm2pt(0 + margin_mm), mm2pt(0 + margin_mm), mm2pt(0 + margin_mm + mark), mm2pt(0 + margin_mm))
    c.line(mm2pt(0 + margin_mm), mm2pt(0 + margin_mm), mm2pt(0 + margin_mm), mm2pt(0 + margin_mm + mark))
    c.line(mm2pt(page.width_mm - margin_mm), mm2pt(0 + margin_mm), mm2pt(page.width_mm - margin_mm - mark), mm2pt(0 + margin_mm))
    c.line(mm2pt(page.width_mm - margin_mm), mm2pt(0 + margin_mm), mm2pt(page.width_mm - margin_mm), mm2pt(0 + margin_mm + mark))

    c.showPage()
    c.save()

    print(f"완료: '{outfile}' 생성\n"
          f"- 페이지: {page.width_mm}×{page.height_mm} mm\n"
          f"- 그리드: {rows}×{cols}, 태그 {tag_mm} mm, 간격비 {spacing_ratio}, 여백 {margin_mm} mm\n"
          f"- 시작 ID: {start_id}, DPI: {dpi}\n"
          f"- 인쇄는 반드시 100% (실제 크기)로 하세요.")

if __name__ == "__main__":
    main()
