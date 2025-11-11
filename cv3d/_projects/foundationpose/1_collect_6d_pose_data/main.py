#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
from typing import Union

import numpy as np
import cv2

from apriltag_detector import ATDetector
from realsense import RealSenseSource, OpenCVSource, FrameSource
from utils import * 

def main():
    from pathlib import Path 
    FILE = Path(__file__).resolve()
    ROOT = FILE.parent

    import types
    import yaml
    with open(str(ROOT / 'config.yaml')) as yf:
        config = yaml.load(yf)

    config = types.SimpleNamespace(**config)

    if config.source.lower() == "realsense":
        src: FrameSource = RealSenseSource(width=config.width, height=config.height, fps=config.fps)
    else:
        try:
            cam_idx = int(config.source)
            source_id: Union[int, str] = cam_idx
        except ValueError:
            source_id = config.source

        src = OpenCVSource(
            source_id,
            width=config.width,
            height=config.height,
            fx=config.fx,
            fy=config.fy,
            cx=config.cx,
            cy=config.cy,
            guess_fov_deg=config.guess_fov,
        )

    intr = src.intrinsics()
    K = intr.K

    os.makedirs(config.output_dir, exist_ok=True)
    rgb_dir, depth_dir, pose_dir = ensure_dirs(config.output_dir)
    save_matrix_txt(os.path.join(config.output_dir, 'K.txt'), K)


    detector = ATDetector(families=config.families, nthreads=2, quad_decimate=config.decimate)
    print(f"[INFO] Using detector backend: {detector.impl}")

    writer = None
    gui_enabled = not config.headless
    if config.output_dir:
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if config.output_dir.lower().endswith(".mp4") else "XVID"))
        writer = cv2.VideoWriter(config.output_dir, fourcc, float(config.fps), (config.width, config.height))

    last = time.time()
    frame_count = 1
    fps_smoothed = 0.0

    while True:
        color, depth_mm = src.read()
        if color is None:
            print("[INFO] End of stream or frame not available.")
            break

        # Ensure size matches the assumed K
        h, w = color.shape[:2]
        # (Optional) If incoming size differs, you could scale K accordingly.

        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        detections = detector.detect(gray, K, tag_size=config.tag_size)

        overlay = color.copy()
        for det_idx, det in enumerate(detections):
            corners = det["corners"]
            center = det["center"]
            R = det["R"]
            t = det["t"]
            tag_id = det["tag_id"]
            
            ob_in_cam = to_SE3(R, t)
            cam_in_ob = np.linalg.inv(ob_in_cam)
            
            stem = format_idx(frame_count)
            rgb_path  = os.path.join(rgb_dir,  f"{stem}.png")
            pose_path = os.path.join(pose_dir, f"{stem}.txt")
            cv2.imwrite(rgb_path, color)
            save_matrix_txt(pose_path, cam_in_ob)

            if depth_mm is not None:
                depth_path = os.path.join(depth_dir, f"{stem}.png")
                # depth_mm is uint16 millimeters (0 == invalid). Keep as 16-bit PNG.
                cv2.imwrite(depth_path, depth_mm)

            draw_tag_outline(overlay, corners, (0, 255, 255), 2)
            draw_cube(overlay, K, R, t, tag_size=config.tag_size)
            draw_axes(overlay, K, R, t, axis_len=config.tag_size * 0.5)

            x, y, z = format_xyz_m(t)
            # Put per-tag text near the tag
            txt = [f"ID {tag_id}  x:{x:+.3f} m  y:{y:+.3f} m  z:{z:+.3f} m"]
            put_tag_text(overlay, (10, 10 + 22), txt, scale=0.4, thickness=1)

        # Global HUD (top-left)
        now = time.time()
        dt = now - last
        frame_count += 1
        if dt >= 0.5:
            fps_inst = frame_count / dt
            fps_smoothed = 0.9 * fps_smoothed + 0.1 * fps_inst if fps_smoothed > 0 else fps_inst
            last = now
            frame_count = 0

        if config.show_cam_info:
            hud_lines = [
                f"FX:{intr.fx:.1f}  FY:{intr.fy:.1f}  CX:{intr.cx:.1f}  CY:{intr.cy:.1f}",
                f"family:{config.families}  tag:{config.tag_size:.3f} m  det:{len(detections)}  FPS:{fps_smoothed:.1f}",
                "Coords in CAMERA frame (x right, y down, z forward)",
            ]
            put_tag_text(overlay, (10, 10 + 22 * len(hud_lines)), hud_lines, scale=0.6, thickness=2)
        draw_principal_point(overlay, K, size=12)

        if writer is not None:
            writer.write(overlay)

        if gui_enabled:
            try:
                cv2.imshow("AprilTag Pose Demo", overlay)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
                if key == ord("s"):
                    ts = int(time.time())
                    snap_path = f"snap_{ts}.png"
                    cv2.imwrite(snap_path, overlay)
                    print(f"[INFO] Saved snapshot -> {snap_path}")
            except cv2.error as e:
                # GUI backend not available; switch to headless on the fly
                print(f"[WARN] HighGUI unavailable ({e}). Switching to --headless mode.")
                gui_enabled = False

    if writer is not None:
        writer.release()
    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
