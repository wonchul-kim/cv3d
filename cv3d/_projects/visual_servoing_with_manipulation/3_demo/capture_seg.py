#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, time, math, argparse
from collections import deque
from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
import threading
import rclpy

# 캡처·동기화
IMG_W, IMG_H = 640, 480
DT_TOL_SEC    = 0.040   # 카메라-로봇 시간 동기 허용오차

WIN_NAME = "handeye-capture"

# -------------------- RealSenseWorker --------------------
try:
    import pyrealsense2 as rs
except Exception as e:
    print(f"[ERROR] While importing pyrealsense2, {repr(e)}")
    print("Attempting to install pyrealsense2...")
    import sys, subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "pyrealsense2"])
    import pyrealsense2 as rs

class RealSenseWorker:
    def __init__(self, width: int = IMG_W, height: int = IMG_H, fps: int = 30,
                 enable_emitter: Optional[int] = 1,
                 laser_power_ratio: Optional[float] = None,
                 manual_exposure_us: Optional[int] = None) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.enable_emitter = enable_emitter
        self.laser_power_ratio = laser_power_ratio
        self.manual_exposure_us = manual_exposure_us

        self._pipe: Optional[rs.pipeline] = None
        self._align: Optional[rs.align] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._ts_ms: Optional[float] = None
        self._color_bgr: Optional[np.ndarray] = None
        self._depth_mm: Optional[np.ndarray] = None
        self._intrinsics: Optional[Dict[str, Any]] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        try:
            if self._pipe is not None:
                self._pipe.stop()
        finally:
            self._pipe = None

    def _run(self) -> None:
        self._pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = self._pipe.start(cfg)
        self._align = rs.align(rs.stream.color)

        # Optional sensor config
        try:
            dev = profile.get_device()
            for s in dev.sensors:
                if self.enable_emitter is not None and s.supports(rs.option.emitter_enabled):
                    s.set_option(rs.option.emitter_enabled, float(self.enable_emitter))
                if self.laser_power_ratio is not None and s.supports(rs.option.laser_power):
                    rng = s.get_option_range(rs.option.laser_power)
                    val = rng.min + (rng.max - rng.min) * float(self.laser_power_ratio)
                    s.set_option(rs.option.laser_power, float(val))
                if self.manual_exposure_us is not None and s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, 0.0)
                    if s.supports(rs.option.exposure):
                        s.set_option(rs.option.exposure, float(self.manual_exposure_us))
        except Exception:
            pass

        # Color intrinsics
        try:
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            i = color_stream.get_intrinsics()
            self._intrinsics = {
                "width": i.width,
                "height": i.height,
                "fx": i.fx,
                "fy": i.fy,
                "ppx": i.ppx,
                "ppy": i.ppy,
                "model": str(i.model),
                "coeffs": list(i.coeffs),
            }
        except Exception:
            self._intrinsics = None

        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=int(1000 / max(self.fps, 1)))
            except Exception:
                continue
            frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            if not color or not depth:
                continue

            ts_ms = float(color.get_timestamp())
            color_np = np.asanyarray(color.get_data()).copy()  # BGR uint8
            depth_raw = np.asanyarray(depth.get_data())        # uint16
            try:
                scale = profile.get_device().first_depth_sensor().get_depth_scale()
            except Exception:
                scale = 0.001
            depth_mm = (depth_raw.astype(np.float32) * (scale * 1000.0)).astype(np.float32)

            with self._lock:
                self._ts_ms = ts_ms
                self._color_bgr = color_np
                self._depth_mm = depth_mm

    def latest(self) -> Tuple[Optional[float], Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]]]:
        with self._lock:
            ts = self._ts_ms
            c = None if self._color_bgr is None else self._color_bgr.copy()
            d = None if self._depth_mm is None else self._depth_mm.copy()
            K = None if self._intrinsics is None else dict(self._intrinsics)
        return ts, c, d, K

def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--out_dir", type=str, default="/wonchul/outputs/object", help="데이터셋 저장 디렉토리")
    args = ap.parse_args()

    rsw = RealSenseWorker(width=IMG_W, height=IMG_H, fps=30); rsw.start(); time.sleep(0.4)
    rclpy.init()
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    from ultralytics import YOLO
    model = YOLO("/wonchul/outputs/segmentation/train4/weights/last.pt")  # load an official model
    cnt = 0
    try:
        while rclpy.ok():
            ts_ms, color, depth_mm, Kinfo = rsw.latest()
            if color is None or Kinfo is None:
                cv2.waitKey(1); continue

            view = color.copy()
            results = model(view)


            if results and results[0].masks is not None:
                result = results[0]
                boxes = result.boxes
                masks = result.masks
                masks = result.masks.data.cpu().numpy()  # mask in matrix format (N x H x W)
                overlay = view.copy()
                alpha = 0.5  # Transparency factor
                
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)] # Blue, Green, Red, Cyan

                for i, mask in enumerate(masks):
                    class_id = 0
                    mask_resized = cv2.resize(mask.astype(np.uint8), (view.shape[1], view.shape[0]))
                    mask_color = colors[class_id % len(colors)] 
                    mask_indices = mask_resized > 0
                    overlay[mask_indices] = mask_color

                    box_xyxy = list(map(int, boxes.xyxy[i]))
                    cv2.rectangle(view, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), mask_color, 0)
                    cx = int((box_xyxy[0] + box_xyxy[2])/2)
                    cy = int((box_xyxy[1] + box_xyxy[3])/2)
                    cv2.circle(view, (cx, cy), 13, (255, 0, 0), -1, cv2.LINE_AA)
                view = cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)


            cv2.imshow(WIN_NAME, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("[INFO] Quit."); break

            # if key == ord(' '):
            #     cnt += 1
            #     image_name = f'cnt_{cnt:04d}.png'
            #     cv2.imwrite(os.path.join(image_dir, image_name), color)
            #     print(f"[OK] {cnt} > saved ")

    finally:
        rsw.stop()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()