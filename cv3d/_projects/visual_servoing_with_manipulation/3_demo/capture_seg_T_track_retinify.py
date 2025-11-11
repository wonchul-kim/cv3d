#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
pip install opencv-contrib-python ultralytics numpy==1.24
apt install xvfb
Xvfb :1 -screen 0 1024x768x16 &
export DISPLAY=:1.0
'''

'''
xhost +SI:localuser:root
'''

import os, json, time, math, argparse
from collections import deque
from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
import threading
import rclpy
import warnings
from collect_single import RobotStateBuffer, INPUT_TOPIC
from aivot_rl.tools.ros import MoveJointClient
from aivot_rl.tools.cal.common import (quat_xyzw_to_T, invert_T, T)



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


# def depth_to_color_projection(depth_image, depth_intrin, color_intrin, extrin):
#     h, w = depth_image.shape
#     color_points = np.zeros((h, w, 2), dtype=np.float32)

#     fx_d, fy_d, cx_d, cy_d = depth_intrin['fx'], depth_intrin['fy'], depth_intrin['ppx'], depth_intrin['ppy']
#     fx_c, fy_c, cx_c, cy_c = color_intrin['fx'], color_intrin['fy'], color_intrin['ppx'], color_intrin['ppy']

#     R = np.array(extrin['rotation']).reshape(3, 3)
#     t = np.array(extrin['translation']).reshape(3, 1)

#     for v in range(h):
#         for u in range(w):
#             z = depth_image[v, u]
#             if z == 0:
#                 continue

#             # Depth 픽셀 → 3D (Depth 카메라 좌표계)
#             X_d = (u - cx_d) * z / fx_d
#             Y_d = (v - cy_d) * z / fy_d
#             Z_d = z

#             # Depth → Color 카메라 좌표계 변환
#             X_c, Y_c, Z_c = (R @ np.array([[X_d], [Y_d], [Z_d]]) + t).flatten()


#             # Color 카메라 평면으로 투영
#             u_c = fx_c * X_c / Z_c + cx_c
#             v_c = fy_c * Y_c / Z_c + cy_c

#             color_points[v, u] = (u_c, v_c)

#     return color_points
import numpy as np

def depth_to_color_projection(depth_image, depth_intrin, color_intrin, extrin):
    h, w = depth_image.shape
    
    # --- 1. Extract Intrinsics and Extrinsics ---
    fx_d, fy_d, cx_d, cy_d = depth_intrin['fx'], depth_intrin['fy'], depth_intrin['ppx'], depth_intrin['ppy']
    fx_c, fy_c, cx_c, cy_c = color_intrin['fx'], color_intrin['fy'], color_intrin['ppx'], color_intrin['ppy']

    R = np.array(extrin['rotation']).reshape(3, 3)
    t = np.array(extrin['translation']).reshape(3, 1)

    # --- 2. Create Pixel Coordinate Grid and Filter Zeros ---
    # Create U and V coordinate grids for all pixels
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))
    
    # Get the Z (depth) values
    Z_d = depth_image.flatten()
    
    # Get the corresponding U and V coordinates, and filter points where depth is zero
    valid_mask = Z_d > 0
    Z_valid = Z_d[valid_mask]
    u_valid = u_grid.flatten()[valid_mask]
    v_valid = v_grid.flatten()[valid_mask]
    
    # --- 3. Depth Pixels to 3D Points (Depth Camera Coordinates) ---
    # X_d = (u - cx_d) * Z / fx_d
    # Y_d = (v - cy_d) * Z / fy_d
    
    X_valid_d = (u_valid - cx_d) * Z_valid / fx_d
    Y_valid_d = (v_valid - cy_d) * Z_valid / fy_d
    
    # Stack into a 3xN matrix of homogeneous coordinates (X_d, Y_d, Z_d)
    points_3d_d = np.vstack([X_valid_d, Y_valid_d, Z_valid]) # Shape (3, N)

    # --- 4. Transform 3D Points (Depth -> Color Camera Coordinates) ---
    # P_c = R @ P_d + t
    points_3d_c = R @ points_3d_d + t # Shape (3, N)

    # Extract X_c, Y_c, Z_c
    X_c, Y_c, Z_c = points_3d_c[0, :], points_3d_c[1, :], points_3d_c[2, :]

    # --- 5. Project 3D Points to Color Image Plane ---
    # u_c = fx_c * X_c / Z_c + cx_c
    # v_c = fy_c * Y_c / Z_c + cy_c
    
    # The division by Z_c is necessary for perspective projection
    # Use np.divide for element-wise division. Z_c is guaranteed to be non-zero
    # (since Z_d > 0 and the camera is typically in front of the object, R is not singular)
    u_c_valid = fx_c * np.divide(X_c, Z_c) + cx_c
    v_c_valid = fy_c * np.divide(Y_c, Z_c) + cy_c

    # --- 6. Map back to the original image structure ---
    color_points = np.zeros((h * w, 2), dtype=np.float32)
    
    # Assign the calculated (u_c, v_c) to the corresponding flat index
    flat_indices = np.where(valid_mask)[0]
    color_points[flat_indices, 0] = u_c_valid
    color_points[flat_indices, 1] = v_c_valid

    # Reshape the result back to (H, W, 2)
    return color_points.reshape(h, w, 2)

def disparity_to_depth(
    disp: np.ndarray,
    fx: float,
    baseline_m: float,
    disp_scale: float = 1.0,     # disparity가 1/16 스케일이면 1/16 대신 1/16의 역수(=1/16? → 보통 1/16로 저장되니 1/16을 곱해 'px'로 맞추세요)
    min_disp: float = 1e-6,      # 0 또는 너무 작은 disparity 무효화
    invalid_val: float = np.nan  # 무효 픽셀은 NaN으로
) -> np.ndarray:
    """
    disp: disparity map (px 단위, float 권장. int라면 float32로 변환)
    fx:   left 카메라 내접행렬 K1[0,0]
    baseline_m: 두 카메라 사이 거리 [m]
    disp_scale: disp 단위 보정(예: SGBM 16배 스케일은 disp_scale=1/16)
    """
    disp = disp.astype(np.float32) * disp_scale
    depth = (fx * baseline_m) / (disp + 1e-12)  # 분모 0 방지

    # 무효 마스크 처리
    invalid_mask = (disp <= min_disp) | ~np.isfinite(disp)
    depth[invalid_mask] = invalid_val
    return depth


def profile_to_dict(p: rs.stream_profile):
    d = {
        "stream": str(p.stream_type()),
        "fmt": str(p.format()),
        "index": p.stream_index(),
        "fps": p.fps()
    }
    # 비디오 스트림이면 해상도/내부파라미터 추출
    try:
        v = p.as_video_stream_profile()
        i = v.get_intrinsics()
        d.update({
            "width": v.width(),
            "height": v.height(),
            "intrinsics": {
                "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "coeffs": list(i.coeffs), "model": str(i.model)
            }
        })
    except Exception:
        pass
    return d

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
        self._ir_left_raw: Optional[np.ndarray] = None
        self._ir_right_raw: Optional[np.ndarray] = None
        self._info: Optional[Dict[str, Any]] = {}

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
        # cfg.enable_stream(rs.stream.infrared, self.width, self.height, rs.format.y8, self.fps)
        cfg.enable_stream(rs.stream.infrared, 1, self.width, self.height, rs.format.y8, self.fps) # 왼쪽
        cfg.enable_stream(rs.stream.infrared, 2, self.width, self.height, rs.format.y8, self.fps) # 오른쪽
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
        
        def find_stereo_sensor(dev: rs.device):
            for s in dev.sensors:
                # 이름 기준(주로 "Stereo Module") 또는 depth_sensor로 캐스팅 가능 여부
                name = s.get_info(rs.camera_info.name) if s.supports(rs.camera_info.name) else ""
                try:
                    _ = rs.depth_sensor(s)  # 캐스팅 성공하면 스테레오 모듈
                    return s, name
                except Exception:
                    pass
            return None, None
        dev = profile.get_device()
        stereo, name = find_stereo_sensor(dev)
        if stereo is None:
            raise RuntimeError("Stereo Module(IR) 센서를 찾지 못했습니다.")

        def opt_range(opt):
            r = stereo.get_option_range(opt)
            return r.min, r.max, r.step, r.default
        
        def clamp(v, lo, hi): return max(lo, min(hi, v))
        
        ### (1) 오토 노출 끄기 → 수동으로 올리기
        if stereo.supports(rs.option.enable_auto_exposure):
            stereo.set_option(rs.option.enable_auto_exposure, 0)
        
        ### (2) 노출(µs) & 게인 올리기
        if stereo.supports(rs.option.exposure):
            lo, hi, st, de = opt_range(rs.option.exposure)
            ### 30fps면 노출 상한이 짧습니다; 15/6fps로 낮추면 더 긴 노출 허용
            target_exposure = clamp(20000, lo, hi) # gain
            stereo.set_option(rs.option.exposure, target_exposure)
            print("exposure set:", target_exposure)

        if stereo.supports(rs.option.gain):
            lo, hi, st, de = opt_range(rs.option.gain)
            target_gain = clamp(50, lo, hi) # gain
            stereo.set_option(rs.option.gain, target_gain)
            print("gain set:", target_gain)


        # Color intrinsics
        try:
            self._info.update({"streams": {}})
            for sp in profile.get_streams():
                try:
                    k = f"{str(sp.stream_type())}_{sp.stream_index()}"
                    self._info["streams"].update({k: {
                                "profile": profile_to_dict(sp),
                                "extrinsics_to_color": None
                            }
                        }
                    )
                except Exception:
                    pass
            
            try:
                color_sp = profile.get_stream(rs.stream.color)
                for sp in profile.get_streams():
                    try:
                        ex = sp.get_extrinsics_to(color_sp)
                        k = f"{str(sp.stream_type())}_{sp.stream_index()}"
                        if k in self._info["streams"]:
                            self._info["streams"][k].update({"extrinsics_to_color":{
                                        "rotation": list(ex.rotation),
                                        "translation": list(ex.translation)
                                    }
                            })
                    except Exception:
                        pass
            except Exception:
                pass

            # color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            # i = color_stream.get_intrinsics()
            # self._info = {
            #     'color': {
            #         "width": i.width,
            #         "height": i.height,
            #         "fx": i.fx,
            #         "fy": i.fy,
            #         "ppx": i.ppx,
            #         "ppy": i.ppy,
            #         "model": str(i.model),
            #         "coeffs": list(i.coeffs),
            #     }

            # }
        except Exception:
            self._info = None

        while not self._stop.is_set():
            try:
                frames = self._pipe.wait_for_frames(timeout_ms=int(1000 / max(self.fps, 1)))
            except Exception:
                continue
            frames = self._align.process(frames)
            color = frames.get_color_frame()
            depth = frames.get_depth_frame()
            ir_left = frames.get_infrared_frame(1)
            ir_right = frames.get_infrared_frame(2)

            if not color or not depth or not ir_left or not ir_right:
                continue

            ts_ms = float(color.get_timestamp())
            color_np = np.asanyarray(color.get_data()).copy()  # BGR uint8
            depth_raw = np.asanyarray(depth.get_data())        # uint16
            ir_left_raw = np.asanyarray(ir_left.get_data()).copy()
            ir_right_raw = np.asanyarray(ir_right.get_data()).copy()
            try:
                scale = profile.get_device().first_depth_sensor().get_depth_scale()
            except Exception:
                scale = 0.001
            depth_mm = (depth_raw.astype(np.float32) * (scale * 1000.0)).astype(np.float32)

            with self._lock:
                self._ts_ms = ts_ms
                self._color_bgr = color_np
                self._depth_mm = depth_mm
                self._ir_left_raw = ir_left_raw
                self._ir_right_raw = ir_right_raw

    def latest(self) -> Tuple[Optional[float], Optional[np.ndarray], 
                              Optional[np.ndarray], Optional[np.ndarray],
                              Optional[np.ndarray],
                              Optional[Dict[str, Any]]]:
        with self._lock:
            ts = self._ts_ms
            c = None if self._color_bgr is None else self._color_bgr.copy()
            d = None if self._depth_mm is None else self._depth_mm.copy()
            i_left = None if self._ir_left_raw is None else self._ir_left_raw.copy() 
            i_right = None if self._ir_right_raw is None else self._ir_right_raw.copy() 
            K = None if self._info is None else dict(self._info)
        return ts, c, d, i_left, i_right, K

T_cam2tcp = np.array([
            [ 0.70350934, -0.60118006, -0.37902129,  0.06244145],
            [ 0.71048044,  0.60776357,  0.35474073, -0.07640022],
            [ 0.01709228, -0.51885063,  0.85469403,  0.23122382],
            [ 0.        ,  0.        ,  0.        ,  1.        ],
        ])


def robust_depth_from_mask(depth_mm, mask, cx, cy, win=15):
    """마스크와 중심 주변 작은 윈도우를 교집합해서 median 깊이(m) 계산."""
    h, w = depth_mm.shape
    x0, x1 = max(0, cx-win), min(w, cx+win+1)
    y0, y1 = max(0, cy-win), min(h, cy+win+1)
    sub = depth_mm[y0:y1, x0:x1]
    msk = (mask[y0:y1, x0:x1] > 0) & (sub > 0)
    vals = sub[msk].astype(np.float32)
    if vals.size < 20:     # 너무 적으면 윈도우만 사용
        msk = (sub > 0)
        vals = sub[msk].astype(np.float32)
    if vals.size == 0:
        return None
    return float(np.median(vals) / 1000.0)  # mm -> m

def pixel_depth_to_cam3d(u, v, depth_m, Kinfo):
    fx, fy = Kinfo['fx'], Kinfo['fy']
    cx, cy = Kinfo['ppx'], Kinfo['ppy']
    X = (u - cx) / fx * depth_m
    Y = (v - cy) / fy * depth_m
    Z = depth_m
    return np.array([X, Y, Z], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--out_dir", type=str, default="/wonchul/outputs/object", help="데이터셋 저장 디렉토리")
    args = ap.parse_args()

    rsw = RealSenseWorker(width=IMG_W, height=IMG_H, fps=30)
    rsw.start()
    time.sleep(0.4)
    rclpy.init()
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    
    robot_state_node = RobotStateBuffer(INPUT_TOPIC, tcp_is_flange=True)
    from aivot_rl.tools.ros.add_motion_tf import RobotMotionPublisher
    robot_ctr_node = RobotMotionPublisher()

    from ultralytics import YOLO
    model = YOLO("/wonchul/outputs/segmentation/train4/weights/last.pt")
    cnt = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(robot_state_node, timeout_sec=0.01)
            ts_ms, color, rs_depth_mm, ir_left, ir_right, Kinfo = rsw.latest()

            if color is None or Kinfo is None or ir_left is None or ir_right is None:
                cv2.waitKey(1); continue

            import sys
            build_dir = '/wonchul/retinify/pybind/build'
            sys.path.append(build_dir) 
            import retinify_py

            ir1 = Kinfo['streams']["stream.infrared_1"]
            ir2 = Kinfo['streams']["stream.infrared_2"]

            def K_from_intr(i):
                fx, fy, cx, cy = i["fx"], i["fy"], i["ppx"], i["ppy"]
                K = np.array([[fx, 0,  cx],
                            [0,  fy, cy],
                            [0,  0,  1 ]], dtype=np.float64)
                return K

            K1 = K_from_intr(ir1["profile"]["intrinsics"])
            K2 = K_from_intr(ir2["profile"]["intrinsics"])

            # RealSense의 distortion "brown_conrady"는 OpenCV의 (k1,k2,p1,p2,k3)와 호환
            def D_from_coeffs(c):
                # JSON엔 5개가 모두 0.0으로 들어있네요. 필요 시 k4..k6까지 확장 가능
                k1,k2,p1,p2,k3 = c[:5]
                return np.array([k1,k2,p1,p2,k3], dtype=np.float64)

            D1 = D_from_coeffs(ir1["profile"]["intrinsics"]["coeffs"])
            D2 = D_from_coeffs(ir2["profile"]["intrinsics"]["coeffs"])

            # --- Extrinsics (IRi -> Color) ---
            def R_t_from_extr(e):
                R = np.array(e["rotation"], dtype=np.float64).reshape(3,3)
                t = np.array(e["translation"], dtype=np.float64).reshape(3,1)  # meters
                return R, t

            R1c, t1c = R_t_from_extr(ir1["extrinsics_to_color"])
            R2c, t2c = R_t_from_extr(ir2["extrinsics_to_color"])

            # IR1 -> IR2 상대 외부파라미터 (핵심)
            # X_c = R_ic * X_i + t_ic  ,  X_2 = R_2c^{-1} * (X_c - t_2c)
            # => X_2 = R_2c^{-1} * (R_1c * X_1 + t_1c - t_2c)
            R_1to2 = R2c.T @ R1c
            T_1to2 = R2c.T @ (t1c - t2c)   # (3,1)

            frame_size = (640, 480)

            flags = cv2.CALIB_ZERO_DISPARITY   # 왼쪽 기준으로 주시점 정렬
            alpha = 0                         # 0=크롭, 1=풀뷰. 필요 시 조절
            R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
                K1, D1, K2, D2, frame_size, R_1to2, T_1to2, flags=flags, alpha=alpha
            )

            # ====== Rectify 맵 생성 ======
            mapLx, mapLy = cv2.initUndistortRectifyMap(K1, D1, R1, P1, frame_size, cv2.CV_32FC1)
            mapRx, mapRy = cv2.initUndistortRectifyMap(K2, D2, R2, P2, frame_size, cv2.CV_32FC1)

            # imgL = cv.imread(lf, cv.IMREAD_GRAYSCALE)
            # imgR = cv.imread(rf, cv.IMREAD_GRAYSCALE)

            rectL = cv2.remap(ir_left, mapLx, mapLy, cv2.INTER_LINEAR)
            rectR = cv2.remap(ir_right, mapRx, mapRy, cv2.INTER_LINEAR)

            # 스캔라인 확인용 시각화
            vis_rectified = np.hstack((cv2.cvtColor(rectL, cv2.COLOR_GRAY2BGR),
                            cv2.cvtColor(rectR, cv2.COLOR_GRAY2BGR)))
            for y in range(0, rectL.shape[0], 50):
                cv2.line(vis_rectified, (0, y), (vis_rectified.shape[1]-1, y), (0, 255, 0), 1)
            # cv2.imwrite(os.path.join(output_dir, f"rect_with_lines_{lbase}_{rbase}.png"), vis)

            # print("Done. Q matrix saved? 필요하면 np.savez로 K1,D1,K2,D2,R1,R2,P1,P2,Q를 저장하세요.")

            disparity = retinify_py.get_depth(rectL, rectR)
            colored_disparity = retinify_py.colorize_native(disparity, scale=256.0)

            retinify_depth_m = disparity_to_depth(disparity,
                                         K1[0, 0], 
                                         np.linalg.norm(T_1to2),
                                         disp_scale=1.0, min_disp=0.1)

            ### alignment
            mapping = depth_to_color_projection(retinify_depth_m, 
                              Kinfo['streams']['stream.depth_0']['profile']['intrinsics'], 
                              Kinfo['streams']['stream.color_0']['profile']['intrinsics'],
                              Kinfo['streams']['stream.infrared_2']['extrinsics_to_color'])

            # aligned_depth = np.zeros_like(retinify_depth_m) 
            # for v in range(retinify_depth_m.shape[0]): 
            #     for u in range(retinify_depth_m.shape[1]): 
            #         u_c, v_c = mapping[v, u] 
            #         if np.isnan(u_c) or np.isnan(v_c):
            #            continue 
            #         if 0 <= int(u_c) < color.shape[1] and 0 <= int(v_c) < color.shape[0]: 
            #             aligned_depth[int(v_c), int(u_c)] = retinify_depth_m[v, u] 

            # 1. Separate the u and v coordinates from the mapping
            #    Assuming mapping has shape (H_depth, W_depth, 2) where
            #    mapping[v, u] = [u_c, v_c] (column, row in color image space)
            u_c = mapping[..., 0]
            v_c = mapping[..., 1]

            # 2. Filter out NaN and out-of-bounds coordinates
            #    This creates a boolean mask for valid source (depth) and target (color) pixels.
            H_color, W_color = color.shape[:2] # Get the dimensions of the color image

            # Create a mask for valid coordinates: not NaN AND within color image bounds
            valid_mask = ~np.isnan(u_c) & ~np.isnan(v_c) & \
                        (u_c >= 0) & (u_c < W_color) & \
                        (v_c >= 0) & (v_c < H_color)

            # 3. Get the valid integer indices for the target (color) image
            #    These are the coordinates [v_c, u_c] where depth values will be placed.
            target_v_coords = v_c[valid_mask].astype(int)
            target_u_coords = u_c[valid_mask].astype(int)

            # 4. Get the source depth values corresponding to the valid indices
            #    These are the depth values that will be moved.
            source_depth_values = retinify_depth_m[valid_mask]

            # 5. Perform the vectorized assignment
            #    Initialize aligned_depth just like before
            aligned_depth = np.zeros_like(retinify_depth_m) # Note: If color.shape is different from retinify_depth_m.shape,
                                                            # you should initialize aligned_depth to np.zeros(color.shape[:2])

            # Direct assignment using advanced indexing:
            aligned_depth[target_v_coords, target_u_coords] = source_depth_values

            rt_aligned_depth_vis = aligned_depth.copy()
            rt_aligned_depth_vis = (rt_aligned_depth_vis / 5.0 * 255).astype(np.uint8)
            rt_aligned_depth_vis = cv2.applyColorMap(255 - rt_aligned_depth_vis, cv2.COLORMAP_JET)

            view = color.copy()
            results = model(view, verbose=False, conf=0.3)
            _cx, _cy = 320, 240
            cv2.circle(view, (_cx, _cy), 30, (0, 255, 0), 0, cv2.LINE_AA)

            if results and results[0].masks is not None:
                result = results[0]
                boxes = result.boxes
                masks = result.masks
                masks = result.masks.data.cpu().numpy()  # mask in matrix format (N x H x W)
                overlay = view.copy()
                alpha = 0.5  # Transparency factor
                
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)] # Blue, Green, Red, Cyan
                flag = False  
                for i, mask in enumerate(masks):
                    class_id = 0
                    mask_resized = cv2.resize(mask.astype(np.uint8), (view.shape[1], view.shape[0]))
                    mask_color = colors[class_id % len(colors)] 
                    mask_indices = mask_resized > 0
                    overlay[mask_indices] = mask_color

                    box_xyxy = list(map(int, boxes.xyxy[i]))
                    M = cv2.moments(mask_resized.astype(np.uint8), binaryImage=True)
                    if abs(M["m00"]) < 1e-6:
                        continue
                    cx = int(M["m10"]/M["m00"])
                    cy = int(M["m01"]/M["m00"])

                    if retinify_depth_m is not None:
                        depth_mm = retinify_depth_m*1000
                    else:
                        depth_mm = rs_depth_mm
                    depth_m = robust_depth_from_mask(depth_mm, mask_resized, cx, cy, win=20)
                    if depth_m is not None and (depth_m > 1 or depth_m < 0.2):
                        depth_m = None
                    if depth_m is None:
                        warnings.warn("depth is none")
                        continue

                    p_cam = pixel_depth_to_cam3d(cx, cy, depth_m, 
                                    Kinfo['streams']['stream.color_0']['profile']['intrinsics'])
                    tcp_pos, joint_pos, dt = robot_state_node.get_pose_at(time.time(), 5)
                    pos_m, quat_xyzw = tcp_pos

                    T_flange2base = quat_xyzw_to_T(pos_m, quat_xyzw)
                    T_flange2tcp = T(np.eye(3), [0.000793, 0.000745, 0.171436])
                    T_tcp2base = T_flange2base @ invert_T(T_flange2tcp)
                    # T_cam2base = T_cam2tcp @ T_tcp2base
                    T_cam2base = T_tcp2base@T_cam2tcp
                    p_cam_h  = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
                    p_base_h = T_cam2base @ p_cam_h
                    p_base   = p_base_h[:3]
                    
                    cv2.rectangle(view, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), mask_color, 0)
                    cv2.circle(view, (cx, cy), 13, (255, 0, 0), -1, cv2.LINE_AA)
                    cv2.putText(view, f"Z={depth_m:.3f}m", (cx+18, cy-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                           
                    if p_base is None:
                        print("[WARN] 타겟 없음/깊이 실패")
                    else:
                        x,y,z = p_base.tolist()
                        print(f'[INFO] Depth is {depth_m:.3f}')
                        print(f"[INFO] target in BASE: {x:.3f}, {y:.3f}, {z:.3f}")

                        print("cx,cy:", cx, cy, "  depth_m:", depth_m)
                        print("pos_m: ", pos_m)
                        print("p_cam:", p_cam)
                        print("p_base (R@p+t):", p_base)
                        print("=====================================")

                        # ---------- Visual Servoing (TOOL frame, RELATIVE) ----------
                        p_cam_c = pixel_depth_to_cam3d(_cx, _cy, depth_m, 
                                        Kinfo['streams']['stream.color_0']['profile']['intrinsics'])
                        p_cam_c_h  = np.array([p_cam_c[0], p_cam_c[1], p_cam_c[2], 1.0])
                        p_base_c_h = T_cam2base @ p_cam_c_h
                        p_base_c   = p_base_c_h[:3]
                        


                        ex = p_base_c[0] - p_base[0] 
                        ey = p_base_c[1] - p_base[1]
                        ez = 0
                        print("p_base: ", p_base)
                        print("p_base_c: ", p_base_c)
                        print("ex, ey, ez: ", ex, ey, ez)


                        Kx, Ky, Kz = 0.6, 0.6, 0.8

                        max_step = 0.01
                        deadband_xy = 0.002
                        deadband_z  = 0.005

                        dx = -Kx * ex
                        dy = -Ky * ey
                        dz = -Kz * ez

                        # 데드밴드
                        if abs(dx) < deadband_xy: dx = 0.0
                        if abs(dy) < deadband_xy: dy = 0.0
                        if abs(dz) < deadband_z : dz = 0.0

                        # 포화
                        def clamp(v, lim): 
                            return max(-lim, min(lim, v))
                        dx = clamp(dx, max_step)
                        dy = clamp(dy, max_step)
                        dz = clamp(dz, max_step)


                        # 컨트롤러가 mm 기준이므로 m → mm 변환
                        to_mm = 1000.0
                        cur_position = [pos_m[0]*1000, pos_m[1]*1000, pos_m[2]*1000, 
                                       quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]]
                        rel_position = [dx*to_mm, dy*to_mm, dz*to_mm]

                        position = [0]*7
                        position[:3] = [a + b for a, b in zip(cur_position[:3], rel_position[:3])]
                        position[3:] = cur_position[3:]
                        print('cur_position: ', cur_position)
                        print('rel_position: ', rel_position)
                        print('position: ', position)

                        ### TOOL 기준, 상대 이동을 주기적으로 송신
                        # robot_ctr_node.publish_tf_move(
                        #         position=position,
                        #         vel=10.,
                        #         acc=10.,
                        #     )
                        # rclpy.spin_once(robot_ctr_node, timeout_sec=0.05)
                        # time.sleep(1)
                        flag = True
                        # # ------------------------------------------------------------

                    cv2.putText(view, f"p_base: x={p_base[0]:.3f}m, y={p_base[1]:.3f}m, z={p_base[2]:.3f}m", 
                                (80, 430),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(view, f"p_base_c: x={p_base_c[0]:.3f}m, y={p_base_c[1]:.3f}m, z={p_base_c[2]:.3f}m", 
                                (80, 455),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

                view = cv2.addWeighted(overlay, alpha, view, 1 - alpha, 0)
                if flag:
                    cv2.imwrite(f"/wonchul/outputs/images/{cnt}.png", view)
                    cnt += 1

            rs_depth_vis = rs_depth_mm.copy()/1000
            rs_depth_vis = (rs_depth_vis / 5.0 * 255).astype(np.uint8)
            rs_depth_vis = cv2.applyColorMap(255 - rs_depth_vis, cv2.COLORMAP_JET)
            rt_depth_vis = retinify_depth_m.copy()
            rt_depth_vis = (rt_depth_vis / 5.0 * 255).astype(np.uint8)
            rt_depth_vis = cv2.applyColorMap(255 - rt_depth_vis, cv2.COLORMAP_JET)
            vis_res = np.vstack([vis_rectified,
                                  np.hstack([view, rs_depth_vis]),
                                  np.hstack([colored_disparity, rt_depth_vis]),
                                  np.hstack([rt_aligned_depth_vis, rt_aligned_depth_vis]),
                            ])
            cv2.imshow(WIN_NAME, vis_res)
            cv2.imwrite(f"/wonchul/outputs/images/{cnt}.png", vis_res)
            cnt += 1
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("[INFO] Quit."); break

            if key == ord(' '):
                pass
                    # === 여기서 여러분 로봇 API 호출로 TCP 목표 포즈 전송 ===
                    # 예: 현재 TCP의 자세는 유지, 위치만 바꿔보기
                    # robot.move_tcp_linear(pos=[x, y, z], ori=current_ori, speed=...)

    finally:
        rsw.stop()
        cv2.destroyAllWindows()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()