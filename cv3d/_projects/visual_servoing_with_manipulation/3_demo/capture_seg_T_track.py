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

    rsw = RealSenseWorker(width=IMG_W, height=IMG_H, fps=30); rsw.start(); time.sleep(0.4)
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
            ts_ms, color, depth_mm, Kinfo = rsw.latest()
            if color is None or Kinfo is None:
                cv2.waitKey(1); continue

            view = color.copy()
            results = model(view, verbose=False, conf=0.3)
            target_base_xyz = None
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

                    depth_m = robust_depth_from_mask(depth_mm, mask_resized, cx, cy, win=20)
                    if depth_m is not None and (depth_m > 1 or depth_m < 0.2):
                        depth_m = None
                    if depth_m is None:
                        warnings.warn("depth is none")
                        continue

                    p_cam = pixel_depth_to_cam3d(cx, cy, depth_m, Kinfo)
                    tcp_pos, joint_pos, dt = robot_state_node.get_pose_at(time.time(), 1)
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

                        print("K:", Kinfo['fx'], Kinfo['fy'], 
                                    Kinfo['ppx'], Kinfo['ppy'])
                        print("cx,cy:", cx, cy, "  depth_m:", depth_m)
                        print("pos_m: ", pos_m)
                        print("p_cam:", p_cam)
                        print("p_base (R@p+t):", p_base)
                        print("=====================================")

                        # ---------- Visual Servoing (TOOL frame, RELATIVE) ----------
                        p_cam_c = pixel_depth_to_cam3d(_cx, _cy, depth_m, Kinfo)
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

                        # from aivot_rl.tools.envs.m1013 import M1013Kinematics
                        # k = M1013Kinematics('/wonchul/aivot-rl/aivot_rl/tools/envs/configs/doosan/m1013_modified_v2.urdf',
                        #         base_elements=['base_link'],
                        #         active_links_mask=[False, True, True, True, True, True, True, False])
                        # initial_position_rad = np.zeros(8)
                        # initial_position_rad[1:7] = np.deg2rad(joint_pos)
                        # rotation = np.eye(3)
                        # position_m = [v/1000 for v in position[:3]]
                        # transformation_matrix = np.vstack([np.hstack([rotation, np.array(position_m).reshape(-1, 1)]), [0, 0, 0, 1]])
                        # new_joint_pos_rad = k.ik_frame(target=transformation_matrix, 
                        #                             initial_position=initial_position_rad,
                        #                             orientation_mode="all")
                        # new_pos = k.fk(new_joint_pos_rad)
                        # new_joint_pos_deg = np.rad2deg(new_joint_pos_rad)
                        # print('new_pos: ', new_pos)
                        # print('position_m: ', position_m)
                        # print('new_joint_pos_deg: ', new_joint_pos_deg)
                        # print('joint_pos: ', joint_pos)
                        # print("--------------------------------------------------")

                        # new_joint_pos_rad = k.ik(target_position=position_m)[1:7]
                        # new_pos = k.fk(new_joint_pos_rad)
                        # new_joint_pos_deg = np.rad2deg(new_joint_pos_rad)
                        # print('new_pos: ', new_pos)
                        # print('position_m: ', position_m)
                        # print('new_joint_pos_deg: ', new_joint_pos_deg)
                        # print('joint_pos: ', joint_pos)

                        # TOOL 기준, 상대 이동을 주기적으로 송신
                        # for _ in range(5):
                        robot_ctr_node.publish_tf_move(
                                position=position,
                                vel=10.,                # mm/s
                                acc=10.,               # mm/s^2
                            )
                        rclpy.spin_once(robot_ctr_node, timeout_sec=0.05)
                        time.sleep(1)
                        flag = True
                            # _tcp_pos, _ = robot_state_node.get_pose_at(time.time(), 10)
                            # if position[0] - _tcp_pos[0]*1000 < 5 and position[1] - _tcp_pos[1]*1000 < 5 and position[2] - _tcp_pos[2]*1000 < 5:
                            #     break
                            

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


            cv2.imshow(WIN_NAME, view)
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