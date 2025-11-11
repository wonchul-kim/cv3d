#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data collector for hand–eye calibration (Doosan M1013 + VGC10 + RealSense, AprilTag 36h11 GridBoard)

Key improvements vs previous draft:
- Save poses with *standard* names for downstream solver:
    - T_board_cam  (4x4, object→camera as returned by estimatePoseBoard)
    - T_cam_board  (inverse of the above)
  This matches the handeye solver script.
- Added motion gating between captures using Δtranslation/Δrotation thresholds
  so you don't accidentally store near-duplicate poses.
- Median over N frames for rvec/tvec and system-time stamping kept.
- Depth ROI sanity check retained.
- Meta file written once at start; each sample stores image_name and basic diagnostics.

Usage:
  python collect_handeye_m1013_vgc10_realsense.py \
      --out_dir /path/to/dataset --input_topic /dsr01/aiv/state/broadcast \
      --robotstate_tcp_is_flange 1

Controls:
  SPACE = capture sample (only when robot STILL and sufficiently moved since last saved)
  q/ESC = quit
"""
import os, json, time, math, argparse
from collections import deque
from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
import threading

# -------------------- 기본 파라미터 --------------------
INPUT_TOPIC = "/dsr01/aiv/state/broadcast"
FLANGE_TO_TIP_M = (0.000793, 0.000745, 0.171436)  # pivot translation [m], rotation=Identity

# 보드 (AprilTag 36h11 GridBoard)
BOARD_ROWS = 6
BOARD_COLS = 7
TAG_SIZE_MM = 30
TAG_GAP_MM  = 10

# 캡처·동기화
IMG_W, IMG_H = 640, 480
DT_TOL_SEC    = 0.040   # 카메라-로봇 시간 동기 허용오차
STILL_HOLD_MS = 300     # moving=False 유지 시간
MEDIAN_N      = 5       # 이만큼 프레임 중앙값

# 캡처 간 모션 최소치 (중복 방지)
MOTION_POS_MIN_MM = 30.0
MOTION_ROT_MIN_DEG = 5.0

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

# -------------------- RobotState 구독/버퍼 --------------------
import rclpy
from rclpy.node import Node
from aivot_interfaces_v1.msg import RobotState

class RobotStateBuffer(Node):
    def __init__(self, input_topic: str, tcp_is_flange: bool):
        super().__init__("handeye_robotstate_buffer")
        self.tcp_is_flange = bool(tcp_is_flange)
        self.sub = self.create_subscription(RobotState, input_topic, self.cb, 10)
        self._buf = deque(maxlen=400)  # (t_sys, pos_m, quat_xyzw, moving)
        self._lock = threading.Lock()

    def cb(self, msg: RobotState):
        pos = msg.tcp_pos.position
        ori = msg.tcp_pos.orientation
        x = float(pos.x)/1000.0; y = float(pos.y)/1000.0; z = float(pos.z)/1000.0
        qx, qy, qz, qw = float(ori.x), float(ori.y), float(ori.z), float(ori.w)
        n = max(1e-12, np.linalg.norm([qx,qy,qz,qw]))
        qx, qy, qz, qw = qx/n, qy/n, qz/n, qw/n
        moving = bool(getattr(msg, "moving", False))
        t_sys = time.time()  # 시스템 시간으로 버퍼링
        with self._lock:
            self._buf.append((t_sys, (x,y,z), (qx,qy,qz,qw), moving))

    def is_still_for(self, hold_ms: int) -> bool:
        cutoff = time.time() - (hold_ms/1000.0)
        with self._lock:
            recent = [m for m in self._buf if m[0] >= cutoff]
            if not recent: return False
            return all((not m[3]) for m in recent)

    def get_pose_at(self, t_target: float, dt_tol: float) -> Tuple[Optional[tuple], Optional[float]]:
        with self._lock:
            if not self._buf: return None, None
            deltas = [abs(t - t_target) for (t,_,_,_) in self._buf]
            idx = int(np.argmin(deltas))
            dt = deltas[idx]
            if dt > dt_tol: return None, float(dt)
            _, pos_m, quat_xyzw, _ = self._buf[idx]
        return (pos_m, quat_xyzw), float(dt)

# -------------------- 보드/수학 유틸 --------------------

def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti

def rvec_tvec_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R,_ = cv2.Rodrigues(rvec.reshape(3,1))
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=tvec.reshape(3); return T

def rot_to_angle_deg(R: np.ndarray) -> float:
    c = (np.trace(R)-1.0)/2.0
    c = max(-1.0,min(1.0,c))
    return math.degrees(math.acos(c))

def quat_to_R(qxyzw):
    x,y,z,w = qxyzw
    n = max(1e-12, np.linalg.norm([x,y,z,w])); x,y,z,w = x/n, y/n, z/n, w/n
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ], dtype=float)
    return R

def reprojection_rmse_board(corners, ids, board, K, dist, rvec, tvec):
    obj_all, img_all = [], []
    board_corners = board.getObjPoints()
    board_ids     = board.getIds().flatten()
    id2idx = {int(i): idx for idx,i in enumerate(board_ids)}
    for det_c, det_id in zip(corners, ids.flatten()):
        det_id = int(det_id)
        if det_id not in id2idx: continue
        idx = id2idx[det_id]
        obj = board_corners[idx].reshape(-1,3); img = det_c.reshape(-1,2)
        obj_all.append(obj); img_all.append(img)
    if not obj_all: return None
    obj = np.vstack(obj_all).astype(np.float64)
    img = np.vstack(img_all).astype(np.float64)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    err = np.linalg.norm(proj - img, axis=1)
    return float(np.sqrt((err**2).mean()))

def project_board_center_px(T_board_cam, K):
    # 보드 원점(0,0,0)을 투영해 깊이 ROI를 잡는다.
    P_cam = T_board_cam[:3,3]
    X,Y,Z = P_cam
    if Z <= 0: return None
    u = K[0,0]*X/Z + K[0,2]
    v = K[1,1]*Y/Z + K[1,2]
    return int(round(u)), int(round(v))

# -------------------- 캡처 루틴 --------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def append_sample(out_path, sample: dict):
    with open(out_path, "a") as f:
        f.write(json.dumps(sample) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="/wonchul/outputs/handeye_calibration", help="데이터셋 저장 디렉토리")
    ap.add_argument("--input_topic", type=str, default=INPUT_TOPIC)
    ap.add_argument("--robotstate_tcp_is_flange", type=int, default=1)
    args = ap.parse_args()

    out_dir = args.out_dir
    ensure_dir(out_dir)
    image_dir = os.path.join(out_dir, 'images'); ensure_dir(image_dir)
    samples_path = os.path.join(out_dir, "samples.jsonl")

    # 카메라
    rsw = RealSenseWorker(width=IMG_W, height=IMG_H, fps=30); rsw.start(); time.sleep(0.4)

    # 보드
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)
    tag_m = TAG_SIZE_MM/1000.0; gap_m = TAG_GAP_MM/1000.0
    board = cv2.aruco.GridBoard((BOARD_COLS, BOARD_ROWS), tag_m, gap_m, dictionary)

    # ROS2
    rclpy.init()
    node = RobotStateBuffer(args.input_topic, tcp_is_flange=bool(args.robotstate_tcp_is_flange))

    # meta 저장
    _, _, depth_mm0, Kinfo = rsw.latest()
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "board": {"rows": BOARD_ROWS, "cols": BOARD_COLS, "tag_size_mm": TAG_SIZE_MM, "gap_mm": TAG_GAP_MM, "dict": "APRILTAG_36h11", "start_id": 0},
        "pivot_flange_to_tcp_m": list(FLANGE_TO_TIP_M),
        "robot_state_tcp_is_flange": bool(args.robotstate_tcp_is_flange),
        "image_size": [IMG_W, IMG_H],
        "intrinsics": Kinfo if Kinfo else None,
        "sync": {"dt_tol_sec": DT_TOL_SEC, "still_hold_ms": STILL_HOLD_MS, "median_N": MEDIAN_N,
                  "motion_pos_min_mm": MOTION_POS_MIN_MM, "motion_rot_min_deg": MOTION_ROT_MIN_DEG}
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    last_saved_pos = None  # (x,y,z) in meters
    last_saved_quat = None # (x,y,z,w)

    def rot_deg_between(q1, q2):
        # Convert to rotation matrices then compute angle of R1^T R2
        R1 = quat_to_R(q1); R2 = quat_to_R(q2)
        R = R1.T @ R2
        c = (np.trace(R)-1.0)/2.0; c = max(-1.0, min(1.0, c))
        return math.degrees(math.acos(c))

    print(f"[INFO] Saving to {samples_path}  (SPACE=capture, q/ESC=quit)\n")

    cnt = 0
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            ts_ms, color, depth_mm, Kinfo = rsw.latest()
            if color is None or Kinfo is None:
                cv2.waitKey(1); continue

            K = np.array([[Kinfo["fx"], 0, Kinfo["ppx"]],
                          [0, Kinfo["fy"], Kinfo["ppy"]],
                          [0, 0, 1]], dtype=np.float64)
            dist = np.array(Kinfo["coeffs"], dtype=np.float64).reshape(1, -1)

            # 미리보기 검출 (단일 프레임)
            view = color.copy()
            corners, ids, _ = detector.detectMarkers(color)
            rmse = None
            if ids is not None and len(ids)>0:
                cv2.aruco.drawDetectedMarkers(view, corners, ids)
                retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
                if retval > 0:
                    cv2.drawFrameAxes(view, K, dist, rvec, tvec, 0.1)
                    rmse = reprojection_rmse_board(corners, ids, board, K, dist, rvec, tvec)
                    if rmse is not None:
                        cv2.putText(view, f"RMSE={rmse:.2f}px", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            status = "STILL" if node.is_still_for(STILL_HOLD_MS) else "MOVING"
            cv2.putText(view, status, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0) if status=="STILL" else (0,0,255), 2)
            cv2.imshow(WIN_NAME, view)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), 27):
                print("[INFO] Quit."); break

            if key == ord(' '):
                # 1) 정지 확인
                if not node.is_still_for(STILL_HOLD_MS):
                    print("[HOLD] Robot not still."); continue

                # 2) N 프레임 중앙값 보드 포즈 + 시스템 시간
                r_list, t_list, t_sys_list = [], [], []
                t_end = time.time() + 1.0
                while len(r_list) < MEDIAN_N and time.time() < t_end:
                    _, img, depth_mm, _Kinfo = rsw.latest()
                    if img is None or _Kinfo is None:
                        time.sleep(0.005); continue
                    sys_now = time.time()
                    cs, ids_m, _ = detector.detectMarkers(img)
                    if ids_m is None or len(ids_m)==0:
                        time.sleep(0.005); continue
                    retval, rvec, tvec = cv2.aruco.estimatePoseBoard(cs, ids_m, board, K, dist, None, None)
                    if retval > 0:
                        r_list.append(rvec.reshape(3)); t_list.append(tvec.reshape(3)); t_sys_list.append(sys_now)
                    else:
                        time.sleep(0.005)
                if not r_list:
                    print("[WARN] Board pose not stable."); continue
                r_med = np.median(np.vstack(r_list), axis=0).reshape(3,1)
                t_med = np.median(np.vstack(t_list), axis=0).reshape(3,1)
                t_sys = float(np.median(np.array(t_sys_list)))
                T_board_cam = rvec_tvec_to_T(r_med, t_med)
                T_cam_board = invert_T(T_board_cam)

                # 3) 시간 동기 로봇 포즈
                tcp_pos, dt = node.get_pose_at(t_sys, DT_TOL_SEC)
                if tcp_pos is None:
                    print(f"[SYNC] Δt too large ({dt:.3f}s). Retry."); continue
                pos_m, quat_xyzw = tcp_pos

                # 4) 캡처 간 모션 체크 (중복 방지)
                if last_saved_pos is not None and last_saved_quat is not None:
                    dpos_mm = 1000.0 * float(np.linalg.norm(np.array(pos_m) - np.array(last_saved_pos)))
                    drot_deg = rot_deg_between(last_saved_quat, quat_xyzw)
                    if dpos_mm < MOTION_POS_MIN_MM and drot_deg < MOTION_ROT_MIN_DEG:
                        print(f"[SKIP] Too little motion since last capture (Δpos={dpos_mm:.1f}mm, Δrot={drot_deg:.1f}°).")
                        continue

                # 5) 깊이-스케일 교차검증용 depth median (보드 원점 근방 투영)
                z_depth_mm = None
                if depth_mm is not None:
                    uv = project_board_center_px(T_board_cam, K)
                    if uv is not None:
                        u,v = uv
                        r0,r1 = max(0,v-5), min(depth_mm.shape[0], v+6)
                        c0,c1 = max(0,u-5), min(depth_mm.shape[1], u+6)
                        patch = depth_mm[r0:r1, c0:c1]
                        vals = patch[np.isfinite(patch) & (patch>0)]
                        if vals.size>20:
                            z_depth_mm = float(np.median(vals))

                # 6) 저장
                cnt += 1
                image_name = f'cnt_{cnt:04d}.png'
                sample = {
                    "t_sys": t_sys,
                    "pos_m": list(pos_m),
                    "quat_xyzw": list(quat_xyzw),
                    "T_board_cam": T_board_cam.reshape(-1).tolist(),
                    "T_cam_board": T_cam_board.reshape(-1).tolist(),
                    "ids": [] if ids is None else np.array(ids).flatten().astype(int).tolist(),
                    "rmse_px": None if ids is None else float(reprojection_rmse_board(corners, ids, board, K, dist, r_med, t_med) or 0.0),
                    "z_depth_mm": z_depth_mm,
                    "image_name": image_name
                }
                append_sample(samples_path, sample)
                cv2.imwrite(os.path.join(image_dir, image_name), color)
                last_saved_pos, last_saved_quat = pos_m, quat_xyzw

                print(f"[OK] {cnt} > saved (Δt={dt*1000:.0f}ms, ids={len(sample['ids'])}, depthZ={z_depth_mm})")

    finally:
        rsw.stop()
        cv2.destroyAllWindows()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
