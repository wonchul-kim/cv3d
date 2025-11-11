#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline hand–eye dataset diagnoser & solver.

Inputs (from capture step):
  out_dir/
    meta.json
    samples.jsonl
  (optional) images_dir/  # if you saved raw color images per sample for re-detection

What it does:
  - Load dataset, validate matrices (orthonormal R, det≈1)
  - Robot motion stats (Δpos, Δrot)
  - If images provided: re-detect AprilTag GridBoard to verify target2cam & reprojection RMSE
  - Scale sweep on target2cam tvec: s in [0.20..2.50] → world-consistency(mm/deg) min search
  - Optional hypotheses:
      * rows/cols swap trial (if images provided)
      * start_id offset trial (if images provided; assumes contiguous ids)
      * fix TCP rotation small-grid trial (±10° around XYZ)
  - Print best combination and re-solve hand–eye with that combo
  - Save result_best.json

Requires:
  pip install opencv-contrib-python numpy tf-transformations
"""

import os, json, math, argparse, glob
import numpy as np
import cv2

# ---------- basic linalg ----------
def invert_T(T):
    R = T[:3,:3]; t = T[:3,3]
    Ti = np.eye(4); Ti[:3,:3]=R.T; Ti[:3,3]=-R.T@t; return Ti

def rot_to_angle_deg(R):
    c = (np.trace(R) - 1.0)/2.0
    c = min(1.0, max(-1.0, c))
    return math.degrees(math.acos(c))

def rpy_to_R(rx, ry, rz):  # radians
    sx, cx = math.sin(rx), math.cos(rx)
    sy, cy = math.sin(ry), math.cos(ry)
    sz, cz = math.sin(rz), math.cos(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

# ---------- dataset I/O ----------
def load_dataset(out_dir):
    meta = json.load(open(os.path.join(out_dir, "meta.json"), "r"))
    samples = []
    with open(os.path.join(out_dir, "samples.jsonl"), "r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            samples.append(json.loads(line))
    return meta, samples

def validate_rot(R, eps=1e-2):
    ok_det = abs(np.linalg.det(R) - 1.0) < 1e-2
    ok_orth = np.allclose(R.T @ R, np.eye(3), atol=eps)
    return ok_det and ok_orth, ok_det, ok_orth

# ---------- metrics ----------
def build_motions_from_absolute(T_bg_list, T_tc_list):
    Rg,tg,Rt,tt = [],[],[],[]
    for i in range(len(T_bg_list)-1):
        A = invert_T(T_bg_list[i]) @ T_bg_list[i+1]   # gripper motion
        B = T_tc_list[i] @ invert_T(T_tc_list[i+1])   # target motion (both target2cam)
        Rg.append(A[:3,:3]); tg.append(A[:3,3])
        Rt.append(B[:3,:3]); tt.append(B[:3,3])
    return np.array(Rg),np.array(tg),np.array(Rt),np.array(tt)

def residuals_AX_XB(T_tc, Rg,tg,Rt,tt):
    rot_err_deg, trans_err_mm = [], []
    for i in range(len(Rg)):
        A = np.eye(4); A[:3,:3]=Rg[i]; A[:3,3]=tg[i]
        B = np.eye(4); B[:3,:3]=Rt[i]; B[:3,3]=tt[i]
        L = A @ T_tc; R = T_tc @ B
        E = invert_T(R) @ L
        rot_err_deg.append(rot_to_angle_deg(E[:3,:3]))
        trans_err_mm.append(np.linalg.norm(E[:3,3])*1000.0)
    return np.array(rot_err_deg), np.array(trans_err_mm)

def world_consistency(T_tc, T_bg_list, T_tc_list):
    Tbbs=[]
    for Tbg,Ttc in zip(T_bg_list, T_tc_list):
        T_cam_target = invert_T(Ttc)
        T_base_cam   = Tbg @ T_tc
        T_base_board = T_base_cam @ T_cam_target
        Tbbs.append(T_base_board)
    ref = Tbbs[0]
    angs, dists = [], []
    for T in Tbbs:
        E = invert_T(ref) @ T
        angs.append(rot_to_angle_deg(E[:3,:3]))
        dists.append(np.linalg.norm(E[:3,3])*1000.0)
    return np.array(angs), np.array(dists)

# ---------- solver ----------
def solve_handeye_absolute(T_bg_list, T_tc_list, method=cv2.CALIB_HAND_EYE_TSAI):
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for Tbg,Ttc in zip(T_bg_list, T_tc_list):
        R_b_g = Tbg[:3,:3]; t_b_g = Tbg[:3,3]
        R_g_b = R_b_g.T
        t_g_b = -R_b_g.T @ t_b_g
        R_g2b.append(R_g_b); t_g2b.append(t_g_b)
        R_t2c.append(Ttc[:3,:3]); t_t2c.append(Ttc[:3,3])
    R_g2b = np.array(R_g2b); t_g2b = np.array(t_g2b)
    R_t2c = np.array(R_t2c); t_t2c = np.array(t_t2c)
    R_tc, t_tc = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T_tc = np.eye(4); T_tc[:3,:3]=R_tc; T_tc[:3,3]=t_tc.reshape(3)
    return T_tc

# ---------- image-based re-detection (optional) ----------
def redetect_target2cam_from_images(images_dir, meta, start_id=0, swap_rc=False):
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.png")) + glob.glob(os.path.join(images_dir, "*.jpg")))
    if not imgs:
        return None, None, "no_images"
    intr = meta.get("intrinsics", None)
    if not intr:
        return None, None, "no_intrinsics"
    K = np.array([[intr["fx"],0,intr["ppx"]],
                  [0,intr["fy"],intr["ppy"]],
                  [0,0,1]], dtype=np.float64)
    dist = np.array(intr.get("coeffs", [0,0,0,0,0]), dtype=np.float64).reshape(1,-1)

    rows = meta["board"]["rows"]; cols = meta["board"]["cols"]
    if swap_rc: rows, cols = cols, rows
    dict_name = meta["board"]["dict"]
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag_mm = float(meta["board"]["tag_size_mm"])
    gap_mm = float(meta["board"]["gap_mm"])
    board = cv2.aruco.GridBoard((cols, rows), tag_mm/1000.0, gap_mm/1000.0, dictionary)

    det = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    T_list = []
    rmse_list = []
    for p in imgs:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None: continue
        corners, ids, _ = det.detectMarkers(im)
        if ids is None or len(ids)==0:
            T_list.append(None); rmse_list.append(None); continue
        retval, rvec, tvec = cv2.aruco.estimatePoseBoard(corners, ids, board, K, dist, None, None)
        if retval <= 0:
            T_list.append(None); rmse_list.append(None); continue
        # reprojection RMSE
        rm = reprojection_rmse_board(corners, ids, board, K, dist, rvec, tvec)
        T_list.append(rvec_tvec_to_T(rvec, tvec))
        rmse_list.append(rm)
    return T_list, rmse_list, None

def reprojection_rmse_board(corners, ids, board, K, dist, rvec, tvec):
    obj_all, img_all = [], []
    board_corners = board.getObjPoints()
    board_ids     = board.getIds().flatten()
    id2idx = {int(i): idx for idx,i in enumerate(board_ids)}
    for det_c, det_id in zip(corners, ids.flatten()):
        di = int(det_id)
        if di not in id2idx: continue
        idx = id2idx[di]
        obj = board_corners[idx].reshape(-1,3)
        img = det_c.reshape(-1,2)
        obj_all.append(obj); img_all.append(img)
    if not obj_all: return None
    obj = np.vstack(obj_all).astype(np.float64)
    img = np.vstack(img_all).astype(np.float64)
    proj,_ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    proj = proj.reshape(-1,2)
    err = np.linalg.norm(proj - img, axis=1)
    return float(np.sqrt((err**2).mean()))

def rvec_tvec_to_T(rvec,tvec):
    R,_ = cv2.Rodrigues(np.array(rvec).reshape(3,1))
    T = np.eye(4); T[:3,:3]=R; T[:3,3]=np.array(tvec).reshape(3); return T

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default='/wonchul/outputs/hande_cal')
    ap.add_argument("--images_dir", default='/wonchul/outputs/hande_cal/images', help="optional: folder of color images for re-detection")
    ap.add_argument("--try_swap_rows_cols", action="store_true")
    ap.add_argument("--try_tcp_fixed_rotation", action="store_true", help="search ±10° small fixed R around XYZ")
    ap.add_argument("--scale_min", type=float, default=0.0020)
    ap.add_argument("--scale_max", type=float, default=2.50)
    ap.add_argument("--scale_step", type=float, default=0.05)
    args = ap.parse_args()

    meta, samples = load_dataset(args.out_dir)
    assert len(samples)>=3, "Not enough samples."

    # assemble absolute poses
    T_bg_list = []
    T_tc_list = []
    for s in samples:
        Tbg = np.array(s["T_base_tcp"], dtype=np.float64).reshape(4,4)
        Ttc = np.array(s["T_target2cam"], dtype=np.float64).reshape(4,4)
        T_bg_list.append(Tbg); T_tc_list.append(Ttc)

    # sanity on rotations
    bad = []
    for k,T in enumerate(T_bg_list):
        ok,okd,oko = validate_rot(T[:3,:3])
        if not ok: bad.append(("robot",k,okd,oko))
    for k,T in enumerate(T_tc_list):
        ok,okd,oko = validate_rot(T[:3,:3])
        if not ok: bad.append(("board",k,okd,oko))
    if bad:
        print("[SANITY] Non-orthonormal rotation found:", bad)

    # robot motion stats
    dmm_list, dang_list = [], []
    for i in range(len(T_bg_list)-1):
        E = invert_T(T_bg_list[i]) @ T_bg_list[i+1]
        dmm_list.append(np.linalg.norm(E[:3,3])*1000.0)
        dang_list.append(rot_to_angle_deg(E[:3,:3]))
    print(f"[ROBOT] Δpos mean±std={np.mean(dmm_list):.1f}±{np.std(dmm_list):.1f} mm, "
          f"Δrot mean±std={np.mean(dang_list):.1f}±{np.std(dang_list):.1f} deg")

    # optional: re-detection to verify target2cam & RMSE
    if args.images_dir:
        T_tc_redet, rmses, err = redetect_target2cam_from_images(args.images_dir, meta, swap_rc=False)
        if err:
            print(f"[REDETECT] skipped: {err}")
        else:
            val = [x for x in rmses if x is not None]
            if val:
                print(f"[REDETECT] reprojection RMSE px: mean±std={np.mean(val):.2f}±{np.std(val):.2f}, "
                      f"min={np.min(val):.2f}, max={np.max(val):.2f}")
            # if many None → board params mismatch or IDs problem

    # scale sweep on tvec
    scales = np.arange(args.scale_min, args.scale_max+1e-9, args.scale_step)
    best = None
    for s in scales:
        T_tc_scaled = []
        for T in T_tc_list:
            Ts = T.copy(); Ts[:3,3] *= s; T_tc_scaled.append(Ts)
        T_tc = solve_handeye_absolute(T_bg_list, T_tc_scaled, method=cv2.CALIB_HAND_EYE_TSAI)
        Rg,tg,Rt,tt = build_motions_from_absolute(T_bg_list, T_tc_scaled)
        ax_deg, ax_mm = residuals_AX_XB(T_tc, Rg,tg,Rt,tt)
        w_deg, w_mm = world_consistency(T_tc, T_bg_list, T_tc_scaled)

        score = ax_mm.mean() + 5.0*ax_deg.mean() + w_mm.mean() + 5.0*w_deg.mean()
        cur = dict(scale=s, score=score,
                   ax_deg_mean=float(ax_deg.mean()), ax_mm_mean=float(ax_mm.mean()),
                   w_deg_mean=float(w_deg.mean()),   w_mm_mean=float(w_mm.mean()),
                   T_tc=T_tc)
        if (best is None) or (cur["score"] < best["score"]):
            best = cur

    print("\n[SCALE SWEEP] best:")
    print(f"  scale={best['scale']:.3f} | score={best['score']:.2f} | "
          f"AX mean: {best['ax_deg_mean']:.2f} deg / {best['ax_mm_mean']:.1f} mm | "
          f"W mean: {best['w_deg_mean']:.2f} deg / {best['w_mm_mean']:.1f} mm")

    T_tc_use = best["T_tc"]
    s_use = best["scale"]

    # optional: fixed TCP rotation micro-scan (±10°) to see if rotation explains residual
    if args.try_tcp_fixed_rotation:
        fine_best = None
        for rx_deg in range(-10, 11, 5):
            for ry_deg in range(-10, 11, 5):
                for rz_deg in range(-10, 11, 5):
                    Rfix = rpy_to_R(math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg))
                    # apply as extra rotation on camera side equivalently: tweak T_tc
                    T_tcf = T_tc_use.copy(); T_tcf[:3,:3] = T_tcf[:3,:3] @ Rfix
                    Rg,tg,Rt,tt = build_motions_from_absolute(T_bg_list, [np.array(T) for T in T_tc_list])
                    ax_deg, ax_mm = residuals_AX_XB(T_tcf, Rg,tg,Rt,tt)
                    w_deg, w_mm = world_consistency(T_tcf, T_bg_list, T_tc_list)
                    score = ax_mm.mean() + 5*ax_deg.mean() + w_mm.mean() + 5*w_deg.mean()
                    cur = (score, rx_deg, ry_deg, rz_deg, T_tcf,
                           ax_deg.mean(), ax_mm.mean(), w_deg.mean(), w_mm.mean())
                    if (fine_best is None) or (score < fine_best[0]):
                        fine_best = cur
        if fine_best:
            _, rx, ry, rz, T_tc_use, axd, axm, wd, wm = fine_best
            print(f"\n[TCP ROT-SCAN] best extra R (deg): rx={rx}, ry={ry}, rz={rz} "
                  f"| AX {axd:.2f}deg/{axm:.1f}mm | W {wd:.2f}deg/{wm:.1f}mm")

    # final print & save
    try:
        from tf_transformations import quaternion_from_matrix
        qx,qy,qz,qw = quaternion_from_matrix(T_tc_use)
        t = T_tc_use[:3,3]
        print("\n[STATIC TF] vgc10_tcp -> camera_color_optical")
        print(f"ros2 run tf2_ros static_transform_publisher "
              f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f} "
              f"vgc10_tcp camera_color_optical")
    except Exception:
        pass

    out = {
        "best_scale": float(s_use),
        "T_tool_camera": T_tc_use.reshape(-1).tolist(),
        "metrics_bestscale": {
            "AX_deg_mean": best["ax_deg_mean"],
            "AX_mm_mean":  best["ax_mm_mean"],
            "W_deg_mean":  best["w_deg_mean"],
            "W_mm_mean":   best["w_mm_mean"],
        }
    }
    json.dump(out, open(os.path.join(args.out_dir, "result_best.json"), "w"), indent=2)
    print(f"\nSaved: {os.path.join(args.out_dir, 'result_best.json')}\n")

if __name__ == "__main__":
    main()
