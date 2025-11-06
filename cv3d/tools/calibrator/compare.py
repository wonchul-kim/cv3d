#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import cv2 as cv

def load_meta(json_path):
    with open(json_path, "r") as f:
        m = json.load(f)
    ir1 = m["streams"]["stream.infrared_1"]
    ir2 = m["streams"]["stream.infrared_2"]
    return ir1, ir2

def K_D_size_from_intr(intr):
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]], np.float64)
    # RealSense JSON의 IR 왜곡계수는 주로 5개(k1,k2,p1,p2,k3). 종종 0으로 저장됨.
    D = np.array(intr["coeffs"][:5], np.float64)
    size = (int(intr["width"]), int(intr["height"]))  # (W,H)
    return K, D, size

def Rt_from_extr(e):
    R = np.array(e["rotation"], np.float64).reshape(3,3)
    t = np.array(e["translation"], np.float64).reshape(3,1)
    return R, t

def build_rectify_maps_from_meta(meta_json):
    ir1, ir2 = load_meta(meta_json)
    K1, D1, size1 = K_D_size_from_intr(ir1["profile"]["intrinsics"])
    K2, D2, size2 = K_D_size_from_intr(ir2["profile"]["intrinsics"])
    assert size1 == size2, "IR1/IR2 해상도 불일치"

    R1c, t1c = Rt_from_extr(ir1["extrinsics_to_color"])
    R2c, t2c = Rt_from_extr(ir2["extrinsics_to_color"])
    # IR1 -> IR2
    R_1to2 = R2c.T @ R1c
    T_1to2 = R2c.T @ (t1c - t2c)

    frame_size = size1  # (W,H)
    flags, alpha = cv.CALIB_ZERO_DISPARITY, 0
    R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
        K1, D1, K2, D2, frame_size, R_1to2, T_1to2, flags=flags, alpha=alpha
    )
    mapLx, mapLy = cv.initUndistortRectifyMap(K1, D1, R1, P1, frame_size, cv.CV_32FC1)
    # Right 맵은 지금 비교에 필요 없음
    return (mapLx, mapLy), frame_size

def metrics(gt, pr, mask):
    a = gt[mask]; b = pr[mask]
    out = {}
    if a.size == 0:
        for k in ["count","MAE","RMSE","AbsRel","SqRel","δ<1.25","δ<1.25^2","δ<1.25^3",">5cm bad%!",">10cm bad%!"]:
            out[k] = np.nan
        out["count"] = 0
        return out
    diff = np.abs(a - b)
    out["count"] = int(a.size)
    out["MAE"] = float(np.mean(diff))
    out["RMSE"] = float(np.sqrt(np.mean((a-b)**2)))
    out["AbsRel"] = float(np.mean(diff / np.maximum(a,1e-6)))
    out["SqRel"]  = float(np.mean(((a-b)**2) / np.maximum(a,1e-6)))
    ratio = np.maximum(a/b, b/a)
    out["δ<1.25"]   = float(np.mean(ratio < 1.25))
    out["δ<1.25^2"] = float(np.mean(ratio < 1.25**2))
    out["δ<1.25^3"] = float(np.mean(ratio < 1.25**3))
    out[">5cm bad%!"]  = float(np.mean(diff > 0.05) * 100.0)
    out[">10cm bad%!"] = float(np.mean(diff > 0.10) * 100.0)
    return out

def viz_depth(z, lo=0.2, hi=6.0):
    z = z.copy().astype(np.float32)
    z[z<=0] = np.nan
    vmax = np.nanpercentile(z, 95) if np.isfinite(z).any() else hi
    hi = min(hi, vmax) if vmax>0 else hi
    n  = (z - lo) / max(hi-lo, 1e-6)
    n  = np.clip(n, 0, 1)
    g  = (n * 255).astype(np.uint8)
    return cv.applyColorMap(g, cv.COLORMAP_MAGMA)

def main():
    ap = argparse.ArgumentParser(description="Compare retinify depth.npy vs RealSense depth_m.npy")
    ap.add_argument("--ret_depth", default='/HDD/etc/outputs/intelrealsense/retinify/depth.npy')
    ap.add_argument("--rs_depth",  default='/HDD/etc/outputs/intelrealsense/inputs/20250924T073057_963938Z_211122062694_depth_m.npy')
    ap.add_argument("--meta_json", default='/HDD/etc/outputs/intelrealsense/meta/calibration_211122062694.json')
    ap.add_argument("--rs_is_rectified", action="store_true", help="RealSense depth가 이미 IR-Left rectified 좌표면 지정")
    ap.add_argument("--ret_scale", type=float, default=1.0, help="retinify depth 보정 스케일 (기본 1)")
    ap.add_argument("--ret_offset", type=float, default=0.0, help="retinify depth 보정 오프셋 (m)")
    ap.add_argument("--min_z", type=float, default=0.2)
    ap.add_argument("--max_z", type=float, default=6.0)
    ap.add_argument("--save_vis", default=True)
    ap.add_argument("--out_dir", default="/HDD/etc/outputs/intelrealsense/compare")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    dep_ret = np.load(args.ret_depth).astype(np.float32)  # [m]
    dep_ret = dep_ret * args.ret_scale + args.ret_offset

    dep_rs  = np.load(args.rs_depth).astype(np.float32)   # [m]

    # 좌표/크기 맞추기
    if args.meta_json and not args.rs_is_rectified:
        (mapLx, mapLy), (W,H) = build_rectify_maps_from_meta(args.meta_json)
        if dep_rs.shape[:2] != (H,W):
            dep_rs = cv.resize(dep_rs, (W,H), interpolation=cv.INTER_NEAREST)
        dep_rs_rect = cv.remap(dep_rs, mapLx, mapLy, interpolation=cv.INTER_NEAREST, borderMode=cv.BORDER_CONSTANT, borderValue=0)
        dep_rs = dep_rs_rect

    # 크기 다르면 retinify를 rs에 맞춤(최근접)
    if dep_ret.shape != dep_rs.shape:
        dep_ret = cv.resize(dep_ret, (dep_rs.shape[1], dep_rs.shape[0]), interpolation=cv.INTER_NEAREST)

    # 유효 마스크
    mask = np.isfinite(dep_ret) & np.isfinite(dep_rs) \
         & (dep_ret > 0) & (dep_rs > 0) \
         & (dep_ret >= args.min_z) & (dep_ret <= args.max_z) \
         & (dep_rs  >= args.min_z) & (dep_rs  <= args.max_z)

    # 메트릭
    m = metrics(dep_rs, dep_ret, mask)  # gt=RealSense, pr=retinify (원하면 반대로도)
    print(f"[INFO] valid pixels: {m['count']}")
    for k,v in m.items():
        if k!="count":
            print(f"{k:>12}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}")

    # 시각화
    if args.save_vis:
        err = np.zeros_like(dep_rs, dtype=np.float32)
        err[mask] = np.abs(dep_rs[mask] - dep_ret[mask])
        vmax = np.percentile(err[mask], 95) if m['count']>0 else 0.5
        vis_err = (np.clip(err / max(vmax,1e-6), 0, 1) * 255).astype(np.uint8)
        vis_err = cv.applyColorMap(vis_err, cv.COLORMAP_JET)

        cv.imwrite(os.path.join(args.out_dir, "ret_depth.png"), viz_depth(dep_ret, args.min_z, args.max_z))
        cv.imwrite(os.path.join(args.out_dir, "rs_depth.png"),  viz_depth(dep_rs,  args.min_z, args.max_z))
        cv.imwrite(os.path.join(args.out_dir, "error_abs.png"), vis_err)

        # 마스크 표시
        cv.imwrite(os.path.join(args.out_dir, "valid_mask.png"), (mask.astype(np.uint8)*255))
        # 에러 히스토그램 (간단 저장)
        hist, bins = np.histogram(err[mask], bins=100, range=(0, max(vmax,1e-6)))
        np.save(os.path.join(args.out_dir, "error_hist.npy"), hist)
        np.save(os.path.join(args.out_dir, "error_bins.npy"), bins)

if __name__ == "__main__":
    main()
