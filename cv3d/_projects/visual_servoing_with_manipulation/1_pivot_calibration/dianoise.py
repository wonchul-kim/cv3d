#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, math, argparse
import numpy as np
import cv2

VAL_T_MM, VAL_R_DEG = 5.0, 2.0

def invert_T(T):
    R,t = T[:3,:3], T[:3,3]
    Ti = np.eye(4); Ti[:3,:3] = R.T; Ti[:3,3] = -R.T @ t
    return Ti

def rot_deg(R):
    c = (np.trace(R)-1.0)/2.0
    c = max(-1.0,min(1.0,c))
    return math.degrees(math.acos(c))

def motions_from_abs(T_bg_list, T_tc_list):
    Rg,tg,Rt,tt = [],[],[],[]
    for i in range(len(T_bg_list)-1):
        A = invert_T(T_bg_list[i]) @ T_bg_list[i+1]           # gripper motion
        B = T_tc_list[i] @ invert_T(T_tc_list[i+1])           # target motion (target2cam)
        Rg.append(A[:3,:3]); tg.append(A[:3,3])
        Rt.append(B[:3,:3]); tt.append(B[:3,3])
    return np.array(Rg), np.array(tg), np.array(Rt), np.array(tt)

def residuals(T_tc, Rg,tg,Rt,tt):
    axd, axm = [], []
    for i in range(len(Rg)):
        A = np.eye(4); A[:3,:3]=Rg[i]; A[:3,3]=tg[i]
        B = np.eye(4); B[:3,:3]=Rt[i]; B[:3,3]=tt[i]
        E = invert_T(T_tc @ B) @ (A @ T_tc)
        axd.append(rot_deg(E[:3,:3]))
        axm.append(np.linalg.norm(E[:3,3])*1000.0)
    return np.array(axd), np.array(axm)

def world_consistency(T_tc, T_bg_list, T_tc_list):
    Tbb = []
    for Tbg, Tt2c in zip(T_bg_list, T_tc_list):
        T_cam_target = invert_T(Tt2c)
        T_base_cam   = Tbg @ T_tc
        T_base_board = T_base_cam @ T_cam_target
        Tbb.append(T_base_board)
    ref = Tbb[0]
    wd, wm = [], []
    for T in Tbb:
        E = invert_T(ref) @ T
        wd.append(rot_deg(E[:3,:3])); wm.append(np.linalg.norm(E[:3,3])*1000.0)
    return np.array(wd), np.array(wm)

def solve_once(T_bg_list, T_tc_list, method=cv2.CALIB_HAND_EYE_TSAI):
    # absolute poses → calibrateHandEye inputs
    Rg2b,tg2b,Rt2c,tt2c = [],[],[],[]
    for Tbg,Tt2c in zip(T_bg_list, T_tc_list):
        Rb, tb = Tbg[:3,:3], Tbg[:3,3]
        Rg2b.append(Rb.T); tg2b.append(-Rb.T @ tb)
        Rt2c.append(Tt2c[:3,:3]); tt2c.append(Tt2c[:3,3])
    Rg2b,tg2b,Rt2c,tt2c = map(lambda x: np.array(x), (Rg2b,tg2b,Rt2c,tt2c))
    Rtc, ttc = cv2.calibrateHandEye(Rg2b,tg2b,Rt2c,tt2c, method=method)
    Ttc = np.eye(4); Ttc[:3,:3]=Rtc; Ttc[:3,3]=ttc.reshape(3)
    Rg,tg,Rt,tt = motions_from_abs(T_bg_list, T_tc_list)
    axd, axm = residuals(Ttc, Rg,tg,Rt,tt)
    wd, wm   = world_consistency(Ttc, T_bg_list, T_tc_list)
    score = wm.mean()+5*wd.mean()+axm.mean()+5*axd.mean()
    return Ttc, axd, axm, wd, wm, float(score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default='/wonchul/outputs/calibration', help="데이터셋 저장 디렉토리")
    ap.add_argument("--pivot_tx", type=float, default=0.000793)
    ap.add_argument("--pivot_ty", type=float, default=0.000745)
    ap.add_argument("--pivot_tz", type=float, default=0.171436)
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.out_dir,"meta.json")))
    S = [json.loads(l) for l in open(os.path.join(args.out_dir,"samples.jsonl")) if l.strip()]

    # load as saved (baseline)
    T_bg = [np.array(s["T_base_tcp"]).reshape(4,4) for s in S]
    T_tc = [np.array(s["T_target2cam"]).reshape(4,4) for s in S]

    # variants to test
    variants = []

    # V0: as-is (현재 결과)
    variants.append(("as_is", T_bg))

    # V1: pivot 제거 (RobotState가 이미 TCP일 가능성)
    Tfix_inv = np.eye(4); Tfix_inv[:3,3] = -np.array([args.pivot_tx,args.pivot_ty,args.pivot_tz])
    variants.append(("remove_pivot", [T @ Tfix_inv for T in T_bg]))

    # V2: pivot 반대로 적용되어 있었다고 가정 (안 맞지만 안전 확인)
    Tfix = np.eye(4); Tfix[:3,3] = np.array([args.pivot_tx,args.pivot_ty,args.pivot_tz])
    variants.append(("apply_pivot_again", [T @ Tfix for T in T_bg]))

    # V3: pivot 벡터가 flange 프레임이 아니라 TCP 프레임 기준이었다고 가정
    # -> T @ [R_b_g * (0,tcp_frame_vec)] 근사 시도 (샘플 의존이라 완벽하진 않지만 경향 확인용)
    Tlist=[]
    for T in T_bg:
        Rb = T[:3,:3]
        t_tcp_in_base = Rb @ np.array([args.pivot_tx,args.pivot_ty,args.pivot_tz])
        Ttmp = T.copy(); Ttmp[:3,3] = T[:3,3] + t_tcp_in_base
        Tlist.append(Ttmp)
    variants.append(("pivot_in_tcp_frame", Tlist))

    # V4: 쿼터니언 순서 뒤집힘(wxyz라고 가정) → 재구축 (가능성 점검)
    # (데이터셋에는 이미 행렬만 있으니, 캡처 단계에서만 의미. 여기선 스킵)

    print("\n[diagnose] trying variants...")
    best=None
    for name, Tbg_list in variants:
        Ttc, axd, axm, wd, wm, score = solve_once(Tbg_list, T_tc)
        print(f"{name:>18} | AX mean {axd.mean():6.2f}deg / {axm.mean():6.1f}mm | "
              f"W mean {wd.mean():6.2f}deg / {wm.mean():6.1f}mm | score {score:8.1f}")
        if (best is None) or (score<best[0]):
            best=(score,name,Ttc,axd,axm,wd,wm)
    print(f"\n[diagnose] BEST = {best[1]}")
    print("T_tool_camera:\n", best[2])

if __name__ == "__main__":
    main()
