import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation as R
from common import (
    quat_xyzw_to_T, invert_T, T_to_rvec_tvec,
    T_from_flat, pose_diff_T, T
)

output_dir = '/HDD/etc/outputs/calibration_single_tag/calibration'
samples = []

try:
    with open(f"{output_dir}/samples.jsonl", 'r') as f:
        for line in f:
            samples.append(json.loads(line))
except FileNotFoundError:
    print("[ERROR] 'samples.jsonl' 파일을 찾을 수 없습니다.")
    exit()

print(f"Loaded {len(samples)} samples.")

T_flange2tcp = None
try:
    with open(f"{output_dir}/meta.json", 'r') as f:
        meta = json.load(f)
except FileNotFoundError:
    print("[ERROR] 'meta.json' 파일을 찾을 수 없습니다.")
    exit()

if "pivot_flange_to_tcp_m" in meta:
    T_flange2tcp = T(np.eye(3), meta["pivot_flange_to_tcp_m"])
    print(f"Loaded meta and T_flange2tcp calculated: ")
    print(f"{T_flange2tcp}")
else:
    print("Warning: 'pivot_flange_to_tcp_m' not found in meta.json. Using Identity.")
    T_flange2tcp = np.eye(4, dtype=np.float64)

    
R_target2cam_list, t_target2cam_list = [], []
R_tcp2base_list, t_tcp2base_list = [], []
for sample in samples:
    T_flange2base = quat_xyzw_to_T(sample['pos_m'], sample['quat_xyzw'])
    # T_tcp2base = invert_T(T_flange2tcp)@T_flange2base
    # T_tcp2base = T_flange2base @ invert_T(T_flange2tcp)   
    T_tcp2base = T_flange2base

    R_tcp2base, t_tcp2base = T_to_rvec_tvec(T_tcp2base)
    R_tcp2base_list.append(R_tcp2base)
    t_tcp2base_list.append(t_tcp2base)

    T_target2cam = T_from_flat(sample['T_tag_cam'])
    R_target2cam, t_target2cam = T_to_rvec_tvec(T_target2cam)
    R_target2cam_list.append(R_target2cam)
    t_target2cam_list.append(t_target2cam)

print("\n--- Performing Hand-Eye Calibration (AX=XB) ---")

R_cam2tcp, t_cam2tcp = cv2.calibrateHandEye(
    R_tcp2base_list, t_tcp2base_list,
    R_target2cam_list, t_target2cam_list,
    method=cv2.CALIB_HAND_EYE_TSAI
)

T_cam2tcp = np.eye(4, dtype=np.float64)
T_cam2tcp[:3, :3] = R_cam2tcp.reshape(3, 3)
T_cam2tcp[:3, 3] = t_cam2tcp.reshape(3)
'''
[[ 0.70350934 -0.60118006 -0.37902129  0.06244145]
 [ 0.71048044  0.60776357  0.35474073 -0.07640022]
 [ 0.01709228 -0.51885063  0.85469403  0.23122382]
 [ 0.          0.          0.          1.        ]]
'''

r_final = R.from_matrix(R_cam2tcp)
euler_zyx_deg = r_final.as_euler('zyx', degrees=True)

print("\n--- Hand-Eye Calibration Result (T_cam2tcp) ---")
print("[Homogeneous Matrix T_cam2tcp (meter)]")
print(T_cam2tcp)
print("\n[Rotation (Euler ZYX, degrees)]")
print(f"  Yaw (Z):   {euler_zyx_deg[0]:.2f}°")
print(f"  Pitch (Y): {euler_zyx_deg[1]:.2f}°")
print(f"  Roll (X):  {euler_zyx_deg[2]:.2f}°")
print("\n[Translation (mm)]")
print(f"  X: {T_cam2tcp[0, 3] * 1000:.2f} mm")
print(f"  Y: {T_cam2tcp[1, 3] * 1000:.2f} mm")
print(f"  Z: {T_cam2tcp[2, 3] * 1000:.2f} mm")


# -------------------- 검증 --------------------
T_target2base_list = []
for sample in samples:
    T_flange2base = quat_xyzw_to_T(sample['pos_m'], sample['quat_xyzw'])
    # T_tcp2base = invert_T(T_flange2tcp)@T_flange2base
    # T_tcp2base = T_flange2base @ invert_T(T_flange2tcp) 
    T_tcp2base = T_flange2base
    T_target2cam = T_from_flat(sample['T_tag_cam'])

    # target → base
    T_target2base = T_tcp2base @ T_cam2tcp @ T_target2cam
    T_target2base_list.append(T_target2base)
    T_base2target = invert_T(T_target2base)
    R_base2target = T_base2target[:3, :3]
    
# 기준 프레임 대비 오차 계산
rot_errors, trans_errors = [], []
T_ref = T_target2base_list[0]
for T_tag_base in T_target2base_list:
    rot_err, trans_err = pose_diff_T(T_ref, T_tag_base)
    rot_errors.append(rot_err)
    trans_errors.append(trans_err)

mean_rot_error = np.mean(rot_errors)
mean_trans_error_m = np.mean(trans_errors)

print("\n--- Validation (Reprojection Consistency) ---")
print(f"  Mean Rotation Error (deg): {mean_rot_error:.3f} °")
print(f"  Mean Translation Error: {mean_trans_error_m*1000:.2f} mm")
