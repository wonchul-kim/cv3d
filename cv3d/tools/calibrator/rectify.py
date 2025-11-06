import os, json, glob, os.path as osp
import numpy as np
import cv2 as cv
from tqdm import tqdm

# ====== 경로 설정 ======
meta_json = "/HDD/etc/outputs/intelrealsense/meta/calibration_211122062694.json"  # 올려준 JSON 저장 파일 경로
input_dir  = "/HDD/etc/outputs/intelrealsense/inputs"
output_dir = "/HDD/etc/outputs/intelrealsense/rectified"
os.makedirs(output_dir, exist_ok=True)

# ====== JSON 로드 ======
with open(meta_json, "r") as f:
    meta = json.load(f)

ir1 = meta["streams"]["stream.infrared_1"]
ir2 = meta["streams"]["stream.infrared_2"]

# --- Intrinsics (IR1/IR2) ---
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

# ====== 입력 영상 목록 (IR1=left*, IR2=right*) ======
left_candidates  = sorted(glob.glob(os.path.join(input_dir, "left*.png")))
right_candidates = sorted(glob.glob(os.path.join(input_dir, "right*.png")))
print(f"There are {len(left_candidates)} left and {len(right_candidates)} right images")
assert len(left_candidates) == len(right_candidates) and len(left_candidates) > 0

# 한 장 읽어 사이즈 확인 (JSON 해상도와 동일해야 가장 깔끔)
imgL0 = cv.imread(left_candidates[0], cv.IMREAD_GRAYSCALE)
imgR0 = cv.imread(right_candidates[0], cv.IMREAD_GRAYSCALE)
h, w = imgL0.shape[:2]
assert imgR0.shape[:2] == (h, w)
frame_size = (w, h)

# 만약 이미지 크기가 JSON intrinsics의 (width,height)와 다르면,
# K의 cx,cy,fx,fy를 비율로 스케일해야 합니다. (여기선 640x480로 동일하다고 가정)

# ====== OpenCV stereoRectify ======
flags = cv.CALIB_ZERO_DISPARITY   # 왼쪽 기준으로 주시점 정렬
alpha = 0                         # 0=크롭, 1=풀뷰. 필요 시 조절
R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
    K1, D1, K2, D2, frame_size, R_1to2, T_1to2, flags=flags, alpha=alpha
)

# ====== Rectify 맵 생성 ======
mapLx, mapLy = cv.initUndistortRectifyMap(K1, D1, R1, P1, frame_size, cv.CV_32FC1)
mapRx, mapRy = cv.initUndistortRectifyMap(K2, D2, R2, P2, frame_size, cv.CV_32FC1)

# ====== 배치 처리 ======
for lf, rf in tqdm(list(zip(left_candidates, right_candidates))):
    lbase = osp.splitext(osp.basename(lf))[0]
    rbase = osp.splitext(osp.basename(rf))[0]

    imgL = cv.imread(lf, cv.IMREAD_GRAYSCALE)
    imgR = cv.imread(rf, cv.IMREAD_GRAYSCALE)

    rectL = cv.remap(imgL, mapLx, mapLy, cv.INTER_LINEAR)
    rectR = cv.remap(imgR, mapRx, mapRy, cv.INTER_LINEAR)

    cv.imwrite(os.path.join(output_dir, f"rect_{lbase}.png"), rectL)
    cv.imwrite(os.path.join(output_dir, f"rect_{rbase}.png"), rectR)

    # 스캔라인 확인용 시각화
    vis = np.hstack((cv.cvtColor(rectL, cv.COLOR_GRAY2BGR),
                     cv.cvtColor(rectR, cv.COLOR_GRAY2BGR)))
    for y in range(100, rectL.shape[0], 100):
        cv.line(vis, (0, y), (vis.shape[1]-1, y), (0, 255, 0), 1)
    cv.imwrite(os.path.join(output_dir, f"rect_with_lines_{lbase}_{rbase}.png"), vis)

print("Done. Q matrix saved? 필요하면 np.savez로 K1,D1,K2,D2,R1,R2,P1,P2,Q를 저장하세요.")
