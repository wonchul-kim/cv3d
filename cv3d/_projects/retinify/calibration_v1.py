import cv2 as cv
import numpy as np
import glob, os, re
import os.path as osp
from pathlib import Path
from tqdm import tqdm
import os.path as osp

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]



CHECKERBOARD = (18, 12)  # (cols, rows) 내부 코너 수
SQUARE_SIZE_MM = 30.0    # 한 칸 길이 [mm]  -> T(번역)도 mm 단위가 됩니다.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-4)

input_dir = str(ROOT / 'data/realsense_calibration_images')
output_dir = '/HDD/etc/outputs/cv3d/calibrator/cal'
os.makedirs(output_dir, exist_ok=True)

import sys
import logging

log_file = Path(osp.join(output_dir, f'cal_log.txt'))
logging.basicConfig(
    level=logging.INFO, # INFO 레벨 이상의 메시지를 기록
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, 'w', 'utf-8'), # log.txt 파일에 기록
        logging.StreamHandler(sys.stdout) # 콘솔에도 동시에 출력
    ]
)
logger = logging.getLogger(__name__)

left_candidates  = sorted(glob.glob(os.path.join(input_dir, '*left*.png')))
right_candidates = sorted(glob.glob(os.path.join(input_dir, '*right*.png')))

def make_key(p: str):
    s = Path(p).stem.lower()
    # 흔한 패턴들 정리: left/right 제거 후 숫자 토큰 추출
    s_clean = re.sub(r'left|right', '', s)
    nums = re.findall(r'\d+', s_clean)
    return (nums[-1] if nums else s_clean)  # 숫자 우선, 없으면 전체 스템

right_map = {}
for rp in right_candidates:
    k = make_key(rp)
    right_map[k] = rp

pairs = []
for lp in left_candidates:
    k = make_key(lp)
    rp = right_map.get(k, None)
    if rp is None:
        # 대체: 단순 치환도 시도
        guess = lp.lower().replace('left', 'right')
        if os.path.exists(guess):
            rp = guess
    if rp and os.path.exists(rp):
        pairs.append((lp, rp))

if len(pairs) == 0:
    raise RuntimeError("좌/우 페어를 하나도 만들지 못했습니다. 파일명 패턴을 확인하세요.")

logger.info(f"[INFO] 매칭된 좌/우 페어 수: {len(pairs)}")

# ---------- 1) 보드 3D 포인트 ----------
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE_MM  # mm 단위

# ---------- 2) 코너 검출 함수 (고전 + SB fallback + 멀티스케일) ----------
def detect_corners(img_bgr):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_FAST_CHECK
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, flags)

    if not ret:
        sb_flags = cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
        ret, corners = cv.findChessboardCornersSB(gray, CHECKERBOARD, flags=sb_flags)

    if not ret:
        # 멀티스케일 시도
        for s in [0.75, 1.25, 1.5]:
            h, w = gray.shape[:2]
            resized = cv.resize(gray, (int(w*s), int(h*s)), interpolation=cv.INTER_LINEAR)
            ret, corners_res = cv.findChessboardCornersSB(resized, CHECKERBOARD,
                                                          flags=cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY)
            if ret:
                corners = corners_res / s
                break

    if ret:
        corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)

    return ret, corners

# ---------- 3) 코너 수집 + 페어 검증(RANSAC) ----------
objpoints, imgpointsL, imgpointsR = [], [], []
frame_size = None
accepted = 0
for i, (lp, rp) in tqdm(enumerate(pairs)):
    imgL = cv.imread(lp)
    imgR = cv.imread(rp)
    if imgL is None or imgR is None:
        continue
    if frame_size is None:
        frame_size = (imgL.shape[1], imgL.shape[0])  # (w,h)

    retL, cornersL = detect_corners(imgL)
    retR, cornersR = detect_corners(imgR)
    if not (retL and retR):
        continue

    # 페어 유효성 체크: 해당 프레임에서 좌/우 코너 간 F행렬 RANSAC 인라이어 비율
    ptsL = cornersL.reshape(-1,2)
    ptsR = cornersR.reshape(-1,2)
    F, mask = cv.findFundamentalMat(ptsL, ptsR, cv.FM_RANSAC, 1.0, 0.99)
    if F is None:
        continue
    inlier_ratio = mask.mean()
    if inlier_ratio < 0.6:  # 비율 낮으면 잘못된 페어로 간주
        continue

    objpoints.append(objp)
    imgpointsL.append(cornersL)
    imgpointsR.append(cornersR)
    accepted += 1

logger.info(f"[INFO] 코너가 검출되고 RANSAC 통과한 페어 수: {accepted}")
if accepted < 10:
    logger.info("[WARN] 유효 페어가 적습니다. 20~30 이상 권장.")

# ---------- 4) 단안 보정 ----------
retL, K1, D1, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frame_size, None, None)
retR, K2, D2, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frame_size, None, None)
logger.info("\n--- Monocular Calibration ---")
logger.info(f"L RMS: {retL} | R RMS: {retR}")
logger.info(f"K1:\n {K1}")
logger.info(f"K2:\n {K2}")



# ---------- 5) 스테레오 보정 (내부 고정!) ----------
flags = (cv.CALIB_FIX_INTRINSIC)  # 내부 파라미터 고정이 핵심!
# 동일 렌즈/센서면 아래 플래그도 유용할 수 있음(상황에 따라):
# flags |= cv.CALIB_SAME_FOCAL_LENGTH
# flags |= cv.CALIB_ZERO_TANGENT_DIST

retStereo, K1f, D1f, K2f, D2f, R, T, E, F = cv.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR,
    K1, D1, K2, D2, frame_size,
    criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    flags=flags
)

logger.info("\n--- Stereo Calibration (FIX_INTRINSIC) ---")
logger.info(f"Stereo RMS: {retStereo}")
logger.info(f"K1(final) should equal K1:\n {K1f}")
logger.info(f"K2(final) should equal K2:\n {K2f}")
logger.info(f"R:\n {R}")
logger.info(f"T [mm]:\n {T}")
baseline = np.linalg.norm(T)
angle = np.degrees(np.arccos(np.clip((np.trace(R)-1)/2, -1.0, 1.0)))
logger.info(f"Baseline: {baseline:.2f} mm  |  Rotation angle: {angle:.3f} deg")


# ---------- 6) Rectify + 저장 ----------
R1, R2, P1, P2, Q, roi1, roi2 = cv.stereoRectify(K1, D1, K2, D2, frame_size, R, T, alpha=0)
mapLx, mapLy = cv.initUndistortRectifyMap(K1, D1, R1, P1, frame_size, cv.CV_32FC1)
mapRx, mapRy = cv.initUndistortRectifyMap(K2, D2, R2, P2, frame_size, cv.CV_32FC1)

np.savez(os.path.join(output_dir, "stereo_calib.npz"), K1=K1, D1=D1, K2=K2, D2=D2, 
                             R=R, T=T, 
                             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q)
logger.info(f'Saved calibration: {os.path.join(output_dir, "stereo_calib.npz")}')

# 샘플 한 장 저장 (시각 확인)
imgL0 = cv.imread(pairs[0][0])
imgR0 = cv.imread(pairs[0][1])
rectL = cv.remap(imgL0, mapLx, mapLy, cv.INTER_LINEAR)
rectR = cv.remap(imgR0, mapRx, mapRy, cv.INTER_LINEAR)
cv.imwrite(os.path.join(output_dir, "rect_left.png"), rectL)
cv.imwrite(os.path.join(output_dir, "rect_right.png"), rectR)

# 가로 스캔라인이 일치하는지 보조선 그려보기
vis = np.hstack((rectL.copy(), rectR.copy()))
for y in range(100, rectL.shape[0], 100):
    cv.line(vis, (0,y), (vis.shape[1]-1, y), (0,255,0), 1)
cv.imwrite(os.path.join(output_dir, "rect_with_lines.png"), vis)
