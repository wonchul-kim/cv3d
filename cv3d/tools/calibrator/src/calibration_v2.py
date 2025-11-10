import cv2 as cv
import numpy as np
import glob, os
from tqdm import tqdm
import os.path as osp
from pathlib import Path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]


input_dir = str(ROOT / 'data/realsense_calibration_images')
output_dir = '/HDD/etc/outputs/cv3d/calibrator'
os.makedirs(output_dir, exist_ok=True)

calib_info = np.load('/HDD/etc/outputs/retinify/cal_results/2/stereo_calib.npz')
K1, D1, K2, D2 = calib_info['K1'], calib_info['D1'], calib_info['K2'], calib_info['D2']
R1, P1, R2, P2 = calib_info['R1'], calib_info['P1'], calib_info['R2'], calib_info['P2']
R, T, Q = calib_info['R'], calib_info['T'], calib_info['Q']

left_candidates  = sorted(glob.glob(os.path.join(input_dir, 'left*.png')))
right_candidates = sorted(glob.glob(os.path.join(input_dir, 'right*.png')))
print(f"There are {len(left_candidates)} left images and {len(right_candidates)} images")

assert len(left_candidates) == len(right_candidates)

for left_img_file, right_img_file in tqdm(zip(left_candidates, right_candidates)):
    
    left_filename = osp.split(osp.splitext(left_img_file)[0])[-1]
    right_filename = osp.split(osp.splitext(right_img_file)[0])[-1]

    imgL0 = cv.imread(left_img_file)
    imgR0 = cv.imread(right_img_file)

    frame_size = (imgL0.shape[1], imgR0.shape[0])  # (w,h)
    
    mapLx, mapLy = cv.initUndistortRectifyMap(K1, D1, R1, P1, frame_size, cv.CV_32FC1)
    mapRx, mapRy = cv.initUndistortRectifyMap(K2, D2, R2, P2, frame_size, cv.CV_32FC1)

    rectL = cv.remap(imgL0, mapLx, mapLy, cv.INTER_LINEAR)
    rectR = cv.remap(imgR0, mapRx, mapRy, cv.INTER_LINEAR)
    cv.imwrite(os.path.join(output_dir, f"rect_{left_filename}.png"), rectL)
    cv.imwrite(os.path.join(output_dir, f"rect_{right_filename}.png"), rectR)

    # 가로 스캔라인이 일치하는지 보조선 그려보기
    vis = np.hstack((rectL.copy(), rectR.copy()))
    for y in range(100, rectL.shape[0], 100):
        cv.line(vis, (0,y), (vis.shape[1]-1, y), (0,255,0), 1)
    cv.imwrite(os.path.join(output_dir, f"rect_with_lines_{left_filename}_{right_filename}.png"), vis)


