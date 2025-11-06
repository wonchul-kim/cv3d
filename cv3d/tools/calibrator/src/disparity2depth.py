import cv2 as cv
import numpy as np
import os.path as osp


def depth_to_points_xyz(depth_m: np.ndarray, K: np.ndarray, delete_valid=False) -> np.ndarray:
    """
    depth_m: (H,W) float32/float64 depth in meters (NaN/0 = invalid)
    K: 3x3 intrinsic matrix, [[fx,0,cx],[0,fy,cy],[0,0,1]]
    return: (H*W, 3) XYZ in meters (invalid 제외 후)
    """
    H, W = depth_m.shape
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # 픽셀 그리드
    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(xs, ys)  # (H,W)

    Z = depth_m
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    # (H,W,3) -> (N,3)
    pts = np.dstack((X, Y, Z)).reshape(-1, 3)
    # 무효 제거
    # valid = np.isfinite(Z).reshape(-1) & (Z > 0)
    if delete_valid:
        valid = np.isfinite(Z).reshape(-1) & (Z.reshape(-1) > 0)

        return pts[valid]

    return pts

def write_ply_xyzrgb(filename: str, points: np.ndarray, colors: np.ndarray = None):
    points = points.reshape(-1, 3)
    if colors is not None:
        colors = colors.reshape(-1, 3)
        assert points.shape[0] == colors.shape[0]
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for (x,y,z), (r,g,b) in zip(points, colors):
                f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
    else:
        with open(filename, 'w') as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\nproperty float y\nproperty float z\nend_header\n")
            for (x,y,z) in points:
                f.write(f"{x} {y} {z}\n")


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

target = 0
# output_dir = '/HDD/etc/outputs/retinify/rectification/1/outputs'
output_dir = '/HDD/etc/outputs/intelrealsense/retinify'
# stereo_calib = np.load('/HDD/etc/outputs/retinify/cal_results/2/stereo_calib.npz')
stereo_calib = np.load('/HDD/etc/outputs/intelrealsense/rectified/stereo_calib.npz')
K1 = stereo_calib['K1']
'''
array([[598.13397217,   0.        , 317.66104126],
       [  0.        , 598.13397217, 248.04055786],
       [  0.        ,   0.        ,   1.        ]])
array([[2.36780382e+03, 0.00000000e+00, 1.25090950e+03],
       [0.00000000e+00, 2.36685016e+03, 9.82725816e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
'''

T = stereo_calib['T']
'''
array([[-0.05499389],
       [ 0.00030266],
       [ 0.0012008 ]])
array([[ 5.67419399e+01],
[ 5.20661472e+00],
[-5.01766976e-02]])
'''

# 1) disparity 로드 (Retinify가 float32 px로 준다고 가정)
# disp = cv.imread("retinify_disparity.exr", cv.IMREAD_UNCHANGED)  # EXR/PNG/… 어떤 포맷이든 로드
# disp = np.load("/HDD/etc/retinify/retinify-opencv-example/disparity.npy")
# disparity_file = f'/HDD/etc/outputs/retinify/rectification/1/outputs/disparity{target}.bin'
disparity_file = f'/HDD/etc/outputs/intelrealsense/retinify/disparity.bin'
# src_img_filename = f'/HDD/etc/outputs/retinify/rectification/1/outputs/rect_left{target}.png'
# src_img_filename = f'/HDD/etc/outputs/retinify/rectification/1/outputs/rect_right{target}.png'
src_img_filename = f'/HDD/etc/outputs/intelrealsense/inputs/left.png'
# left_img_filename = None
# .bin 파일 읽기
with open(disparity_file, 'rb') as f:
    # C++에서 저장한 메타 정보 (rows, cols, type) 읽기
    rows = np.frombuffer(f.read(4), dtype=np.int32)[0]
    cols = np.frombuffer(f.read(4), dtype=np.int32)[0]
    # OpenCV Mat 타입 (CV_32FC1 = 5)
    _ = np.frombuffer(f.read(4), dtype=np.int32)[0] 
    
    # 실제 데이터 읽기 및 NumPy 배열로 변환
    disparity_data = np.frombuffer(f.read(), dtype=np.float32)
    disparity_map = disparity_data.reshape((rows, cols))

# 2) 보정값 (캘리브레이션 결과에서)
#  - fx = K1[0,0]
#  - baseline: stereoCalibrate(T)의 노름. 보드 단위를 mm로 썼다면 반드시 m로 변환.
# K1 = np.array([[2396.97, 0, 1244.18],
#                [0, 2396.98, 1011.86],
#                [0, 0, 1]], dtype=np.float32)
fx = float(K1[0,0])
# fx = 598.1339721679688

baseline_m = np.linalg.norm(T)
# baseline_m = baseline_mm / 1000.0
# 0.05698033958511742

depth_m = disparity_to_depth(disparity_map, fx, baseline_m, disp_scale=1.0, min_disp=0.1)

# 4) 시각화(선택): 0~5m 클램프 후 컬러맵
np.save(osp.join(output_dir, 'depth.npy'), depth_m)
depth_vis = depth_m.copy()
# depth_vis = np.clip(depth_vis, 0, 5.0)   # 5m까지만 보기
depth_vis = (depth_vis / 5.0 * 255).astype(np.uint8)
depth_vis = cv.applyColorMap(255 - depth_vis, cv.COLORMAP_JET)  # 가까울수록 빨강
cv.imwrite(osp.join(output_dir, "depth_vis{target}.png"), depth_vis)

# # -------- 예시: depth + K ----------
# # 1) depth 로드 (미터 단위, float32)
# depth_m = cv.imread("depth_m.exr", cv.IMREAD_UNCHANGED).astype(np.float32)  # 예시: EXR
# # NaN/0 값은 무효 취급

# 2) 카메라 내정수행렬 K (좌카메라)
K = K1

# 3) 포인트 생성
points_wo_valid = depth_to_points_xyz(depth_m, K, delete_valid=False)
points_w_valid = depth_to_points_xyz(depth_m, K, delete_valid=True)

# 4) 색 (선택)
src_img = cv.imread(src_img_filename)
if src_img is not None and src_img.shape[:2] == depth_m.shape[:2]:
    colors = src_img.reshape(-1,3)[np.isfinite(depth_m).reshape(-1) & (depth_m.reshape(-1) > 0)]
    colors = colors[:, ::-1]  # BGR->RGB
else:
    colors = None

# 5) 저장
write_ply_xyzrgb(osp.join(output_dir, f"points_w_valid_color{target}.ply"), points_w_valid, colors)
write_ply_xyzrgb(osp.join(output_dir, f"points_w_valid{target}.ply"), points_w_valid, None)
write_ply_xyzrgb(osp.join(output_dir, f"points_wo_valid{target}.ply"), points_wo_valid, None)
print("PLY 저장 완료 ==========================")