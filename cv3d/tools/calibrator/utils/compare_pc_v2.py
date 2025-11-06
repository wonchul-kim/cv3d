import numpy as np
import open3d as o3d
from pathlib import Path
import logging
import sys
import json

target1 = 1
target2 = 3

# 로거 설정
log_file = Path(f'/HDD/etc/outputs/retinify/rectification/1/outputs/log{target1}{target2}.txt')
logging.basicConfig(
    level=logging.INFO, # INFO 레벨 이상의 메시지를 기록
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, 'w', 'utf-8'), # log.txt 파일에 기록
        logging.StreamHandler(sys.stdout) # 콘솔에도 동시에 출력
    ]
)
logger = logging.getLogger(__name__)

filename1 = f'/HDD/etc/outputs/retinify/rectification/1/outputs/points_w_valid{target1}.ply'
filename2 = f'/HDD/etc/outputs/retinify/rectification/1/outputs/points_w_valid{target2}.ply'

state1 = f'/HDD/etc/outputs/retinify/rectification/1/state_{target1}.json'
state2 = f'/HDD/etc/outputs/retinify/rectification/1/state_{target2}.json'
with open(state1, 'r') as f:
    state1 = json.load(f)
with open(state2, 'r') as f:
    state2 = json.load(f)


expected_shift_m = abs(state1['tcp_pos']['position']['y'] - state2['tcp_pos']['position']['y'])/1000
tolerance_m = 0.02

def read_cloud(path):
    assert Path(path).exists(), f"파일 없음: {path}"
    pcd = o3d.io.read_point_cloud(path)
    n = np.asarray(pcd.points).shape[0]
    logger.info(f"[LOAD] {path} -> {n} pts")
    if n == 0:
        raise RuntimeError(f"포인트가 0개입니다: {path}")
    return pcd

def auto_voxel_size(pcd, target_pts=120_000):
    """점 개수와 박스 크기로 대략적 voxel 추정 (너무 크면 줄이고, 너무 작으면 늘림)"""
    n = np.asarray(pcd.points).shape[0]
    extent = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())
    diag = float(np.linalg.norm(extent)) + 1e-9
    # 장면 대각선의 1/300 ~ 1/50 범위에서 시작
    base = np.clip(diag / 300.0, 0.0005, diag / 50.0)
    # 점이 너무 많으면 조금 크게, 너무 적으면 작게
    scale = np.sqrt(max(n, 1) / target_pts)
    vox = base * scale
    return float(np.clip(vox, 0.0005, 0.02))  # 0.5~20mm 사이로 클램프

def safe_downsample(pcd, voxel):
    """다운샘플 백오프: 실패 시 점점 더 작은 voxel로 재시도, 끝까지 안되면 원본 유지"""
    orig_n = np.asarray(pcd.points).shape[0]
    for v in [voxel, voxel*0.7, voxel*0.5, voxel*0.3, voxel*0.2, 0.0]:
        if v <= 0:
            logger.info(f"[DS] voxel=0 → 다운샘플 스킵 (원본 유지)")
            return pcd, 0.0
        ds = pcd.voxel_down_sample(v)
        n = np.asarray(ds.points).shape[0]
        logger.info(f"[DS] voxel={v:.5f} -> {n} pts")
        # 다운샘플 후 너무 적으면 다음 후보로
        if n == 0 or n < min(1000, orig_n * 0.01):
            continue
        return ds, v
    # 모든 시도 실패 → 원본 반환
    logger.info("[DS] 모든 다운샘플 시도 실패 → 원본 유지")
    return pcd, 0.0

def safe_denoise(pcd):
    """통계적 외란 제거 백오프: 과하면 롤백"""
    n0 = np.asarray(pcd.points).shape[0]
    if n0 < 5000:
        logger.info(f"[DN] pts={n0} -> 포인트 적음, denoise 스킵")
        return pcd
    for nb, sr in [(20,1.5), (30,2.0), (50,2.0)]:
        dn, ind = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=sr)
        n = np.asarray(dn.points).shape[0]
        logger.info(f"[DN] nb={nb}, std={sr} -> {n} pts")
        # 남은 점이 너무 적으면 롤백
        if n < max(2000, n0*0.2):
            continue
        return dn
    logger.info("[DN] 과도한 제거 감지 → denoise 롤백")
    return pcd

def estimate_normals(pcd):
    ext = np.asarray(pcd.get_axis_aligned_bounding_box().get_extent())
    rad = float(np.linalg.norm(ext) * 0.02 + 1e-6)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=rad, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))

def icp_point_to_plane(src, dst, base_voxel):
    # 대응 거리: voxel의 3~5배 정도
    mcd = (base_voxel if base_voxel > 0 else 0.005) * 4.0
    logger.info(f"[ICP] max_correspondence_distance = {mcd:.5f} m")
    return o3d.pipelines.registration.registration_icp(
        src, dst, max_correspondence_distance=mcd,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )

# ---------- 메인 ----------
pcd1 = read_cloud(filename1)
pcd2 = read_cloud(filename2)

# 자동 voxel 추정 + 안전 다운샘플 + 안전 denoise
vox1 = auto_voxel_size(pcd1)
vox2 = auto_voxel_size(pcd2)
# vox1 = 0.005
# vox2 = 0.005
logger.info(f"[AUTO] voxel1≈{vox1:.5f} m, voxel2≈{vox2:.5f} m")

pcd1_ds, vox1 = safe_downsample(pcd1, vox1)
pcd2_ds, vox2 = safe_downsample(pcd2, vox2)

pcd1_ds = safe_denoise(pcd1_ds)
pcd2_ds = safe_denoise(pcd2_ds)

n1 = np.asarray(pcd1_ds.points).shape[0]
n2 = np.asarray(pcd2_ds.points).shape[0]
if n1 == 0 or n2 == 0:
    raise RuntimeError(f"전처리 후 포인트가 0개입니다. voxel_size를 더 작게(예: 0.002~0.005) 하거나 denoise를 끄세요.")

estimate_normals(pcd1_ds)
estimate_normals(pcd2_ds)

base_voxel = np.median([v for v in [vox1, vox2] if v > 0]) if (vox1>0 or vox2>0) else 0.0
icp = icp_point_to_plane(pcd2_ds, pcd1_ds, base_voxel)
T = icp.transformation
t = T[:3,3]
shift_mag = float(np.linalg.norm(t))

logger.info("\n[RESULT]")
logger.info("T:\n", T)
logger.info(f"|t| = {shift_mag:.4f} m   expected = {expected_shift_m:.4f} m")
logger.info(f"fitness = {icp.fitness:.3f}, inlier_rmse = {icp.inlier_rmse:.4f} m")
delta = abs(shift_mag - expected_shift_m)
logger.info(f"→ error = {delta:.4f} m")
