import open3d as o3d
import numpy as np

filename1 = '/HDD/etc/stereo_cam_calibration/images/outputs5/cloud_from_depth_1.ply'
filename2 = '/HDD/etc/stereo_cam_calibration/images/outputs5/cloud_from_depth_2.ply'

# 파일 읽기
pcd_1 = o3d.io.read_point_cloud(filename1)
pcd_2 = o3d.io.read_point_cloud(filename2)

# 읽은 데이터 확인
print(pcd_1)
print("1. 점 개수:", len(pcd_1.points))
print("2. 점 개수:", len(pcd_2.points))

# NumPy 배열로 변환하여 사용
points_np_1 = np.asarray(pcd_1.points) # 또는 np.asarray(pcd.points)
points_np_2 = np.asarray(pcd_2.points) # 또는 np.asarray(pcd.points)

H, W = 2048, 2448

reshaped_points_1 = points_np_1.reshape(H, W, 3)
reshaped_points_2 = points_np_2.reshape(H, W, 3)

u, v = 1200, 600
# 해당 픽셀의 3D 좌표 (x, y, z) 가져오기
# Z 값은 깊이
x_coord_1 = reshaped_points_1[v, u, 0]
y_coord_1 = reshaped_points_1[v, u, 1]
z_coord_1 = reshaped_points_1[v, u, 2]

print(f"1. 픽셀 ({u}, {v})에 해당하는 깊이(z) 값: {z_coord_1}")
print(f"1. 픽셀 ({u}, {v})에 해당하는 3D 좌표: ({x_coord_1}, {y_coord_1}, {z_coord_1})")

x_coord_2 = reshaped_points_2[v, u, 0]
y_coord_2 = reshaped_points_2[v, u, 1]
z_coord_2 = reshaped_points_2[v, u, 2]

print(f"2. 픽셀 ({u}, {v})에 해당하는 깊이(z) 값: {z_coord_2}")
print(f"2. 픽셀 ({u}, {v})에 해당하는 3D 좌표: ({x_coord_2}, {y_coord_2}, {z_coord_2})")