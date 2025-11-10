import numpy as np 
from scipy.spatial.transform import Rotation as R
import cv2
import math

def mat4_to_list(M): return [[float(x) for x in row] for row in M]


def quat_xyzw_to_rot(q):
    """Quaternion [x,y,z,w] -> 3x3 rotation."""
    x, y, z, w = q
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = x/n, y/n, z/n, w/n
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x + y*y)]
    ], dtype=float)

def quat_xyzw_to_T(pos, quat_xyzw):
    R_mat = R.from_quat(quat_xyzw).as_matrix()
    # R_mat = quat_xyzw_to_rot(quat_xyzw)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_mat
    T[:3, 3] = np.array(pos)
    return T

def rot_to_quat_xyzw(R):
    """3x3 rotation -> quaternion [x,y,z,w]"""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
            w = (R[2,1] - R[1,2]) / s
        elif i == 1:
            s = math.sqrt(1.0 - R[0,0] + R[1,1] - R[2,2]) * 2
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
            w = (R[0,2] - R[2,0]) / s
        else:
            s = math.sqrt(1.0 - R[0,0] - R[1,1] + R[2,2]) * 2
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
            w = (R[1,0] - R[0,1]) / s
    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q

def T(R, t):
    """Compose 4x4 transform."""
    M = np.eye(4)
    M[:3,:3] = R
    M[:3, 3] = np.asarray(t).reshape(3)
    return M

def T_from_flat(flat_matrix):
    return np.array(flat_matrix, dtype=np.float64).reshape(4, 4)

def T_to_rvec_tvec(T):
    R_mat = T[:3, :3].astype(np.float64)
    tvec = T[:3, 3].astype(np.float64)
    rvec, _ = cv2.Rodrigues(R_mat)
    return rvec.reshape(3, 1), tvec.reshape(3, 1)

def invert_T(T):
    R_mat = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R_mat.T
    Ti[:3, 3] = -R_mat.T @ t
    return Ti

def pose_diff_T(T1, T2):
    T_rel = invert_T(T1) @ T2
    r_rel = R.from_matrix(T_rel[:3, :3])
    rot_diff_deg = r_rel.magnitude() * 180.0 / np.pi 
    trans_diff_m = np.linalg.norm(T_rel[:3, 3])
    return rot_diff_deg, trans_diff_m


if __name__ == '__main__':
    print(quat_xyzw_to_T([55, 561, 810], [0.017, 0.078, 0.27, 0.95]))

