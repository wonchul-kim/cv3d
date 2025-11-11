import cv2 
import numpy as np
from typing import Tuple, List
import os

def draw_tag_outline(img: np.ndarray, corners: np.ndarray, color=(0, 255, 0), thickness: int = 2) -> None:
    pts = corners.astype(int).reshape(-1, 2)
    for i in range(4):
        p1 = tuple(pts[i])
        p2 = tuple(pts[(i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)


def draw_axes(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, axis_len: float = 0.05) -> None:
    """Draw XYZ axes with length axis_len (meters) on the tag coordinate frame."""
    # Points in tag frame (origin and three axes)
    obj_pts = np.float32([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
    rvec, _ = cv2.Rodrigues(R)
    dcoeff = np.zeros(5)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, dcoeff)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    O, Xp, Yp, Zp = img_pts
    cv2.line(img, tuple(O), tuple(Xp), (0, 0, 255), 2, cv2.LINE_AA)   # X - red
    cv2.line(img, tuple(O), tuple(Yp), (0, 255, 0), 2, cv2.LINE_AA)   # Y - green
    cv2.line(img, tuple(O), tuple(Zp), (255, 0, 0), 2, cv2.LINE_AA)   # Z - blue


def draw_cube(img: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray, tag_size: float) -> None:
    """Draw a wireframe cube standing on top of the tag square."""
    s = tag_size / 2.0
    # 8 vertices: bottom square (z=0) and top square (z=-tag_size)
    obj_pts = np.float32(
        [
            [-s, -s, 0],
            [s, -s, 0],
            [s, s, 0],
            [-s, s, 0],
            [-s, -s, -tag_size],
            [s, -s, -tag_size],
            [s, s, -tag_size],
            [-s, s, -tag_size],
        ]
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
        ]
    )

    rvec, _ = cv2.Rodrigues(R)
    dcoeff = np.zeros(5)
    img_pts, _ = cv2.projectPoints(obj_pts, rvec, t, K, dcoeff)
    img_pts = img_pts.reshape(-1, 2).astype(int)

    for i, j in edges:
        cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), (0, 255, 0), 2, cv2.LINE_AA)


def put_tag_text(
    img: np.ndarray,
    origin_xy: Tuple[int, int],
    text_lines: List[str],
    scale: float = 0.6,
    thickness: int = 2,
    color=(255, 255, 255),
    bg=True,
) -> None:
    """Draw multi-line text with an optional dark background for readability."""
    x, y = origin_xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    if bg:
        # Compute bounding box
        sizes = [cv2.getTextSize(t, font, scale, thickness)[0] for t in text_lines]
        box_w = max((w for (w, h) in sizes), default=0) + 10
        box_h = sum((h + 8 for (w, h) in sizes), 0) + 10
        cv2.rectangle(img, (x - 5, y - box_h + 5), (x - 5 + box_w, y + 5), (0, 0, 0), -1)
    # Draw lines
    y_cursor = y
    for t in text_lines:
        cv2.putText(img, t, (x, y_cursor), font, scale, color, thickness, cv2.LINE_AA)
        y_cursor += int(22 * scale) + 6

def draw_principal_point(img: np.ndarray, K: np.ndarray, size: int = 12) -> None:
    cx, cy = int(K[0, 2]), int(K[1, 2])
    H, W = img.shape[:2]
    cx = max(0, min(cx, W-1)); cy = max(0, min(cy, H-1))
    cv2.line(img, (cx - size, cy), (cx + size, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - size), (cx, cy + size), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 3, (0, 0, 0), -1, cv2.LINE_AA)


def format_xyz_m(t: np.ndarray) -> Tuple[float, float, float]:
    x, y, z = float(t[0, 0]), float(t[1, 0]), float(t[2, 0])
    return x, y, z

def format_idx(i: int) -> str:
    return f"{i:07d}"

def to_SE3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R.astype(np.float32)
    T[:3, 3] = t.reshape(3).astype(np.float32)
    return T

def save_matrix_txt(path: str, T: np.ndarray) -> None:
    # Save 4x4 or 3x3 as text, rows on newlines, space-separated
    np.savetxt(path, T, fmt="%.9f")

def ensure_dirs(root: str) -> Tuple[str,str,str]:
    rgb_dir   = os.path.join(root, "rgb")
    depth_dir = os.path.join(root, "depth")
    pose_dir  = os.path.join(root, "cam_in_ob")
    os.makedirs(rgb_dir,   exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir,  exist_ok=True)
    return rgb_dir, depth_dir, pose_dir
