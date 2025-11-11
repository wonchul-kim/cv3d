import numpy as np
from typing import List

class ATDetector:
    """
    Wrapper that uses pupil_apriltags if available, otherwise falls back to apriltag.

    Unified output for each detection:
        - corners: (4,2) float32 array (pixel coords, CCW)
        - center:  (2,) float32 array (pixel coords)
        - R:       (3,3) rotation matrix (tag->camera)
        - t:       (3,1) translation vector in meters (tag center in camera frame)
        - tag_id:  int
    """

    def __init__(self, families: str = "tag36h11", nthreads: int = 2, quad_decimate: float = 1.0):
        self.impl = None
        self.detector = None

        # Try pupil_apriltags first
        try:
            from pupil_apriltags import Detector as PupilDetector  # type: ignore
            self.impl = "pupil"
            self.detector = PupilDetector(
                families=families,
                nthreads=nthreads,
                quad_decimate=quad_decimate,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
        except Exception:
            # Fallback to apriltag
            try:
                import apriltag  # type: ignore

                self.impl = "apriltag"
                self.detector = apriltag.Detector(
                    apriltag.DetectorOptions(families=families, quad_decimate=quad_decimate)
                )
            except Exception as e:
                raise RuntimeError(
                    "Neither pupil_apriltags nor apriltag could be imported. Please install one:\n"
                    "  pip install pupil-apriltags\n"
                    "or\n"
                    "  pip install apriltag\n"
                ) from e

    def detect(
        self, gray: np.ndarray, K: np.ndarray, tag_size: float
    ) -> List[dict]:
        """Run detection and pose estimation.

        Args:
            gray: Grayscale image (uint8)
            K: 3x3 camera intrinsic matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
            tag_size: tag side length in meters

        Returns:
            List of dicts with keys: corners, center, R, t, tag_id
        """
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        if self.impl == "pupil":
            # pupil_apriltags
            results = self.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=tag_size,
            )
            out = []
            for r in results:
                R = np.asarray(r.pose_R, dtype=np.float32)
                t = np.asarray(r.pose_t, dtype=np.float32).reshape(3, 1)
                corners = np.asarray(r.corners, dtype=np.float32)
                center = np.asarray(r.center, dtype=np.float32)
                out.append(
                    {"corners": corners, "center": center, "R": R, "t": t, "tag_id": int(r.tag_id)}
                )
            return out

        elif self.impl == "apriltag":
            # apriltag (SWIG bindings)
            results = self.detector.detect(gray)
            out = []
            for r in results:
                pose_M, e0, e1 = self.detector.detection_pose(
                    r, (fx, fy, cx, cy), tag_size
                )
                pose_M = np.asarray(pose_M, dtype=np.float64)  # 4x4
                R = pose_M[:3, :3].astype(np.float32)
                t = pose_M[:3, 3].astype(np.float32).reshape(3, 1)
                corners = np.asarray(r.corners, dtype=np.float32)
                center = np.asarray(r.center, dtype=np.float32)
                tag_id = int(getattr(r, "tag_id", getattr(r, "id", -1)))
                out.append(
                    {"corners": corners, "center": center, "R": R, "t": t, "tag_id": tag_id}
                )
            return out

        else:
            raise RuntimeError("Detector not initialized")

