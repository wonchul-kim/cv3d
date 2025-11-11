from dataclasses import dataclass
import cv2 
from typing import Optional, Tuple, Union
import math
import numpy as np


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)


class FrameSource:
    """Abstract frame source."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError

    def intrinsics(self) -> Intrinsics:
        raise NotImplementedError

    def release(self) -> None:
        pass


class RealSenseSource(FrameSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        try:
            import pyrealsense2 as rs  # type: ignore
        except Exception as e:
            raise RuntimeError("pyrealsense2 not installed") from e
        
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16,   fps)
        self.profile = self.pipeline.start(self.config)
        # align depth to color
        self.align = rs.align(rs.stream.color)

        color_stream = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()
        self._intr = Intrinsics(fx=float(intr.fx), fy=float(intr.fy), cx=float(intr.ppx), cy=float(intr.ppy))

        # depth scale (meters per unit)
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        c = aligned.get_color_frame()
        d = aligned.get_depth_frame()
        if not c or not d:
            return None, None
        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data())  # uint16 in native units
        # Convert native units to millimeters for saving as 16-bit PNG
        depth_mm = (depth.astype(np.float32) * self.depth_scale * 1000.0).round().astype(np.uint16)
        return color, depth_mm

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def release(self) -> None:
        self.pipeline.stop()


class OpenCVSource(FrameSource):
    def __init__(
        self,
        src: Union[int, str] = 0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        guess_fov_deg: float = 60.0,
    ):
        self.cap = cv2.VideoCapture(src)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Get frame size
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or (width or 640))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or (height or 480))

        # If intrinsics provided, use them; else create a rough guess
        if fx and fy and cx is not None and cy is not None:
            self._intr = Intrinsics(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy))
        else:
            # Fallback heuristic (⚠️ not accurate; use proper calibration for real scale)
            f = (w / 2.0) / math.tan(math.radians(guess_fov_deg) / 2.0)
            self._intr = Intrinsics(fx=float(f), fy=float(f), cx=w / 2.0, cy=h / 2.0)
            print(
                "[WARN] Using a rough focal-length guess from FOV=%.1f°. "
                "Provide --fx --fy --cx --cy for accurate metric scale." % guess_fov_deg
            )

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self.cap.read()
        return frame if ok else None

    def intrinsics(self) -> Intrinsics:
        return self._intr

    def release(self) -> None:
        self.cap.release()