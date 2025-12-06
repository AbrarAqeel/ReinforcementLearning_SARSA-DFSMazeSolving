# video.py
"""
Minimal MP4 writer using OpenCV.
"""
import cv2
import numpy as np
from typing import Optional


class VideoRecorder:
    def __init__(self, output_path: str, fps: int = 10):
        self.output_path = output_path
        self.fps = fps
        self.writer: Optional[cv2.VideoWriter] = None

    def write(self, frame: np.ndarray):
        if frame is None:
            return
        h, w, _ = frame.shape
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (w, h))
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(bgr)

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            print(f"[VIDEO] Saved video to {self.output_path}")
