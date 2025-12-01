from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import subprocess

from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # Path to compiled C program that controls the gimbal
        # (Assuming siyi_gimbal is in the same directory as this script)
        self.c_program_path = "./siyi_gimbal"

        # Whether to get video frames from the buffer
        self.use_frame = True
        self._frame = None

        if not os.path.exists(self.c_program_path):
            print(f"[ERROR] C program not found at: {self.c_program_path}")
            print("Please compile it first: gcc siyi_gimbal.c -o siyi_gimbal -lm")

    def set_frame(self, frame):
        self._frame = frame

    def get_frame(self):
        return self._frame


# -----------------------------------------------------------------------------------------------
# Main callback function called by the GStreamerPoseEstimationApp
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Increase frame counter if available
    try:
        user_data.increment()
    except AttributeError:
        pass

    # Get caps (format, width, height) from the pad
    caps_format, width, height = get_caps_from_pad(pad)

    # Retrieve frame as numpy array if requested
    frame = None
    if (
        user_data.use_frame
        and caps_format is not None
        and width is not None
        and height is not None
    ):
        frame = get_numpy_from_buffer(buffer, caps_format, width, height)

    # Get Hailo ROI and detections
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    keypoints = get_keypoints()

    # -------------------------------------------------------------------------------------------
    # 1) Select target person (highest confidence)
    # -------------------------------------------------------------------------------------------
    target_bbox = None
    max_confidence = 0.0

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        if label == "person" and confidence > max_confidence:
            max_confidence = confidence
            # Save bbox as [ymin, xmin, ymax, xmax] in normalized coordinates
            target_bbox = [bbox.ymin(), bbox.xmin(), bbox.ymax(), bbox.xmax()]

        # Visualization of landmarks (eyes) – same style as 네 원본
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks and frame is not None:
            points = landmarks[0].get_points()
            for eye in ["left_eye", "right_eye"]:
                keypoint_index = keypoints[eye]
                point = points[keypoint_index]
                x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # -------------------------------------------------------------------------------------------
    # 2) Call C program to control the gimbal
    # -------------------------------------------------------------------------------------------
    # 형식:
    #   - 사람이 있음  -> ./siyi_gimbal ymin xmin ymax xmax
    #   - 사람이 없음  -> ./siyi_gimbal           (argc == 1 → C에서 auto-center)
    cmd_args = [user_data.c_program_path]

    if target_bbox is not None:
        cmd_args.extend(str(f) for f in target_bbox)

    try:
        subprocess.run(cmd_args, check=False)
    except Exception as e:
        print(f"[Error] Failed to run C program: {e}")

    # -------------------------------------------------------------------------------------------
    # 3) Draw center lines and set frame back to user_data for UI
    # -------------------------------------------------------------------------------------------
    if frame is not None:
        # Draw center crosshair
        if width and height:
            cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 0, 255), 1)
            cv2.line(frame, (0, height // 2), (width, height // 2), (0, 0, 255), 1)

        # Hailo 예제는 RGB 기준이 많아서 BGR로 변환해서 GUI에 넘김
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# Keypoint indices helper
# -----------------------------------------------------------------------------------------------
def get_keypoints():
    keypoints = {
        "nose": 0,
        "left_eye": 1,
        "right_eye": 2,
        "left_ear": 3,
        "right_ear": 4,
        "left_shoulder": 5,
        "right_shoulder": 6,
        "left_elbow": 7,
        "right_elbow": 8,
        "left_wrist": 9,
        "right_wrist": 10,
        "left_hip": 11,
        "right_hip": 12,
        "left_knee": 13,
        "right_knee": 14,
        "left_ankle": 15,
        "right_ankle": 16,
    }
    return keypoints


# -----------------------------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Hailo .env 로딩 (원래 예제와 동일한 방식)
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
