from pathlib import Path
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import subprocess
import time

from hailo_apps.hailo_app_python.core.common.buffer_utils import (
    get_caps_from_pad,
    get_numpy_from_buffer,
)
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import (
    GStreamerPoseEstimationApp,
)


# -----------------------------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------------------------
def pose_to_vector(points):
    """
    Convert Hailo landmark points to a 1D numpy vector [x0, y0, x1, y1, ...].
    (Currently only used for visualization if needed.)
    """
    coords = []
    for p in points:
        coords.append(p.x())
        coords.append(p.y())
    return np.array(coords, dtype=np.float32)


def on_pose_stable(user_data, frame, pose_vec):
    """
    Called once when the selected target has been stable enough.
    Triggers the yaw-gesture via the external C program.
    """
    print("[DEBUG] on_pose_stable() called")

    if getattr(user_data, "gesture_done", False):
        print("[POSE] Gesture already done, skipping.")
        return

    user_data.gesture_done = True  # lock so it is triggered only once
    print(
        "[POSE] Target has been stable for %.1f seconds. Triggering gesture..."
        % user_data.center_stable_sec
    )

    try:
        subprocess.Popen([user_data.c_program_path, "gesture"])
        print(f"[POSE] Started gesture process: {user_data.c_program_path} gesture")
    except Exception as e:
        print(f"[Error] Failed to run gesture: {e}")


def bbox_iou(a, b):
    """
    IoU of two bboxes in [ymin, xmin, ymax, xmax] normalized coords.
    """
    aymin, axmin, aymax, axmax = a
    bymin, bxmin, bymax, bxmax = b

    inter_ymin = max(aymin, bymin)
    inter_xmin = max(axmin, bxmin)
    inter_ymax = min(aymax, bymax)
    inter_xmax = min(axmax, bxmax)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, (aymax - aymin)) * max(0.0, (axmax - axmin))
    area_b = max(0.0, (bymax - bymin)) * max(0.0, (bxmax - bxmin))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0

    return inter_area / union


def get_keypoints():
    """
    Pose keypoint index mapping (Hailo pose model).
    """
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
# User callback class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        # Absolute path to the compiled C gimbal control binary
        script_dir = Path(__file__).resolve().parent
        self.c_program_path = str(script_dir / "siyi_gimbal")

        self.gesture_done = False   # ensure gesture can fire once

        # Whether to get video frames from the buffer
        self.use_frame = True
        self._frame = None

        # (Optional) pose-based stability state (not strictly needed now)
        self.filtered_pose_vec = None
        self.last_pose_vec = None
        self.pose_stable_time = 0.0

        # Bbox center-based stability state (used for gesture trigger)
        self.center_filtered = None      # EMA-filtered bbox center [cx, cy]
        self.center_ref = None           # reference center
        self.center_stable_time = 0.0    # accumulated "almost not moving" time

        # Tuning parameters
        self.pose_alpha = 0.2        # EMA factor (0~1, larger = quicker reaction)
        self.center_eps = 0.03       # allowed center movement (in normalized coords)
        self.center_stable_sec = 3.0 # required stability before triggering gesture

        # Single active target tracking
        self.active_bbox = None
        self.target_lost_frames = 0
        self.max_lost_frames = 15    # after this many lost frames, drop the target

        # Time / debug helpers
        self.last_ts = None
        self.debug_counter = 0
        self.debug_pose = True

        if not os.path.exists(self.c_program_path):
            print(f"[ERROR] C program not found at: {self.c_program_path}")
            print("Please compile it first: gcc siyi_gimbal.c -o siyi_gimbal -lm")

    def set_frame(self, frame):
        self._frame = frame

    def get_frame(self):
        return self._frame


# -----------------------------------------------------------------------------------------------
# Main callback function
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
    # 1) Collect all persons in this frame
    #    persons = [(bbox_list, confidence, pose_vec), ...]
    # -------------------------------------------------------------------------------------------
    persons = []

    for detection in detections:
        label = detection.get_label()
        if label != "person":
            continue

        bbox = detection.get_bbox()
        confidence = detection.get_confidence()

        ymin = bbox.ymin()
        xmin = bbox.xmin()
        ymax = bbox.ymax()
        xmax = bbox.xmax()

        pose_vec = None
        landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
        if landmarks:
            points = landmarks[0].get_points()
            pose_vec = pose_to_vector(points)

            # Visualization of eyes for debug
            if frame is not None:
                for eye in ["left_eye", "right_eye"]:
                    keypoint_index = keypoints[eye]
                    point = points[keypoint_index]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        persons.append(([ymin, xmin, ymax, xmax], confidence, pose_vec))

    # -------------------------------------------------------------------------------------------
    # 2) Single active target selection (IoU-based tracking)
    # -------------------------------------------------------------------------------------------
    target_bbox = None
    target_pose_vec = None

    if not persons:
        # No persons at all in this frame
        user_data.target_lost_frames += 1
        if user_data.target_lost_frames > user_data.max_lost_frames:
            user_data.active_bbox = None

        # reset center stability
        user_data.center_filtered = None
        user_data.center_ref = None
        user_data.center_stable_time = 0.0

    else:
        # If no active target yet -> choose person closest to image center
        if user_data.active_bbox is None:
            def center_dist(bbox_list):
                ymin, xmin, ymax, xmax = bbox_list
                cx = (xmin + xmax) * 0.5
                cy = (ymin + ymax) * 0.5
                return (cx - 0.5) ** 2 + (cy - 0.5) ** 2

            persons.sort(key=lambda p: center_dist(p[0]))
            user_data.active_bbox = persons[0][0]
            user_data.target_lost_frames = 0

        # Find the detection that best matches the active_bbox (IoU)
        best_iou = 0.0
        best_person = None
        for bbox_list, conf, pose_vec in persons:
            iou = bbox_iou(user_data.active_bbox, bbox_list)
            if iou > best_iou:
                best_iou = iou
                best_person = (bbox_list, conf, pose_vec)

        if best_iou < 0.3:
            # assume we lost the target
            user_data.target_lost_frames += 1
            if user_data.target_lost_frames > user_data.max_lost_frames:
                user_data.active_bbox = None
            target_bbox = None
            target_pose_vec = None
        else:
            # same person as previous target
            user_data.active_bbox = best_person[0]
            target_bbox = best_person[0]
            target_pose_vec = best_person[2]
            user_data.target_lost_frames = 0

    # -------------------------------------------------------------------------------------------
    # 3) Stability check (using bbox center)
    #    - If bbox center is almost not moving for given duration -> trigger gesture once
    # -------------------------------------------------------------------------------------------
    now = time.time()
    if user_data.last_ts is None:
        dt = 0.0
    else:
        dt = now - user_data.last_ts
    user_data.last_ts = now

    if dt <= 0.0 or dt > 1.0:
        # If timing is weird, assume ~30 FPS
        dt = 1.0 / 30.0

    # Debug frame counter
    user_data.debug_counter += 1

    if target_bbox is not None:
        # Compute bbox center in normalized coords
        ymin, xmin, ymax, xmax = target_bbox
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5
        center_vec = np.array([cx, cy], dtype=np.float32)

        # 1) EMA update
        if user_data.center_filtered is None:
            user_data.center_filtered = center_vec.copy()
            user_data.center_ref = center_vec.copy()
            user_data.center_stable_time = 0.0
        else:
            alpha = user_data.pose_alpha
            user_data.center_filtered = (
                alpha * center_vec + (1.0 - alpha) * user_data.center_filtered
            )

        # 2) distance between filtered center and reference
        diff_center = np.linalg.norm(user_data.center_filtered - user_data.center_ref)

        if diff_center < user_data.center_eps:
            # almost no movement -> accumulate stable time
            user_data.center_stable_time += dt
        else:
            # some movement -> reduce stable time and update reference
            user_data.center_stable_time = max(
                0.0, user_data.center_stable_time - dt * 0.5
            )
            user_data.center_ref = user_data.center_filtered.copy()

        # Debug output every 10 frames
        if user_data.debug_pose and (user_data.debug_counter % 10 == 0):
            print(
                f"[POSEDBG] center_diff={diff_center:.4f}, "
                f"center_stable_time={user_data.center_stable_time:.2f}, "
                f"gesture_done={user_data.gesture_done}"
            )

        # 3) Trigger gesture once when stable enough
        if (
            (not user_data.gesture_done)
            and user_data.center_stable_time >= user_data.center_stable_sec
        ):
            print("[POSEDBG] Center stable enough → calling on_pose_stable()")
            on_pose_stable(user_data, frame, None)

    else:
        # No valid target -> reset stability state
        user_data.center_filtered = None
        user_data.center_ref = None
        user_data.center_stable_time = 0.0

    # -------------------------------------------------------------------------------------------
    # 4) Call C program to control the gimbal (tracking + auto-center)
    # -------------------------------------------------------------------------------------------
    # Format:
    #   - If target exists: ./siyi_gimbal ymin xmin ymax xmax
    #   - If no target    : ./siyi_gimbal        (argc == 1 → auto-center in C)
    cmd_args = [user_data.c_program_path]

    if target_bbox is not None:
        cmd_args.extend(str(f) for f in target_bbox)

    try:
        subprocess.run(cmd_args, check=False)
    except Exception as e:
        print(f"[Error] Failed to run C program: {e}")

    # -------------------------------------------------------------------------------------------
    # 5) Draw center lines and set frame back to user_data for UI
    # -------------------------------------------------------------------------------------------
    if frame is not None:
        # Draw center crosshair
        if width and height:
            cv2.line(
                frame,
                (width // 2, 0),
                (width // 2, height),
                (0, 0, 255),
                1,
            )
            cv2.line(
                frame,
                (0, height // 2),
                (width, height // 2),
                (0, 0, 255),
                1,
            )

        # Mark active target bbox
        if target_bbox is not None:
            ymin, xmin, ymax, xmax = target_bbox
            x1 = int(xmin * width)
            y1 = int(ymin * height)
            x2 = int(xmax * width)
            y2 = int(ymax * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Hailo examples often use RGB internally; convert to BGR for OpenCV display
        user_data.set_frame(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return Gst.PadProbeReturn.OK


# -----------------------------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Load Hailo .env (same as original example)
    project_root = Path(__file__).resolve().parent.parent
    env_file = project_root / ".env"
    os.environ["HAILO_ENV_FILE"] = str(env_file)

    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()
