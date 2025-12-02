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
import threading  # ← 제스처를 별도 스레드에서 돌리기 위해 추가

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
    """
    coords = []
    for p in points:
        coords.append(p.x())
        coords.append(p.y())
    return np.array(coords, dtype=np.float32)


def run_gimbal_gesture_with_a8(user_data):
    """
    A8miniControl을 이용해서:
      1) 오른쪽으로 회전 (3)
      2) 잠깐 대기
      3) 왼쪽으로 회전 (4)
      4) 잠깐 대기
      5) 정지 (5)
    를 순차적으로 실행. (좌우로 끝까지 왕복하는 제스처 느낌)
    """
    a8_path = getattr(user_data, "a8_program_path", None)
    if not a8_path or not os.path.exists(a8_path):
        print(f"[GESTURE] A8miniControl not found at: {a8_path}")
        return

    try:
        print("[GESTURE] → A8miniControl 3 (Rotate Right)")
        subprocess.run([a8_path, "3"], check=False)
        time.sleep(2.0)  # 필요하면 1.5 ~ 3.0 정도로 조정

        print("[GESTURE] → A8miniControl 4 (Rotate Left)")
        subprocess.run([a8_path, "4"], check=False)
        time.sleep(2.0)

        print("[GESTURE] → A8miniControl 5 (Stop)")
        subprocess.run([a8_path, "5"], check=False)

        print("[GESTURE] Done.")
    except Exception as e:
        print(f"[GESTURE] Error while running A8miniControl gesture: {e}")


def on_pose_stable(user_data, frame, pose_vec):
    """
    Pose가 user_data.pose_stable_sec 동안 안정적일 때 한 번만 호출됨.
    여기서 A8miniControl 기반 제스처를 비동기로 실행.
    """
    print("[DEBUG] on_pose_stable() called")

    if getattr(user_data, "gesture_done", False):
        print("[POSE] Gesture already done, skipping.")
        return

    user_data.gesture_done = True  # 다시 못 하게 잠금
    print(
        "[POSE] Pose has been stable for %.1f seconds. Triggering gesture..."
        % user_data.pose_stable_sec
    )

    # GStreamer 콜백을 막지 않도록 별도 스레드에서 제스처 실행
    t = threading.Thread(
        target=run_gimbal_gesture_with_a8,
        args=(user_data,),
        daemon=True,
    )
    t.start()


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


# -----------------------------------------------------------------------------------------------
# User callback class
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()

        # 이 스크립트(.py)가 있는 디렉토리 기준으로 절대경로 생성
        script_dir = Path(__file__).resolve().parent
        self.c_program_path = str(script_dir / "siyi_gimbal")      # 추적용 C
        self.a8_program_path = str(script_dir / "A8miniControl")   # 제스처용 A8miniControl

        self.gesture_done = False   # 제스처는 1번만

        # Whether to get video frames from the buffer
        self.use_frame = True
        self._frame = None

        # pose stability state
        self.filtered_pose_vec = None
        self.last_pose_vec = None
        self.pose_stable_time = 0.0

        # 튜닝 파라미터 (조금 느슨하게)
        self.pose_alpha = 0.2
        self.pose_eps = 0.10          # 허용되는 포즈 변화 (L2 norm)
        self.pose_stable_sec = 3.0    # 안정 유지 시간

        # 싱글 타겟 추적
        self.active_bbox = None
        self.target_lost_frames = 0
        self.max_lost_frames = 15

        # 디버그용
        self.last_ts = None
        self.debug_counter = 0
        self.debug_pose = True

        if not os.path.exists(self.c_program_path):
            print(f"[ERROR] C program not found at: {self.c_program_path}")
            print("  → gcc siyi_gimbal.c -o siyi_gimbal -lm")

        if not os.path.exists(self.a8_program_path):
            print(f"[WARN] A8miniControl not found at: {self.a8_program_path}")
            print("  → 이 이름/경로가 다르면 self.a8_program_path 를 수정해야 함.")


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

            # Visualization of eyes
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

        # reset pose stability
        user_data.filtered_pose_vec = None
        user_data.last_pose_vec = None
        user_data.pose_stable_time = 0.0

    else:
        # If no active target yet -> choose person closest to center
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
    # 3) Pose stability check (for the active target only)
    #    - EMA filtering + stable_time accumulation
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

    # 디버그용 프레임 카운터
    user_data.debug_counter += 1

    if target_pose_vec is not None:
        # 1) EMA filter update
        if user_data.filtered_pose_vec is None:
            user_data.filtered_pose_vec = target_pose_vec.copy()
            user_data.last_pose_vec     = user_data.filtered_pose_vec.copy()
            user_data.pose_stable_time  = 0.0
        else:
            alpha = user_data.pose_alpha
            user_data.filtered_pose_vec = (
                alpha * target_pose_vec
                + (1.0 - alpha) * user_data.filtered_pose_vec
            )

        # 2) diff between filtered pose and reference pose
        diff = np.linalg.norm(user_data.filtered_pose_vec - user_data.last_pose_vec)

        if diff < user_data.pose_eps:
            # almost no movement -> accumulate stable time
            user_data.pose_stable_time += dt
        else:
            # some movement -> decrease stable time gradually
            user_data.pose_stable_time = max(
                0.0, user_data.pose_stable_time - dt * 0.5
            )
            user_data.last_pose_vec = user_data.filtered_pose_vec.copy()

        # 디버그 출력: 10프레임마다 한 번
        if user_data.debug_pose and (user_data.debug_counter % 10 == 0):
            print(
                f"[POSEDBG] diff={diff:.4f}, "
                f"stable_time={user_data.pose_stable_time:.2f}, "
                f"gesture_done={user_data.gesture_done}"
            )

        # 3) trigger gesture once when stable enough
        if (
            (not user_data.gesture_done)
            and user_data.pose_stable_time >= user_data.pose_stable_sec
        ):
            print("[POSEDBG] Stable enough → calling on_pose_stable()")
            on_pose_stable(user_data, frame, user_data.filtered_pose_vec)
    else:
        # No valid pose -> reset stability state
        user_data.filtered_pose_vec = None
        user_data.last_pose_vec = None
        user_data.pose_stable_time = 0.0

    # -------------------------------------------------------------------------------------------
    # 4) Call C program to control the gimbal (tracking)
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

        # mark active target bbox (optional 시각화)
        if target_bbox is not None:
            ymin, xmin, ymax, xmax = target_bbox
            x1 = int(xmin * width)
            y1 = int(ymin * height)
            x2 = int(xmax * width)
            y2 = int(ymax * height)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

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
