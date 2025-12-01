from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

# [추가됨] 짐벌 제어 모듈 임포트
# 같은 폴더에 siyi_gimbal.py 파일이 있어야 합니다.
from siyi_gimbal import SiyiGimbal

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
# Inheritance from the app_callback_class
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # [추가됨] 짐벌 컨트롤러 초기화 (IP는 상황에 맞게 수정될 수 있음)
        print("[System] Initializing SIYI Gimbal Connection...")
        self.gimbal = SiyiGimbal(debug=True) # 디버그 모드 켜서 터미널에서 움직임 확인 가능

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------

# This is the callback function that will be called when data is available from the pipeline
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    # string_to_print = f"Frame count: {user_data.get_count()}\n" # 로그 너무 많으면 주석 처리

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    # Get the keypoints
    keypoints = get_keypoints()

    # [추가됨] 추적할 대상의 BBox를 저장할 변수
    target_bbox_list = None
    max_confidence = 0.0 # 가장 확실한 사람을 찾기 위해

    # Parse the detections
    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            # [추가됨] 가장 신뢰도가 높은 사람을 타겟으로 선정
            if confidence > max_confidence:
                max_confidence = confidence
                # Hailo BBox 객체에서 좌표 추출 (ymin, xmin, ymax, xmax)
                target_bbox_list = [bbox.ymin(), bbox.xmin(), bbox.ymax(), bbox.xmax()]

            # (기존 로직) 로그 출력 및 그리기
            # Get track ID
            track_id = 0
            track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            if len(track) == 1:
                track_id = track[0].get_id()
            # string_to_print += (f"Detection: ID: {track_id} Label: {label} Confidence: {confidence:.2f}\n")

            # Pose estimation landmarks from detection (if available)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                for eye in ['left_eye', 'right_eye']:
                    keypoint_index = keypoints[eye]
                    point = points[keypoint_index]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    # string_to_print += f"{eye}: x: {x:.2f} y: {y:.2f}\n"
                    if user_data.use_frame:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # [추가됨] 짐벌 제어 명령 전송 (매 프레임마다 호출)
    # target_bbox_list가 None이면(사람 없으면) siyi_gimbal.py 내부에서 알아서 정지함
    if width is not None and height is not None:
        user_data.gimbal.track_object(target_bbox_list, width, height)

    if user_data.use_frame:
        # Convert the frame to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # [추가됨] 화면 중앙에 십자선 그리기 (추적 중심 확인용)
        if width and height:
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 0, 255), 1)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 0, 255), 1)

        user_data.set_frame(frame)

    # print(string_to_print) # 로그가 너무 많아 터미널이 느려질 수 있어 주석 처리 권장
    return Gst.PadProbeReturn.OK

# This function can be used to get the COCO keypoints coorespondence map
def get_keypoints():
    """Get the COCO keypoints and their left/right flip coorespondence map."""
    keypoints = {
        'nose': 0,
        'left_eye': 1,
        'right_eye': 2,
        'left_ear': 3,
        'right_ear': 4,
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12,
        'left_knee': 13,
        'right_knee': 14,
        'left_ankle': 15,
        'right_ankle': 16,
    }

    return keypoints

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    # Create an instance of the user app callback class
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()