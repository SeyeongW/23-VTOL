from pathlib import Path
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import numpy as np
import cv2
import hailo
import subprocess # C 프로그램 실행을 위해 추가

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        # C 프로그램 실행 경로 설정 (현재 스크립트와 같은 폴더에 있다고 가정)
        self.c_program_path = "./siyi_gimbal"
        
        # 파일 존재 확인
        if not os.path.exists(self.c_program_path):
            print(f"[ERROR] C program not found at: {self.c_program_path}")
            print("Please compile it first: gcc siyi_gimbal.c -o siyi_gimbal -lm")

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # If the user_data.use_frame is set to True, we can get the video frame from the buffer
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    roi = hailo.get_roi_from_buffer(buffer)
    detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

    keypoints = get_keypoints()

    # [수정됨] 추적할 대상 찾기 (가장 신뢰도 높은 사람)
    target_bbox = None
    max_confidence = 0.0

    for detection in detections:
        label = detection.get_label()
        bbox = detection.get_bbox()
        confidence = detection.get_confidence()
        
        if label == "person":
            if confidence > max_confidence:
                max_confidence = confidence
                # BBox 좌표 저장 [ymin, xmin, ymax, xmax]
                target_bbox = [bbox.ymin(), bbox.xmin(), bbox.ymax(), bbox.xmax()]

            # 시각화 (기존 코드 유지)
            landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
            if len(landmarks) != 0:
                points = landmarks[0].get_points()
                for eye in ['left_eye', 'right_eye']:
                    keypoint_index = keypoints[eye]
                    point = points[keypoint_index]
                    x = int((point.x() * bbox.width() + bbox.xmin()) * width)
                    y = int((point.y() * bbox.height() + bbox.ymin()) * height)
                    if user_data.use_frame:
                        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    # [핵심 수정] C 프로그램 호출하여 짐벌 제어
    # 매 프레임마다 호출하면 부하가 클 수 있으므로, 실제 적용 시에는 
    # frame_count % 3 == 0 등으로 호출 빈도를 줄이는 것을 고려해볼 수 있습니다.
    
    cmd_args = [user_data.c_program_path]
    
    if target_bbox:
        # 사람이 있으면 좌표 전달
        # ./siyi_gimbal ymin xmin ymax xmax
        cmd_args.extend([str(f) for f in target_bbox])
        # print(f"[Track] Person found: {target_bbox}") # 디버깅용
    else:
        # 사람이 없으면 인자 없이 호출 -> C 프로그램 내부에서 정지 명령 처리 필요
        # (아까 작성한 C 코드는 인자가 없으면 종료되므로, 여기서는 호출 안 함)
        # 만약 정지 명령을 보내고 싶다면 C 코드를 수정해서 0 0 0 0을 보내거나 해야 함
        pass 

    # 프로세스 실행 (비동기로 실행하지 않고 기다렸다가 넘어감 - 실시간성 확인 필요)
    if target_bbox:
        try:
            # subprocess.run은 실행이 끝날 때까지 기다립니다.
            # 짐벌 제어 패킷 하나 보내는 건 매우 빠르므로 큰 지연은 없을 것입니다.
            subprocess.run(cmd_args, check=False)
        except Exception as e:
            print(f"[Error] Failed to run C program: {e}")

    if user_data.use_frame:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if width and height:
            cv2.line(frame, (width//2, 0), (width//2, height), (0, 0, 255), 1)
            cv2.line(frame, (0, height//2), (width, height//2), (0, 0, 255), 1)
        user_data.set_frame(frame)

    return Gst.PadProbeReturn.OK

def get_keypoints():
    keypoints = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
        'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
        'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16,
    }
    return keypoints

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    env_file     = project_root / ".env"
    env_path_str = str(env_file)
    os.environ["HAILO_ENV_FILE"] = env_path_str
    
    user_data = user_app_callback_class()
    app = GStreamerPoseEstimationApp(app_callback, user_data)
    app.run()