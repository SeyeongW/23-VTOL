import gi
import time
import threading
import numpy as np
import cv2
import hailo
import queue

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from hailo_apps.hailo_app_python.core.common.buffer_utils import get_caps_from_pad, get_numpy_from_buffer
from hailo_apps.hailo_app_python.core.gstreamer.gstreamer_app import app_callback_class
from hailo_apps.hailo_app_python.apps.pose_estimation.pose_estimation_pipeline import GStreamerPoseEstimationApp
from hailo_apps.hailo_app_python.core.common.core import get_default_parser

# =============================================================================
# [1] MAVROS Interface Node (가상)
# - 역할: 비행제어(PX4)와 통신 담당
# =============================================================================
class MavrosInterfaceNode:
    def __init__(self):
        self.connected = True
        print("[MavrosInterface] Node Initialized. Connection to FCU established.")

    def send_gps_location(self, lat, lon):
        # 실제 드론에서는 여기서 MAVLink 메시지를 보냅니다.
        print(f"\033[96m[MavrosInterface] SENDING TARGET GPS to PX4: Lat={lat}, Lon={lon}\033[0m")

    def set_flight_mode(self, mode):
        print(f"[MavrosInterface] Flight Mode Changed to: {mode}")

# =============================================================================
# [2] Gimbal Control Node (가상)
# - 역할: 짐벌 각도 제어 (Pitch/Yaw)
# =============================================================================
class GimbalControlNode:
    def __init__(self):
        self.current_pitch = 0
        self.current_yaw = 0
        print("[GimbalControl] Node Initialized.")

    def look_at_target(self, x, y, width, height):
        # 화면 중앙과의 오차를 계산해서 짐벌을 움직임
        center_x = width / 2
        center_y = height / 2
        error_x = x - center_x
        
        # 시연용 로그 출력
        # print(f"[GimbalControl] Adjusting Camera... Error X: {error_x:.1f}")

# =============================================================================
# [3] Mission Planner Node
# - 역할: 재난 상황 판단, 자동 구조 미션 수행, Trigger Zone 판단
# =============================================================================
class MissionPlannerNode:
    def __init__(self, mavros, gimbal):
        self.mavros = mavros
        self.gimbal = gimbal
        self.state = "SEARCHING" # 초기 상태: 탐색 중
        self.rescue_triggered = False
        
        # 설정값
        self.FALL_TIME_LIMIT = 5.0  # 5초 이상 미동 없으면 구조
        self.last_move_times = {}   # ID별 마지막 움직임 시간

    def update_person_status(self, person_id, is_moving, bbox):
        current_time = time.time()
        
        # 1. 새로운 사람 등록
        if person_id not in self.last_move_times:
            self.last_move_times[person_id] = current_time

        # 2. 움직임 감지 시 타이머 리셋
        if is_moving:
            self.last_move_times[person_id] = current_time
            self.state = "SEARCHING"
            return "ACTIVE", (0, 255, 0) # 초록색

        # 3. 움직임 없음 -> 시간 체크
        elapsed_time = current_time - self.last_move_times[person_id]

        if elapsed_time < self.FALL_TIME_LIMIT:
            self.state = "ANALYZING"
            return f"ANALYZING ({elapsed_time:.1f}s)", (0, 255, 255) # 노란색
        else:
            # 4. 골든타임 구조 상황
            if self.state != "RESCUE":
                self.trigger_rescue(person_id)
            self.state = "RESCUE"
            return "!! RESCUE TRIGGERED !!", (0, 0, 255) # 빨간색

    def trigger_rescue(self, person_id):
        print(f"\n\033[41m[MissionPlanner] EMERGENCY! Person {person_id} detected as UNCONSCIOUS! \033[0m")
        print("[MissionPlanner] Initiating Automatic Rescue Sequence...")
        
        # 타 노드와 연동
        self.mavros.set_flight_mode("HOLD") # 제자리 비행
        self.mavros.send_gps_location(37.12345, 127.12345) # 현재 위치 전송
        self.gimbal.look_at_target(0, 0, 640, 480) # 타겟 고정

# =============================================================================
# [4] Cam Feedback Node (시각화)
# - 역할: 영상 좌표 전송, 로그 기록, 화면 오버레이
# =============================================================================
class CamFeedbackNode:
    def draw_overlay(self, frame, bbox, text, color, keypoints):
        # 박스 그리기
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
        
        # 텍스트 배경 및 텍스트
        cv2.rectangle(frame, (xmin, ymin-35), (xmax, ymin), color, -1)
        cv2.putText(frame, text, (xmin + 5, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # 관절 포인트 그리기
        for pt in keypoints:
            cv2.circle(frame, pt, 4, color, -1)
            
        # 시스템 상태 표시 (HUD)
        cv2.putText(frame, "System: Ascend X - Rescue AAM", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# =============================================================================
# [5] Object Detection Node (Main Logic)
# - 역할: OpenCV + YOLOv8(Pose) 기반 객체 인식 및 좌표 변환
# =============================================================================
class ObjectDetectionNode(app_callback_class):
    def __init__(self):
        super().__init__()
        # 하위 노드들 초기화 (시스템 구성)
        self.mavros_node = MavrosInterfaceNode()
        self.gimbal_node = GimbalControlNode()
        self.mission_node = MissionPlannerNode(self.mavros_node, self.gimbal_node)
        self.feedback_node = CamFeedbackNode()
        
        self.frame_history = {} # 포즈 히스토리 저장
        self.MOVEMENT_THRESHOLD = 500

    def process_frame(self, pad, info):
        buffer = info.get_buffer()
        if buffer is None: return Gst.PadProbeReturn.OK

        # 영상 프레임 획득
        format, width, height = get_caps_from_pad(pad)
        frame = None
        if self.use_frame and format and width and height:
            frame = get_numpy_from_buffer(buffer, format, width, height)

        # AI 추론 결과 획득
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)

        for detection in detections:
            if detection.get_label() == "person":
                # ID 식별
                track_obj = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                track_id = track_obj[0].get_id() if len(track_obj) == 1 else 0
                
                # 좌표 및 키포인트 추출
                bbox = detection.get_bbox()
                landmarks = detection.get_objects_typed(hailo.HAILO_LANDMARKS)
                
                if landmarks:
                    points = landmarks[0].get_points()
                    keypoints = [(int((p.x() * bbox.width() + bbox.xmin()) * width),
                                  int((p.y() * bbox.height() + bbox.ymin()) * height)) for p in points]
                    
                    # 히스토리 관리
                    if track_id not in self.frame_history: self.frame_history[track_id] = []
                    self.frame_history[track_id].append(keypoints)
                    if len(self.frame_history[track_id]) > 5: self.frame_history[track_id].pop(0)

                    # [알고리즘] 움직임 계산 (Total Energy Control 개념 응용 - 운동량 변화 감지)
                    is_moving = True
                    if len(self.frame_history[track_id]) >= 2:
                        prev = np.array(self.frame_history[track_id][-2])
                        curr = np.array(self.frame_history[track_id][-1])
                        # 유클리드 거리 합산으로 움직임 양 측정
                        movement = np.sum(np.linalg.norm(curr - prev, axis=1))
                        if movement < self.MOVEMENT_THRESHOLD:
                            is_moving = False

                    # [Mission Planner] 상태 판단 요청
                    status_text, color = self.mission_node.update_person_status(track_id, is_moving, bbox)

                    # [Cam Feedback] 시각화
                    if self.use_frame and frame is not None:
                        pixel_bbox = (int(bbox.xmin() * width), int(bbox.ymin() * height),
                                      int((bbox.xmin() + bbox.width()) * width), int((bbox.ymin() + bbox.height()) * height))
                        self.feedback_node.draw_overlay(frame, pixel_bbox, status_text, color, keypoints)

        if self.use_frame and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self.set_frame(frame)

        return Gst.PadProbeReturn.OK

# =============================================================================
# Main Execution
# =============================================================================
def app_callback(pad, info, user_data):
    return user_data.process_frame(pad, info)

if __name__ == "__main__":
    parser = get_default_parser()
    parser.set_defaults(frame_rate=30)
    
    # 시스템 시작
    print("=================================================")
    print("   Ascend X - Auto Rescue AAM System Initiated   ")
    print("=================================================")
    
    user_data = ObjectDetectionNode() # 메인 노드 생성
    app = GStreamerPoseEstimationApp(app_callback, user_data, parser)
    app.run()