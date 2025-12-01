import socket
import struct
import time
import threading

class SiyiGimbal:
    """
    SIYI A8 Mini 짐벌을 제어하기 위한 클래스입니다.
    UDP 프로토콜을 사용하여 짐벌의 회전 속도를 제어합니다.
    """
    def __init__(self, ip='192.168.144.25', port=37260, debug=False):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.debug = debug
        
        # PID 제어 변수 (값 조절이 필요할 수 있음)
        self.kp_x = 0.15  # Yaw(좌우) 반응 속도
        self.kp_y = 0.15  # Pitch(상하) 반응 속도
        
        # 화면 중앙 오차 범위 (Deadzone) - 이 안에 있으면 안 움직임
        self.deadzone = 0.05 # 화면 크기의 5%

        print(f"[SIYI] Gimbal controller initialized. Target: {self.ip}:{self.port}")

    def _append_crc16(self, data):
        """SIYI SDK용 CRC16 체크섬 계산"""
        crc = 0
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = (crc >> 1)
        return struct.pack('<H', crc)

    def send_speed(self, yaw_speed, pitch_speed):
        """
        짐벌 회전 속도 전송
        yaw_speed: -100 ~ 100 (음수: 좌, 양수: 우)
        pitch_speed: -100 ~ 100 (음수: 하, 양수: 상)
        """
        # 속도 범위 제한 (-100 ~ 100)
        yaw_speed = max(-100, min(100, int(yaw_speed)))
        pitch_speed = max(-100, min(100, int(pitch_speed)))

        # SIYI SDK Protocol Packet Construction
        # Header (2B) + Ctrl (1B) + DataLen (2B) + Seq (2B) + CmdID (1B) + Data (2B) + CRC (2B)
        
        STX = b'\x55\x66'
        CTRL = b'\x00' # Need ACK: 0 (No)
        # Data Length: 2 bytes (Yaw speed) + 2 bytes (Pitch speed) = 4 bytes
        DataLen = struct.pack('<H', 4) 
        SEQ = struct.pack('<H', 0) # Sequence number (not critical for UDP speed control)
        CMD_ID = b'\x0B' # Command ID for Gimbal Rotation
        
        # Payload: Yaw(int16) + Pitch(int16)
        DATA = struct.pack('<h', yaw_speed) + struct.pack('<h', pitch_speed)
        
        packet_body = CTRL + DataLen + SEQ + CMD_ID + DATA
        crc = self._append_crc16(packet_body)
        
        final_packet = STX + packet_body + crc
        
        try:
            self.sock.sendto(final_packet, (self.ip, self.port))
            if self.debug:
                print(f"[SIYI] Sent Speed - Yaw: {yaw_speed}, Pitch: {pitch_speed}")
        except Exception as e:
            print(f"[SIYI] Error sending packet: {e}")

    def track_object(self, bbox, frame_width, frame_height):
        """
        BBox 정보를 받아 짐벌이 중앙을 향하도록 제어합니다.
        bbox: [ymin, xmin, ymax, xmax] (Hailo 정규화 좌표 0.0~1.0) 
              또는 [x, y, w, h] (픽셀 좌표) - 상황에 맞춰 자동 계산
        """
        if bbox is None:
            # 대상이 없으면 정지
            self.send_speed(0, 0)
            return

        # Hailo BBox 포맷 처리 (보통 ymin, xmin, ymax, xmax 형태이며 0.0~1.0 정규화됨)
        # 예제 코드의 bbox 포맷을 확인해야 하지만, 보통 정규화된 좌표로 가정합니다.
        ymin, xmin, ymax, xmax = bbox
        
        # 1. BBox의 중심점 계산 (0.0 ~ 1.0)
        center_x = (xmin + xmax) / 2.0
        center_y = (ymin + ymax) / 2.0
        
        # 2. 화면 중앙(0.5, 0.5)과의 오차 계산
        # 화면 중앙은 0.5입니다.
        # 오차 = 목표값(0.5) - 현재값(center)
        # SIYI 짐벌은 (오른쪽으로 가려면 양수, 위로 가려면 양수) -> 방향 체크 필요
        
        error_x = center_x - 0.5
        error_y = center_y - 0.5

        # 3. PID (여기서는 P제어만 사용)로 속도 계산
        # Deadzone 체크 (너무 작은 움직임은 무시해서 떨림 방지)
        if abs(error_x) < self.deadzone: error_x = 0
        if abs(error_y) < self.deadzone: error_y = 0

        # 속도 계산 (P gain 곱하기)
        # X축: 오차가 양수(오른쪽에 있음) -> 오른쪽으로 돌려야 함 -> 양수 속도
        # Y축: 오차가 양수(아래에 있음) -> 아래로 내려야 함 -> 음수 속도 (SIYI Pitch는 위가 양수일 수 있음, 반대면 - 부호 변경)
        
        yaw_cmd = error_x * 100 * self.kp_x * 2 # -1.0~1.0 범위를 -100~100으로 매핑
        pitch_cmd = -error_y * 100 * self.kp_y * 2 # Pitch는 방향 반대일 확률 높음

        # 디버깅 출력
        if self.debug and (yaw_cmd != 0 or pitch_cmd != 0):
            print(f"BBox Center: ({center_x:.2f}, {center_y:.2f}) -> Error: ({error_x:.2f}, {error_y:.2f}) -> Cmd: ({int(yaw_cmd)}, {int(pitch_cmd)})")

        self.send_speed(yaw_cmd, pitch_cmd)

# 테스트 코드 (이 파일을 직접 실행하면 짐벌이 조금씩 움직임)
if __name__ == "__main__":
    gimbal = SiyiGimbal(debug=True)
    print("Testing Gimbal connection...")
    
    # 오른쪽으로 살짝 회전
    gimbal.send_speed(10, 0)
    time.sleep(1)
    
    # 정지
    gimbal.send_speed(0, 0)
    print("Test done.")