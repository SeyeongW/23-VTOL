import socket
import struct
import time

class SiyiGimbal:
    """
    SIYI A8 Mini 짐벌 제어 (C 코드 프로토콜 완벽 호환 버전)
    """
    def __init__(self, ip='192.168.144.25', port=37260, debug=False):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.debug = debug
        self.seq = 0
        
        # 제어 파라미터
        self.kp_x = 0.5  # Yaw 반응 속도
        self.kp_y = 0.5  # Pitch 반응 속도
        self.deadzone = 0.05

    def _crc16_cal(self, data, crc=0):
        """SIYI SDK CRC16"""
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 1:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc = (crc >> 1)
        return crc

    def send_packet(self, cmd_id, data_bytes):
        """패킷 생성 및 전송"""
        STX = b'\x55\x66'
        CTRL = b'\x01' # C코드와 동일하게 1 설정
        
        # 데이터 길이 (Data Length): 2 bytes, Little Endian
        data_len = len(data_bytes)
        DATA_LEN = struct.pack('<H', data_len)
        
        # 시퀀스 (Sequence): C코드처럼 0으로 고정해도 되지만, 증가시켜도 무방
        # self.seq += 1 
        SEQ = struct.pack('<H', 0) # C코드와 똑같이 0으로 고정
        
        CMD_ID = struct.pack('<B', cmd_id)
        
        # CRC 계산을 위한 앞부분
        packet_without_crc = STX + CTRL + DATA_LEN + SEQ + CMD_ID + data_bytes
        
        # CRC 계산
        crc_val = self._crc16_cal(packet_without_crc, 0)
        CRC = struct.pack('<H', crc_val)
        
        final_packet = packet_without_crc + CRC
        
        try:
            self.sock.sendto(final_packet, (self.ip, self.port))
            if self.debug:
                # 패킷 내용을 16진수로 출력 (C 코드 디버그와 비교 가능)
                hex_str = " ".join([f"{b:02x}" for b in final_packet])
                print(f"[TX] {hex_str}")
        except Exception as e:
            print(f"[SIYI] Send Error: {e}")

    def send_speed(self, yaw_speed, pitch_speed):
        """
        짐벌 속도 제어
        CMD_ID: 0x07 (GIMBAL_SPEED)
        Payload: [Yaw(1byte)][Pitch(1byte)] -> Total 2 bytes (signed int8)
        Range: -100 ~ 100
        """
        # 범위 제한 (-100 ~ 100)
        yaw_speed = int(max(-100, min(100, yaw_speed)))
        pitch_speed = int(max(-100, min(100, pitch_speed)))
        
        # [핵심 수정] C코드와 동일하게 'b' (signed char, 1byte) 사용
        # 총 2바이트 페이로드 생성
        data = struct.pack('<b', yaw_speed) + struct.pack('<b', pitch_speed)
        
        # 0x07 명령어로 전송
        self.send_packet(0x07, data)

    def track_object(self, bbox, w, h):
        """객체 추적 로직"""
        if not bbox:
            self.send_speed(0, 0)
            return

        # Hailo BBox (0.0 ~ 1.0)
        ymin, xmin, ymax, xmax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        
        # 중앙(0.5)에서의 오차
        err_x = cx - 0.5
        err_y = cy - 0.5
        
        if abs(err_x) < self.deadzone: err_x = 0
        if abs(err_y) < self.deadzone: err_y = 0
        
        # 속도 계산 (P 제어)
        # 100을 곱해서 -100~100 범위로 만듦
        # 방향이 반대면 - 부호를 붙이거나 떼세요
        yaw_cmd = err_x * 100 * self.kp_x
        pitch_cmd = -err_y * 100 * self.kp_y 
        
        self.send_speed(yaw_cmd, pitch_cmd)

# 테스트: 직접 실행 시 짐벌이 움직여야 함
if __name__ == "__main__":
    g = SiyiGimbal(debug=True)
    print("Testing Gimbal Move (Right 15)...")
    
    # 0.1초 간격으로 10번 전송 (UDP라 여러번 보내는 게 좋음)
    for _ in range(10):
        g.send_speed(15, 0) 
        time.sleep(0.1)
        
    print("Stopping...")
    g.send_speed(0, 0)
