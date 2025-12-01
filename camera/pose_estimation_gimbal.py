import socket
import struct
import time


class SiyiGimbal:
    """
    SIYI A8 Mini gimbal control (compatible with A8miniControl C protocol).

    Features:
      1. Speed control (CMD 0x07)  -> for tracking
      2. Auto centering (CMD 0x08) -> for recentering when target is lost
    """

    def __init__(self, ip="192.168.144.25", port=37260, debug=False):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.debug = debug

        # Control parameters
        self.kp_x = 0.5   # yaw gain (left-right)
        self.kp_y = 0.5   # pitch gain (up-down)
        self.deadzone = 0.05  # normalized deadzone around image center

    def _crc16_cal(self, data: bytes, crc: int = 0) -> int:
        """
        CRC16-CCITT (poly = 0x1021, init = 0x0000, MSB-first),
        same as used in SIYI A8miniControl C code.
        """
        poly = 0x1021
        crc &= 0xFFFF

        for b in data:
            crc ^= (b << 8) & 0xFFFF
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) & 0xFFFF) ^ poly
                else:
                    crc = (crc << 1) & 0xFFFF

        return crc & 0xFFFF

    def send_packet(self, cmd_id: int, data_bytes: bytes):
        """
        Build and send a SIYI packet over UDP.

        Packet format:
          [0-1]  STX      : 0x55 0x66
          [2]    CTRL     : 0x01
          [3-4]  DATA_LEN : uint16 LE
          [5-6]  SEQ      : uint16 LE (0)
          [7]    CMD_ID   : uint8
          [8..]  DATA     : variable length
          [..]   CRC      : uint16 LE, CRC16-CCITT over all bytes before CRC
        """
        stx = b"\x55\x66"
        ctrl = b"\x01"
        data_len = len(data_bytes)
        data_len_bytes = struct.pack("<H", data_len)
        seq_bytes = struct.pack("<H", 0)  # sequence fixed to 0
        cmd_id_bytes = struct.pack("<B", cmd_id)

        packet_without_crc = (
            stx + ctrl + data_len_bytes + seq_bytes + cmd_id_bytes + data_bytes
        )
        crc_val = self._crc16_cal(packet_without_crc, 0)
        crc_bytes = struct.pack("<H", crc_val)

        final_packet = packet_without_crc + crc_bytes

        try:
            self.sock.sendto(final_packet, (self.ip, self.port))
            if self.debug:
                hex_str = " ".join(f"{b:02X}" for b in final_packet)
                print(f"[SIYI TX] {hex_str}")
        except Exception as e:
            print(f"[SIYI] Send error: {e}")

    def center(self):
        """
        Send auto-centering command.

        CMD_ID: 0x08
        DATA  : 0x01
        This matches the A8miniControl example:
        {0x55,0x66,0x01,0x01,0x00,0x00,0x00,0x08,0x01,0xD1,0x12}
        """
        if self.debug:
            print("[SIYI] Target lost -> auto centering")
        self.send_packet(0x08, b"\x01")

    def send_speed(self, yaw_speed: int, pitch_speed: int):
        """
        Send gimbal speed command.

        CMD_ID: 0x07
        DATA  : yaw(int8), pitch(int8)
        The SIYI sample "Stop Rotation" packet is essentially CMD 0x07
        with both speeds set to 0.
        """
        # Clamp range to [-100, 100]
        yaw_speed = int(max(-100, min(100, yaw_speed)))
        pitch_speed = int(max(-100, min(100, pitch_speed)))

        data = struct.pack("<bb", yaw_speed, pitch_speed)
        if self.debug:
            print(f"[SIYI] Speed command yaw={yaw_speed}, pitch={pitch_speed}")
        self.send_packet(0x07, data)

    def track_object(self, bbox, w: int, h: int):
        """
        High-level tracking logic.

        Args:
            bbox: normalized [ymin, xmin, ymax, xmax] or None
            w   : image width  (currently not used, kept for interface)
            h   : image height (currently not used, kept for interface)

        Behavior:
          - If bbox is None/invalid  -> send auto-center command.
          - If bbox is valid        -> send speed command to move target to center.
        """
        # No target: auto center
        if bbox is None or len(bbox) != 4:
            self.center()
            return

        # Normalized coordinates from detection (0.0 ~ 1.0)
        ymin, xmin, ymax, xmax = bbox
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0

        # Error from image center (0.5, 0.5)
        err_x = cx - 0.5
        err_y = cy - 0.5

        # Deadzone around center to avoid oscillation
        if abs(err_x) < self.deadzone:
            err_x = 0.0
        if abs(err_y) < self.deadzone:
            err_y = 0.0

        # P-control: convert error to speed command
        yaw_cmd = err_x * 100.0 * self.kp_x
        pitch_cmd = -err_y * 100.0 * self.kp_y  # pitch direction inverted

        if self.debug:
            print(
                f"[TRACK] bbox={bbox}, cx={cx:.3f}, cy={cy:.3f}, "
                f"err=({err_x:.3f},{err_y:.3f}), cmd=({yaw_cmd:.1f},{pitch_cmd:.1f})"
            )

        self.send_speed(int(yaw_cmd), int(pitch_cmd))

    def close(self):
        """Close UDP socket."""
        try:
            self.sock.close()
        except Exception:
            pass


# Simple standalone test
if __name__ == "__main__":
    g = SiyiGimbal(debug=True)
    print("Testing SIYI gimbal")

    print("1) Move right for a short time...")
    for _ in range(5):
        g.send_speed(15, 0)
        time.sleep(0.1)

    print("2) Auto centering...")
    g.center()
    g.close()
    print("Done.")
