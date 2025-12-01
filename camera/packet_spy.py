import socket

# SIYI 짐벌 포트
UDP_IP = "0.0.0.0" # 모든 신호 수신
UDP_PORT = 37260

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for SIYI packets on port {UDP_PORT}...")
print("Now run your './A8miniControl' in another terminal!")

while True:
    data, addr = sock.recvfrom(1024)
    # 받은 데이터를 HEX(16진수)로 예쁘게 출력
    hex_data = " ".join([f"{b:02X}" for b in data])
    print(f"[{addr[0]}] Length: {len(data)} | Data: {hex_data}")
