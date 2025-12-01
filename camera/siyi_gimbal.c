// siyi_gimbal.c
// SIYI A8 Mini gimbal control (UDP)
// Usage:
//   1) Auto center:
//      ./siyi_gimbal
//   2) Tracking with bbox (normalized coords):
//      ./siyi_gimbal <ymin> <xmin> <ymax> <xmax>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>

// Gimbal network settings
#define GIMBAL_IP   "192.168.144.25"
#define GIMBAL_PORT 37260

// Control parameters
#define DEADZONE    0.05f
#define KP_X        0.5f
#define KP_Y        0.5f

// CRC16-CCITT (poly = 0x1021, init = 0x0000, MSB-first)
static uint16_t crc16_ccitt(const uint8_t *data, size_t len, uint16_t crc_init)
{
    uint16_t crc = crc_init & 0xFFFF;
    const uint16_t poly = 0x1021;

    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint16_t)(data[i] << 8);
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 0x8000) {
                crc = (uint16_t)((crc << 1) ^ poly);
            } else {
                crc <<= 1;
            }
        }
    }
    return crc & 0xFFFF;
}

static void print_hex(const uint8_t *buf, size_t len)
{
    for (size_t i = 0; i < len; ++i) {
        printf("%02X ", buf[i]);
    }
    printf("\n");
}

// Generic packet builder
static void send_siyi_packet(int sockfd,
                             struct sockaddr_in *addr,
                             uint8_t cmd_id,
                             const uint8_t *data,
                             uint16_t data_len)
{
    uint8_t packet[32];
    size_t idx = 0;

    // Header: STX
    packet[idx++] = 0x55;
    packet[idx++] = 0x66;

    // CTRL
    packet[idx++] = 0x01;

    // DATA_LEN (little endian)
    packet[idx++] = (uint8_t)(data_len & 0xFF);
    packet[idx++] = (uint8_t)((data_len >> 8) & 0xFF);

    // SEQ (0)
    packet[idx++] = 0x00;
    packet[idx++] = 0x00;

    // CMD_ID
    packet[idx++] = cmd_id;

    // DATA
    for (uint16_t i = 0; i < data_len; ++i) {
        packet[idx++] = data[i];
    }

    // CRC16 over all bytes so far
    uint16_t crc = crc16_ccitt(packet, idx, 0x0000);
    packet[idx++] = (uint8_t)(crc & 0xFF);        // low
    packet[idx++] = (uint8_t)((crc >> 8) & 0xFF); // high

    // Send via UDP
    sendto(sockfd, packet, idx, 0, (struct sockaddr *)addr, sizeof(*addr));

    printf("[SIYI TX] cmd=0x%02X len=%zu | ", cmd_id, idx);
    print_hex(packet, idx);
}

// Auto center command (CMD 0x08, DATA 0x01)
static void send_auto_center(int sockfd, struct sockaddr_in *addr)
{
    uint8_t data[1] = {0x01};
    send_siyi_packet(sockfd, addr, 0x08, data, 1);
}

// Speed command (CMD 0x07, DATA = yaw(int8), pitch(int8))
static void send_speed(int sockfd,
                       struct sockaddr_in *addr,
                       int yaw_speed,
                       int pitch_speed)
{
    // Clamp range to [-100, 100]
    if (yaw_speed > 100) yaw_speed = 100;
    if (yaw_speed < -100) yaw_speed = -100;
    if (pitch_speed > 100) pitch_speed = 100;
    if (pitch_speed < -100) pitch_speed = -100;

    int8_t yaw = (int8_t)yaw_speed;
    int8_t pitch = (int8_t)pitch_speed;

    uint8_t data[2];
    data[0] = (uint8_t)yaw;
    data[1] = (uint8_t)pitch;

    printf("[SIYI] Speed command yaw=%d pitch=%d\n", yaw_speed, pitch_speed);
    send_siyi_packet(sockfd, addr, 0x07, data, 2);
}

int main(int argc, char *argv[])
{
    int sockfd;
    struct sockaddr_in servaddr;

    // Create UDP socket
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        perror("socket creation failed");
        return -1;
    }

    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_port   = htons(GIMBAL_PORT);
    servaddr.sin_addr.s_addr = inet_addr(GIMBAL_IP);

    // Case 1: no arguments -> auto center
    if (argc == 1) {
        printf("[INFO] No bbox provided -> auto center.\n");
        send_auto_center(sockfd, &servaddr);
        close(sockfd);
        return 0;
    }

    // Case 2: bbox mode (ymin, xmin, ymax, xmax)
    if (argc != 5) {
        fprintf(stderr,
                "Usage:\n"
                "  %s                 # auto center\n"
                "  %s <ymin> <xmin> <ymax> <xmax>  # tracking (normalized)\n",
                argv[0], argv[0]);
        close(sockfd);
        return -1;
    }

    float ymin = strtof(argv[1], NULL);
    float xmin = strtof(argv[2], NULL);
    float ymax = strtof(argv[3], NULL);
    float xmax = strtof(argv[4], NULL);

    // Compute bbox center in normalized coordinates
    float cx = (xmin + xmax) * 0.5f;
    float cy = (ymin + ymax) * 0.5f;

    // Error from image center (0.5, 0.5)
    float err_x = cx - 0.5f;
    float err_y = cy - 0.5f;

    // Apply deadzone
    if (fabsf(err_x) < DEADZONE) err_x = 0.0f;
    if (fabsf(err_y) < DEADZONE) err_y = 0.0f;

    int yaw_cmd   = (int)(err_x * 100.0f * KP_X);
    int pitch_cmd = (int)(-err_y * 100.0f * KP_Y);  // invert pitch if needed

    printf("[INFO] bbox=(%.3f, %.3f, %.3f, %.3f), center=(%.3f, %.3f), "
           "err=(%.3f, %.3f), cmd=(%d, %d)\n",
           ymin, xmin, ymax, xmax, cx, cy, err_x, err_y, yaw_cmd, pitch_cmd);

    // Send speed command
    send_speed(sockfd, &servaddr, yaw_cmd, pitch_cmd);

    close(sockfd);
    return 0;
}
