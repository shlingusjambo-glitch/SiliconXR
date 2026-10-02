// VR4Mac shared definitions: wire protocol packets (Mac <-> Quest) and the
// shared-memory block between the Wine OpenXR runtime and the Mac app.
// Plain C, little-endian, packed. Included by mac/ (bridging header), runtime/ and android/.
#pragma once
#include <stdint.h>

#define VR4_PORT_TCP 9945      // Mac app listens here
#define VR4_PORT_DISCOVERY 9944 // was: Mac broadcast "VR4MAC 9945" here every second (disabled 26112d5; Wi-Fi pairing needs another way)

enum { VR4_HELLO = 1, VR4_CONFIG = 2, VR4_TRACKING = 3, VR4_VIDEO = 4, VR4_HAPTICS = 5, VR4_AUDIO = 6, VR4_REQUEST_IDR = 7, VR4_MIC = 8, VR4_STATUS = 9 };
// VR4_STATUS C->S, about once a second (optional: older Macs ignore unknown types, older clients never send it): UTF-8 JSON,
// every key optional. "battery" headset charge 0-100, "charging" bool, "decode_fps" frames shown per second,
// "latency_ms" average age of the shown frame's pose (pipeline latency left for prediction/timewarp to hide),
// "receive_ms" receive-to-decoder-release, "mbps" video received, "dropped" frames dropped in that second.
// VR4_AUDIO S->C (only if HELLO had "audio": true): u64 time_ns (Mac clock, informational) + interleaved PCM s16le,
// 48000 Hz, 2 channels, 480 frames (10 ms, 1920 bytes) per packet.
// VR4_MIC C->S (only after CONFIG "mic": true; HELLO "mic": true = RECORD_AUDIO granted): u64 time_ns (Quest clock)
// + mono PCM s16le, 48000 Hz, 480 samples (10 ms, 960 bytes) per packet.
enum { VR4_BTN_A = 1, VR4_BTN_B = 2, VR4_BTN_X = 4, VR4_BTN_Y = 8, VR4_BTN_MENU = 16, VR4_BTN_STICK_CLICK = 32,
       VR4_BTN_TRIGGER_TOUCH = 64, VR4_BTN_THUMB_TOUCH = 128, VR4_BTN_STICK_TOUCH = 256 };
enum { VR4_HAND_ACTIVE = 1, VR4_HAND_POSE_VALID = 2, VR4_HAND_TRACKED = 4, VR4_HAND_PINCH_READY = 8 };   // 4/8: a tracked hand (no controller); its pinch is the trigger, 8 = thumb and index poised to pinch
static const uint32_t VR4_MAX_PAYLOAD = 8u << 20;

#pragma pack(push, 1)
typedef struct { float px, py, pz, qx, qy, qz, qw; } VR4Pose;          // OpenXR STAGE space
typedef struct { float left, right, up, down; } VR4Fov;                  // radians, OpenXR XrFovf order
typedef struct { uint32_t flags, buttons; VR4Pose aim, grip; float trigger, squeeze, stick_x, stick_y; } VR4Hand;

// Packet header: [u8 type][u32 len] then payload.
typedef struct { VR4Pose pose; VR4Fov fov; } VR4Eye;
typedef struct {                       // VR4_TRACKING payload, exact wire layout (284 bytes)
    uint64_t time_ns;
    VR4Pose head;
    VR4Eye eye[2];
    VR4Hand hand[2];
} VR4Tracking;

// Hand tracking: while a hand is tracked (controllers put down) the VR4_TRACKING payload is followed by
// VR4HandJoints[2] (left, right): the 26 XR_EXT_hand_tracking joints in OpenXR order, same space as the poses.
enum { VR4_HAND_JOINTS = 26 };
typedef struct { uint32_t tracked; VR4Pose joint[VR4_HAND_JOINTS]; } VR4HandJoints;

typedef struct { uint64_t frame_id, time_ns; uint8_t flags; } VR4VideoHeader;  // followed by Annex-B NALs
typedef struct { uint8_t hand; float amplitude, duration_s, frequency_hz; } VR4Haptics;
#pragma pack(pop)

// ---------------- shared memory: Wine runtime <-> Mac app ----------------
// File /tmp/vr4mac/shm (Wine: Z:\tmp\vr4mac\shm), mmap'd by both sides.
#define VR4_SHM_PATH_MAC "/tmp/vr4mac/shm"
#define VR4_SHM_PATH_WIN "Z:\\tmp\\vr4mac\\shm"
static const uint32_t VR4_SHM_MAGIC = 0x3452564Du, VR4_SHM_VERSION = 4;
static const uint32_t VR4_FRAME_OFFSET = 4096u;
static const uint32_t VR4_FRAME_MAX = 2u * 2048u * 2048u * 4u;   // one side-by-side BGRA frame
static const uint32_t VR4_SHM_SIZE = 4096u + 2u * VR4_FRAME_MAX;

typedef struct {
    uint32_t magic, version;
    // Mac app -> runtime
    uint32_t eye_w, eye_h;             // per-eye size the game should render
    float fps;
    uint32_t client_connected;
    volatile uint32_t track_seq;       // seqlock: odd while writing
    VR4Tracking track;
    volatile uint32_t input_blocked;   // 1 while the dashboard is open (game sees no input)
    float render_scale, world_scale;   // Mac -> runtime per-app overrides (0 = default 1.0)
    // runtime -> Mac app
    volatile uint32_t haptic_seq;
    VR4Haptics haptic;
    volatile uint64_t runtime_heartbeat_ns; // runtime's last xrEndFrame (Mac CLOCK_MONOTONIC via Wine)
    char app_name[128];
    volatile uint32_t frame_seq;       // incremented after frame buffer [frame_seq % 2] is complete
    uint64_t frame_time_ns[2];         // track.time_ns the frame was rendered for
    VR4Pose frame_eye_pose[2][2];      // [buffer][eye] poses used to render (sent back for timewarp)
    uint32_t frame_w[2], frame_h[2];   // full side-by-side size of each buffer
    uint32_t frame_rgba[2];            // 1 = buffer holds RGBA (the Mac swaps to BGRA with vImage; keeps it off the game thread)
    // Mac app -> runtime (v4): hand-tracking joints, written under the same track_seq seqlock as `track`.
    // hand_joints[h].tracked = 0 while that hand holds a controller (or isn't seen).
    VR4HandJoints hand_joints[2];
} VR4Shm;

static inline void vr4_fence(void) { __atomic_thread_fence(__ATOMIC_SEQ_CST); }
static inline uint8_t *vr4_frame(VR4Shm *s, uint32_t i) { return (uint8_t *)s + VR4_FRAME_OFFSET + (i % 2) * VR4_FRAME_MAX; }
