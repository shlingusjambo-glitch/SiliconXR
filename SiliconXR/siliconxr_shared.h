// Shared by SiliconXR's OpenXR and OpenVR runtimes (both talk to MacVR through common/vr4mac.h):
// the tracking + hand-joint seqlock read, haptics, finger curls and the lens visibility mask.
#pragma once
#include <dispatch/dispatch.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include "../common/vr4mac.h"

/// Seqlock read of the Mac app's latest tracking sample and hand joints. Returns 1 if the sample changed since *seq.
static inline int sxr_read(VR4Shm *shm, VR4Tracking *t, VR4HandJoints j[2], uint32_t *seq) {
    if (!shm || !t || !j || !seq) return 0;
    for (int tries = 0; tries < 100; tries++) {
        uint32_t s1 = shm->track_seq; vr4_fence();
        if (s1 & 1) continue;
        VR4Tracking tt = shm->track; VR4HandJoints jj[2] = {shm->hand_joints[0], shm->hand_joints[1]}; vr4_fence();
        if (shm->track_seq != s1) continue;
        int fresh = s1 != *seq;
        *t = tt; j[0] = jj[0]; j[1] = jj[1]; *seq = s1;
        return fresh;
    }
    return 0;
}

/// Haptics go through one shm slot that MacVR polls every millisecond, so two pulses written back to back (both hands
/// from one call) used to lose the first. Writes are serialized and spaced 2 ms apart, off the caller's thread.
static inline void sxr_haptic(VR4Shm *shm, int hand, float amp, float dur, float freq, float delay) {
    static dispatch_queue_t q; static dispatch_once_t once; static uint64_t last;
    dispatch_once(&once, ^{ q = dispatch_queue_create("SiliconXR.haptics", DISPATCH_QUEUE_SERIAL); });
    if (!shm || hand < 0 || hand > 1) return;
    if (!(delay >= 0)) delay = 0; else if (delay > 60) delay = 60;   // clamp: int64 ns must not overflow
    if (!(dur >= 0)) dur = 0; else if (dur > 60) dur = 60;
    VR4Haptics h = {(uint8_t)hand, amp < 0 ? 0 : amp > 1 ? 1 : amp, dur, freq > 0 ? freq : 0};
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, delay > 0 ? (int64_t)(delay * 1e9) : 0), q, ^{
        uint64_t now = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        if (last && now - last < 2000000) usleep((useconds_t)((2000000 - (now - last)) / 1000));
        shm->haptic = h; vr4_fence(); shm->haptic_seq++;
        last = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    });
}

/// XR_EXT_hand_tracking joint index of the metacarpal of finger f (0 thumb .. 4 little); the tip is the last joint.
static inline int sxr_meta(int f) { return f ? 1 + 5 * f : 2; }
static inline int sxr_tip(int f) { return f ? 5 + 5 * f : 5; }
static inline float sxr_dist(VR4Pose a, VR4Pose b) {
    float x = a.px - b.px, y = a.py - b.py, z = a.pz - b.pz; return sqrtf(x * x + y * y + z * z);
}
/// Finger curl 0 (straight) .. 1 (folded) from the angle at the proximal joint between the metacarpal and the tip.
static inline float sxr_curl(const VR4HandJoints *j, int f) {
    VR4Pose m = j->joint[sxr_meta(f)], p = j->joint[sxr_meta(f) + 1], t = j->joint[sxr_tip(f)];
    float ax = m.px - p.px, ay = m.py - p.py, az = m.pz - p.pz, bx = t.px - p.px, by = t.py - p.py, bz = t.pz - p.pz;
    float la = sqrtf(ax * ax + ay * ay + az * az), lb = sqrtf(bx * bx + by * by + bz * bz);
    if (la < 1e-5f || lb < 1e-5f) return 1;
    float c = (ax * bx + ay * by + az * bz) / (la * lb);
    return 1 - acosf(c < -1 ? -1 : c > 1 ? 1 : c) / (float)M_PI;
}
/// Pinch strength between the thumb tip and finger f's tip: 1 touching (<= 1 cm), 0 apart (>= 5 cm).
static inline float sxr_pinch(const VR4HandJoints *j, int f) {
    float d = (sxr_dist(j->joint[sxr_tip(0)], j->joint[sxr_tip(f)]) - 0.01f) / 0.04f;
    return d < 0 ? 1 : d > 1 ? 0 : 1 - d;
}

/// Lens visibility mask, in normalized eye coordinates (-1..1 across the field of view, +y up): the visible area is a
/// circle of radius SXR_MASK_R clipped to the eye rectangle, so only the corners the lenses never show are hidden.
/// Ring k is at angle 2*pi*k/SXR_MASK_N (counter-clockwise); out[k] is on the visible boundary, rect[k] on the rectangle.
enum { SXR_MASK_N = 32 };
#define SXR_MASK_R 1.25f
static inline void sxr_mask_ring(float out[SXR_MASK_N][2], float rect[SXR_MASK_N][2]) {
    for (int k = 0; k < SXR_MASK_N; k++) {
        float a = 2 * (float)M_PI * k / SXR_MASK_N, c = cosf(a), s = sinf(a);
        float edge = 1 / fmaxf(fabsf(c), fabsf(s)), r = fminf(SXR_MASK_R, edge);
        out[k][0] = c * r; out[k][1] = s * r; rect[k][0] = c * edge; rect[k][1] = s * edge;
    }
}
/// Triangles (counter-clockwise, 3 points each) of the hidden corners (hidden = 1) or of the visible area (hidden = 0).
/// Returns the triangle count; tri must hold 2 * SXR_MASK_N triangles.
static inline int sxr_mask_triangles(int hidden, float tri[][3][2]) {
    if (!tri) return 0;
    float in[SXR_MASK_N][2], out[SXR_MASK_N][2]; int n = 0;
    sxr_mask_ring(in, out);
    for (int k = 0; k < SXR_MASK_N; k++) {
        int k1 = (k + 1) % SXR_MASK_N;
        const float *t[2][3] = {{in[k], out[k], out[k1]}, {in[k], out[k1], in[k1]}};
        const float zero[2] = {0, 0};
        for (int h = 0; h < (hidden ? 2 : 1); h++) {
            const float *p0 = hidden ? t[h][0] : zero, *p1 = hidden ? t[h][1] : in[k], *p2 = hidden ? t[h][2] : in[k1];
            float area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
            if (area < 1e-6f) continue;   // ring and rectangle meet here: nothing hidden
            for (int v = 0; v < 2; v++) { tri[n][0][v] = p0[v]; tri[n][1][v] = p1[v]; tri[n][2][v] = p2[v]; }
            n++;
        }
    }
    return n;
}
