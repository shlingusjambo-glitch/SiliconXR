// MacVR native OpenVR runtime: libopenvr_api.dylib for macOS apps that speak OpenVR directly (Vivecraft via
// LWJGL). Implements the "FnTable:" C interfaces LWJGL binds to, backed by the MacVR shared memory
// (common/vr4mac.h): poses + buttons come from the Mac app, OpenGL eye textures are read back into the
// side-by-side frame buffers the Mac app streams to the headset. Input follows the app's SteamVR action
// manifest and its oculus_touch default bindings.
#define GL_SILENCE_DEPRECATION
#import <Foundation/Foundation.h>
#import <AppKit/AppKit.h>
#include <OpenGL/gl3.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>
#include "openvr_capi.h"
#include <OpenGL/OpenGL.h>
#include "siliconxr_shared.h"

#define EXPORT __attribute__((visibility("default")))

_Static_assert(sizeof(InputDigitalActionData_t) == 24 && sizeof(InputAnalogActionData_t) == 48, "LWJGL layout");
_Static_assert(sizeof(InputPoseActionData_t) == 96 && sizeof(TrackedDevicePose_t) == 80, "LWJGL layout");
_Static_assert(sizeof(InputOriginInfo_t) == 144 && sizeof(VRActiveActionSet_t) == 32, "LWJGL layout");
_Static_assert(sizeof(InputSkeletalActionData_t) == 16 && sizeof(VRBoneTransform_t) == 32 && sizeof(VRSkeletalSummaryData_t) == 36, "LWJGL layout");

static VR4Shm *shm;
static VR4Tracking track;              // latest snapshot (WaitGetPoses / UpdateActionState)
static VR4HandJoints joints[2];        // hand-tracking joints of the same snapshot
static uint32_t lastSeq;
static ETrackingUniverseOrigin trackingSpace = ETrackingUniverseOrigin_TrackingUniverseStanding;

static void logmsg(const char *fmt, ...) {
    static FILE *f;
    if (!f) f = fopen("/tmp/vr4mac/openvr.log", "a");
    if (!f) return;
    va_list a; va_start(a, fmt); vfprintf(f, fmt, a); va_end(a); fputc('\n', f); fflush(f);
}
static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static float fps(void) { return shm && shm->fps > 0 ? shm->fps : 72; }
static float world_scale(void) { return shm && shm->world_scale > 0.01f ? shm->world_scale : 1; }

static void read_tracking(void) { sxr_read(shm, &track, joints, &lastSeq); }   // seqlock read of the Mac app's latest sample

// ---------------------------------------------------------------- math
static HmdMatrix34_t mat(VR4Pose p) {
    float x = p.qx, y = p.qy, z = p.qz, w = p.qw, ws = world_scale();
    if (x == 0 && y == 0 && z == 0 && w == 0) w = 1;
    return (HmdMatrix34_t){{
        {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), p.px / ws},
        {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), p.py / ws},
        {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), p.pz / ws}}};
}
static HmdMatrix34_t mul34(HmdMatrix34_t a, HmdMatrix34_t b) {   // a * b, both rigid
    HmdMatrix34_t r;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 4; j++)
        r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j] + (j == 3 ? a.m[i][3] : 0);
    return r;
}
static HmdMatrix34_t inv34(HmdMatrix34_t a) {
    HmdMatrix34_t r;
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) r.m[i][j] = a.m[j][i];
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * a.m[0][3] + r.m[i][1] * a.m[1][3] + r.m[i][2] * a.m[2][3]);
    return r;
}

typedef struct { float x, y, z, w; } Quat;
static inline Quat q_mul(Quat a, Quat b) {
    return (Quat){a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                  a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
                  a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
                  a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
static inline Quat q_conj(Quat a) { return (Quat){-a.x, -a.y, -a.z, a.w}; }
static inline Quat device_quat(int i) {
    VR4Pose p = i == 0 ? track.head : track.hand[i - 1].grip;
    float w = p.qw;
    if (p.qx == 0 && p.qy == 0 && p.qz == 0 && w == 0) w = 1.0f;
    return (Quat){p.qx, p.qy, p.qz, w};
}

// ---------------------------------------------------------------- devices: 0 = HMD, 1 = left hand, 2 = right hand
static int hand_valid(int h) { return (track.hand[h].flags & VR4_HAND_POSE_VALID) != 0; }
static TrackedDevicePose_t prevPose[3]; static Quat prevQuat[3]; static uint64_t prevPoseT;
static TrackedDevicePose_t device_pose(int i) {
    TrackedDevicePose_t p = {0};
    if (i < 0 || i > 2) return p;
    p.bDeviceIsConnected = 1;
    p.bPoseIsValid = i == 0 || hand_valid(i - 1);
    p.eTrackingResult = p.bPoseIsValid ? ETrackingResult_TrackingResult_Running_OK : ETrackingResult_TrackingResult_Running_OutOfRange;
    p.mDeviceToAbsoluteTracking = mat(i == 0 ? track.head : track.hand[i - 1].grip);
    return p;
}
static void fill_velocities(TrackedDevicePose_t *p, int n) {   // finite difference against the previous sample
    double dt = prevPoseT && track.time_ns > prevPoseT ? (double)(track.time_ns - prevPoseT) * 1e-9 : 0;
    for (int i = 0; i < n && i < 3; i++) {
        Quat curQ = device_quat(i);
        if (dt >= 0.002 && dt <= 0.2) {
            float vx = (p[i].mDeviceToAbsoluteTracking.m[0][3] - prevPose[i].mDeviceToAbsoluteTracking.m[0][3]) / (float)dt;
            float vy = (p[i].mDeviceToAbsoluteTracking.m[1][3] - prevPose[i].mDeviceToAbsoluteTracking.m[1][3]) / (float)dt;
            float vz = (p[i].mDeviceToAbsoluteTracking.m[2][3] - prevPose[i].mDeviceToAbsoluteTracking.m[2][3]) / (float)dt;
            float vMag = sqrtf(vx * vx + vy * vy + vz * vz);
            if (vMag > 40.0f) { float s = 40.0f / vMag; vx *= s; vy *= s; vz *= s; }
            if (prevPose[i].vVelocity.v[0] != 0 || prevPose[i].vVelocity.v[1] != 0 || prevPose[i].vVelocity.v[2] != 0) {
                p[i].vVelocity.v[0] = 0.75f * vx + 0.25f * prevPose[i].vVelocity.v[0];
                p[i].vVelocity.v[1] = 0.75f * vy + 0.25f * prevPose[i].vVelocity.v[1];
                p[i].vVelocity.v[2] = 0.75f * vz + 0.25f * prevPose[i].vVelocity.v[2];
            } else {
                p[i].vVelocity.v[0] = vx; p[i].vVelocity.v[1] = vy; p[i].vVelocity.v[2] = vz;
            }

            Quat qrel = q_mul(curQ, q_conj(prevQuat[i]));
            if (qrel.w < 0.0f) qrel = (Quat){-qrel.x, -qrel.y, -qrel.z, -qrel.w};
            if (qrel.w > 1.0f) qrel.w = 1.0f;
            float angle = 2.0f * acosf(qrel.w);
            float sinHalf = sqrtf(fmaxf(0.0f, 1.0f - qrel.w * qrel.w));
            float wx = 0, wy = 0, wz = 0;
            if (sinHalf > 1e-4f && angle > 1e-4f) {
                float factor = (angle / (float)dt) / sinHalf;
                wx = qrel.x * factor; wy = qrel.y * factor; wz = qrel.z * factor;
            }
            float wMag = sqrtf(wx * wx + wy * wy + wz * wz);
            if (wMag > 100.0f) { float s = 100.0f / wMag; wx *= s; wy *= s; wz *= s; }
            if (prevPose[i].vAngularVelocity.v[0] != 0 || prevPose[i].vAngularVelocity.v[1] != 0 || prevPose[i].vAngularVelocity.v[2] != 0) {
                p[i].vAngularVelocity.v[0] = 0.75f * wx + 0.25f * prevPose[i].vAngularVelocity.v[0];
                p[i].vAngularVelocity.v[1] = 0.75f * wy + 0.25f * prevPose[i].vAngularVelocity.v[1];
                p[i].vAngularVelocity.v[2] = 0.75f * wz + 0.25f * prevPose[i].vAngularVelocity.v[2];
            } else {
                p[i].vAngularVelocity.v[0] = wx; p[i].vAngularVelocity.v[1] = wy; p[i].vAngularVelocity.v[2] = wz;
            }
        } else {
            p[i].vVelocity = prevPose[i].vVelocity;
            p[i].vAngularVelocity = prevPose[i].vAngularVelocity;
        }
    }
    if (track.time_ns != prevPoseT) {
        for (int i = 0; i < n && i < 3; i++) {
            prevPose[i] = p[i];
            prevQuat[i] = device_quat(i);
        }
        prevPoseT = track.time_ns;
    }
}

// ---------------------------------------------------------------- input: manifest + oculus_touch bindings
enum { MODE_BUTTON, MODE_TRIGGER, MODE_JOYSTICK, MODE_TOGGLE, MODE_SCROLL };
typedef struct { char name[128]; char type[16]; int hand, isAim; float x, y, px, py; int state, prevState, origin; } Action;   // hand: pose/haptic actions
typedef struct { int set, action, hand, mode; char comp[24], input[16]; } Binding;
typedef struct { int set, action, hand[2]; char comp[2][24]; } Chord;
static Action actions[512]; static int nactions;
static char sets[64][128]; static int nsets;
static Binding binds[1024]; static int nbinds;
static Chord chords[64]; static int nchords;
static uint64_t activeSets;            // bitmask of set indices from the last UpdateActionState
static int toggled[2][16];             // toggle_button latch per hand/component slot
static uint64_t pressStart[2][16];     // long-press timers per hand/component slot
static char paths[256][128]; static int npaths = 2;   // input source handles; 1/2 = left/right hand

static int find_ci(char (*list)[128], int n, const char *s) {
    for (int i = 0; i < n; i++) if (!strcasecmp(list[i], s)) return i;
    return -1;
}
static int action_index(const char *name) {
    for (int i = 0; i < nactions; i++) if (!strcasecmp(actions[i].name, name)) return i;
    return -1;
}
static int add_action(const char *name, const char *type) {
    int i = action_index(name);
    if (i >= 0 || nactions == 512) return i;
    snprintf(actions[nactions].name, 128, "%s", name); snprintf(actions[nactions].type, 16, "%s", type ? type : "boolean");
    actions[nactions].hand = -1;
    return nactions++;
}
static int set_index(const char *name) {
    int i = find_ci(sets, nsets, name);
    if (i >= 0 || nsets == 64) return i;
    snprintf(sets[nsets], 128, "%s", name);
    return nsets++;
}
static int hand_of(NSString *path) { return [path hasPrefix:@"/user/hand/left"] ? 0 : [path hasPrefix:@"/user/hand/right"] ? 1 : -1; }
static NSString *comp_of(NSString *path) { return [[path componentsSeparatedByString:@"/"] objectAtIndex:5]; }   // /user/hand/X/input/<comp>

static void load_bindings(NSString *file) {
    NSDictionary *root = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:file] ?: [NSData data] options:0 error:nil];
    NSDictionary *bindings = [root isKindOfClass:NSDictionary.class] ? root[@"bindings"] : nil;
    if (![bindings isKindOfClass:NSDictionary.class]) { logmsg("no bindings in %s", file.UTF8String); return; }
    for (NSString *setName in bindings) {
        int set = set_index(setName.UTF8String);
        NSDictionary *s = bindings[setName];
        for (NSDictionary *p in s[@"poses"]) {
            int a = add_action([p[@"output"] UTF8String], "pose");
            if (a >= 0) {
                actions[a].hand = hand_of(p[@"path"]);
                NSString *path = p[@"path"];
                actions[a].isAim = path && [path rangeOfString:@"/aim/" options:NSCaseInsensitiveSearch].location != NSNotFound;
            }
        }
        for (NSDictionary *p in s[@"haptics"]) {
            int a = add_action([p[@"output"] UTF8String], "vibration");
            if (a >= 0) actions[a].hand = hand_of(p[@"path"]);
        }
        for (NSDictionary *p in s[@"skeleton"]) {   // {"output": ..., "path": "/user/hand/left/input/skeleton/left"}
            int a = add_action([p[@"output"] UTF8String], "skeleton");
            if (a >= 0) actions[a].hand = hand_of(p[@"path"]);
        }
        for (NSDictionary *src in s[@"sources"]) {
            NSString *path = src[@"path"], *mode = src[@"mode"];
            NSDictionary *inputs = src[@"inputs"];
            int hand = hand_of(path);
            if (hand < 0 || [path componentsSeparatedByString:@"/"].count < 6 || ![inputs isKindOfClass:NSDictionary.class]) continue;
            int m = [mode isEqual:@"trigger"] ? MODE_TRIGGER : [mode isEqual:@"joystick"] || [mode isEqual:@"trackpad"] ? MODE_JOYSTICK
                  : [mode isEqual:@"toggle_button"] ? MODE_TOGGLE : [mode isEqual:@"scroll"] ? MODE_SCROLL : MODE_BUTTON;
            for (NSString *in in inputs) {
                NSString *out = inputs[in][@"output"];
                int a = out ? add_action(out.UTF8String, NULL) : -1;
                if (a < 0 || nbinds == 1024) continue;
                Binding *b = &binds[nbinds++];
                *b = (Binding){.set = set, .action = a, .hand = hand, .mode = m};
                snprintf(b->comp, 24, "%s", comp_of(path).UTF8String); snprintf(b->input, 16, "%s", in.UTF8String);
            }
        }
        for (NSDictionary *c in s[@"chords"]) {
            NSArray *ins = c[@"inputs"];
            int a = add_action([c[@"output"] UTF8String], NULL);
            if (a < 0 || ins.count != 2 || nchords == 64) continue;
            Chord *ch = &chords[nchords++];
            ch->set = set; ch->action = a;
            for (int k = 0; k < 2; k++) { ch->hand[k] = hand_of(ins[k][0]); snprintf(ch->comp[k], 24, "%s", comp_of(ins[k][0]).UTF8String); }
        }
    }
    logmsg("bindings %s: %d sources, %d chords, %d actions", file.lastPathComponent.UTF8String, nbinds, nchords, nactions);
}

static EVRInputError SetActionManifestPath(char *path) {
    @autoreleasepool {
        NSString *p = [NSString stringWithUTF8String:path];
        NSDictionary *m = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:p] ?: [NSData data] options:0 error:nil];
        if (![m isKindOfClass:NSDictionary.class]) { logmsg("bad action manifest %s", path); return EVRInputError_VRInputError_InvalidParam; }
        for (NSDictionary *s in m[@"action_sets"]) set_index([s[@"name"] UTF8String]);
        for (NSDictionary *a in m[@"actions"]) {
            int i = add_action([a[@"name"] UTF8String], [a[@"type"] UTF8String]);
            if (i >= 0) snprintf(actions[i].type, 16, "%s", [a[@"type"] UTF8String]);
            if (i >= 0 && [a[@"skeleton"] isKindOfClass:NSString.class]) actions[i].hand = [a[@"skeleton"] hasSuffix:@"right"] ? 1 : 0;   // "/skeleton/hand/left"
        }
        NSString *file = nil;
        for (NSDictionary *d in m[@"default_bindings"])
            if ([d[@"controller_type"] isEqual:@"oculus_touch"] || !file) file = d[@"binding_url"];
        if (file) load_bindings([[p stringByDeletingLastPathComponent] stringByAppendingPathComponent:file]);
    }
    return EVRInputError_VRInputError_None;
}

/// Current value of a controller component (trigger, grip, a, b, x, y, joystick, ...).
static int comp_slot(const char *c) {
    static const char *names[] = {"trigger", "grip", "a", "b", "x", "y", "joystick", "thumbstick", "system", "application_menu", "thumbrest"};
    for (int i = 0; i < 11; i++) if (!strcmp(c, names[i])) return i;
    return 15;
}
static float comp_pull(const VR4Hand *h, const char *c) {
    if (!strcmp(c, "trigger")) return h->trigger;
    if (!strcmp(c, "grip")) return h->squeeze;
    return 0;
}
static int comp_click(int hand, const char *c) {
    const VR4Hand *h = &track.hand[hand];
    static int held[2][16];                     // hysteresis for analog clicks
    int slot = comp_slot(c);
    if (!strcmp(c, "trigger") || !strcmp(c, "grip")) {
        float v = comp_pull(h, c);
        held[hand][slot] = held[hand][slot] ? v > 0.6f : v > 0.75f;
        return held[hand][slot];
    }
    uint32_t bit = !strcmp(c, "a") ? VR4_BTN_A : !strcmp(c, "b") ? VR4_BTN_B : !strcmp(c, "x") ? VR4_BTN_X : !strcmp(c, "y") ? VR4_BTN_Y
                 : !strcmp(c, "joystick") || !strcmp(c, "thumbstick") ? VR4_BTN_STICK_CLICK
                 : !strcmp(c, "system") || !strcmp(c, "application_menu") ? VR4_BTN_MENU : 0;
    return (h->buttons & bit) != 0;
}
static int comp_touch(int hand, const char *c) {
    uint32_t b = track.hand[hand].buttons;
    if (!strcmp(c, "trigger")) return (b & VR4_BTN_TRIGGER_TOUCH) != 0;
    if (!strcmp(c, "joystick") || !strcmp(c, "thumbstick")) return (b & VR4_BTN_STICK_TOUCH) != 0;
    if (!strcmp(c, "thumbrest") || !strcmp(c, "a") || !strcmp(c, "b") || !strcmp(c, "x") || !strcmp(c, "y")) return (b & VR4_BTN_THUMB_TOUCH) != 0;
    return comp_click(hand, c);
}

static EVRInputError UpdateActionState(VRActiveActionSet_t *s, uint32_t size, uint32_t n) {
    read_tracking();
    activeSets = 0;
    for (uint32_t i = 0; i < n; i++) {
        const VRActiveActionSet_t *a = (const VRActiveActionSet_t *)((const char *)s + (size_t)i * size);
        if (a->ulActionSet >= 1 && a->ulActionSet <= 64) activeSets |= 1ull << (a->ulActionSet - 1);
    }
    for (int i = 0; i < nactions; i++) {
        Action *a = &actions[i];
        a->prevState = a->state; a->px = a->x; a->py = a->y;
        a->state = 0; a->x = a->y = 0; a->origin = 0;
    }
    int blocked = shm && shm->input_blocked;
    uint64_t t = now_ns();
    int clickNow[2][16] = {{0}};
    for (int h = 0; h < 2; h++) for (int k = 0; k < 16; k++) clickNow[h][k] = -1;
    for (int i = 0; !blocked && i < nbinds; i++) {
        Binding *b = &binds[i];
        if (!(activeSets >> b->set & 1)) continue;
        Action *a = &actions[b->action];
        const VR4Hand *h = &track.hand[b->hand];
        int slot = comp_slot(b->comp), on = 0;
        float x = 0, y = 0;
        if (!strcmp(b->input, "position") || !strcmp(b->input, "scroll")) {
            x = h->stick_x; y = h->stick_y;
            if (b->mode == MODE_SCROLL) {   // discrete scroll: one tick per push, repeating every 200 ms while held
                static uint64_t next[2];
                float v = fabsf(y) > 0.6f ? (y > 0 ? 1 : -1) : 0;
                if (!v) next[b->hand] = 0;
                else if (t >= next[b->hand]) next[b->hand] = t + (next[b->hand] ? 200000000ull : 400000000ull);
                else v = 0;
                x = 0; y = v;
            }
        } else if (!strcmp(b->input, "pull")) {
            x = comp_pull(h, b->comp);
        } else if (!strcmp(b->input, "touch")) {
            on = comp_touch(b->hand, b->comp);
        } else {                            // click / long / held
            int c = clickNow[b->hand][slot] >= 0 ? clickNow[b->hand][slot] : (clickNow[b->hand][slot] = comp_click(b->hand, b->comp));
            if (b->mode == MODE_TOGGLE) {
                static int last[2][16];
                if (c && !last[b->hand][slot]) toggled[b->hand][slot] ^= 1;
                last[b->hand][slot] = c;
                on = toggled[b->hand][slot];
            } else if (!strcmp(b->input, "long")) {
                on = c && pressStart[b->hand][slot] && t - pressStart[b->hand][slot] > 500000000ull;
            } else on = c;
        }
        if (on || fabsf(x) > fabsf(a->x) || fabsf(y) > fabsf(a->y)) a->origin = b->hand + 1;
        a->state |= on;
        if (fabsf(x) > fabsf(a->x)) a->x = x;
        if (fabsf(y) > fabsf(a->y)) a->y = y;
    }
    for (int h = 0; h < 2; h++) for (int k = 0; k < 16; k++)
        if (clickNow[h][k] >= 0) pressStart[h][k] = clickNow[h][k] ? (pressStart[h][k] ? pressStart[h][k] : t) : 0;
    for (int i = 0; !blocked && i < nchords; i++) {
        Chord *c = &chords[i];
        if (!(activeSets >> c->set & 1) || c->hand[0] < 0 || c->hand[1] < 0) continue;
        if (comp_click(c->hand[0], c->comp[0]) && comp_click(c->hand[1], c->comp[1])) { actions[c->action].state = 1; actions[c->action].origin = c->hand[0] + 1; }
    }
    return EVRInputError_VRInputError_None;
}

static Action *action(VRActionHandle_t h) { return h >= 1 && h <= (VRActionHandle_t)nactions ? &actions[h - 1] : NULL; }
static EVRInputError GetActionSetHandle(char *name, VRActionSetHandle_t *h) {
    int i = set_index(name);
    *h = i < 0 ? 0 : (VRActionSetHandle_t)i + 1;
    return i < 0 ? EVRInputError_VRInputError_MaxCapacityReached : EVRInputError_VRInputError_None;
}
static EVRInputError GetActionHandle(char *name, VRActionHandle_t *h) {
    int i = add_action(name, NULL);
    *h = i < 0 ? 0 : (VRActionHandle_t)i + 1;
    return i < 0 ? EVRInputError_VRInputError_MaxCapacityReached : EVRInputError_VRInputError_None;
}
static EVRInputError GetInputSourceHandle(char *path, VRInputValueHandle_t *h) {
    if (!strcasecmp(path, "/user/hand/left")) { *h = 1; return 0; }
    if (!strcasecmp(path, "/user/hand/right")) { *h = 2; return 0; }
    int i = find_ci(paths, npaths, path);
    if (i < 0 && npaths < 256) snprintf(paths[i = npaths++], 128, "%s", path);
    *h = i < 0 ? 0 : (VRInputValueHandle_t)i + 1;
    return EVRInputError_VRInputError_None;
}
static int restrict_ok(Action *a, VRInputValueHandle_t r) { return !r || (VRInputValueHandle_t)a->origin == r; }
static EVRInputError GetDigitalActionData(VRActionHandle_t h, InputDigitalActionData_t *d, uint32_t size, VRInputValueHandle_t r) {
    Action *a = action(h);
    if (!a) return EVRInputError_VRInputError_InvalidHandle;
    memset(d, 0, size);
    int st = restrict_ok(a, r) && a->state;
    d->bActive = 1; d->activeOrigin = a->origin; d->bState = st; d->bChanged = st != a->prevState;
    return EVRInputError_VRInputError_None;
}
static EVRInputError GetAnalogActionData(VRActionHandle_t h, InputAnalogActionData_t *d, uint32_t size, VRInputValueHandle_t r) {
    Action *a = action(h);
    if (!a) return EVRInputError_VRInputError_InvalidHandle;
    memset(d, 0, size);
    if (!restrict_ok(a, r)) { d->bActive = 1; return 0; }
    int scroll = 0;
    for (int i = 0; i < nbinds; i++) if (binds[i].action == (int)h - 1 && binds[i].mode == MODE_SCROLL) scroll = 1;
    d->bActive = 1; d->activeOrigin = a->origin; d->x = a->x; d->y = a->y;
    d->deltaX = scroll ? a->x : a->x - a->px; d->deltaY = scroll ? a->y : a->y - a->py;   // scroll: ticks this update
    return EVRInputError_VRInputError_None;
}
static EVRInputError GetPoseActionDataForNextFrame(VRActionHandle_t h, ETrackingUniverseOrigin o, InputPoseActionData_t *d, uint32_t size, VRInputValueHandle_t r) {
    (void)o; (void)r;
    Action *a = action(h);
    if (!a) return EVRInputError_VRInputError_InvalidHandle;
    memset(d, 0, size);
    if (a->hand < 0) return EVRInputError_VRInputError_None;
    d->bActive = 1; d->activeOrigin = (VRInputValueHandle_t)a->hand + 1;
    TrackedDevicePose_t p[3] = {device_pose(0), device_pose(1), device_pose(2)};
    fill_velocities(p, 3);
    d->pose = p[a->hand + 1];
    if (a->isAim && hand_valid(a->hand)) {
        d->pose.mDeviceToAbsoluteTracking = mat(track.hand[a->hand].aim);
    }
    return EVRInputError_VRInputError_None;
}
static EVRInputError GetPoseActionDataRelativeToNow(VRActionHandle_t h, ETrackingUniverseOrigin o, float t, InputPoseActionData_t *d, uint32_t size, VRInputValueHandle_t r) {
    (void)t; return GetPoseActionDataForNextFrame(h, o, d, size, r);   // poses are already predicted to the display time
}
static EVRInputError GetOriginTrackedDeviceInfo(VRInputValueHandle_t origin, InputOriginInfo_t *info, uint32_t size) {
    memset(info, 0, size);
    if (origin != 1 && origin != 2) return EVRInputError_VRInputError_InvalidHandle;
    info->devicePath = origin; info->trackedDeviceIndex = (TrackedDeviceIndex_t)origin;
    return EVRInputError_VRInputError_None;
}
static EVRInputError GetActionOrigins(VRActionSetHandle_t set, VRActionHandle_t h, VRInputValueHandle_t *out, uint32_t n) {
    memset(out, 0, n * sizeof *out);
    uint32_t k = 0; int seen[2] = {0};
    for (int i = 0; i < nbinds && k < n; i++)
        if (binds[i].action == (int)h - 1 && binds[i].set == (int)set - 1 && !seen[binds[i].hand]) { seen[binds[i].hand] = 1; out[k++] = (VRInputValueHandle_t)binds[i].hand + 1; }
    return EVRInputError_VRInputError_None;
}
static EVRInputError GetOriginLocalizedName(VRInputValueHandle_t origin, char *name, uint32_t size, int32_t sections) {
    (void)sections;
    snprintf(name, size, "%s", origin == 1 ? "Left Hand" : origin == 2 ? "Right Hand" : "Unknown");
    return EVRInputError_VRInputError_None;
}
static EVRInputError TriggerHapticVibrationAction(VRActionHandle_t h, float start, float dur, float freq, float amp, VRInputValueHandle_t r) {
    Action *a = action(h);
    if (!a) return EVRInputError_VRInputError_InvalidHandle;
    int hand = a->hand >= 0 ? a->hand : r == 1 ? 0 : r == 2 ? 1 : -1;
    sxr_haptic(shm, hand, amp, dur > 0 ? dur : 0.02f, freq, start);   // start: seconds from now
    return EVRInputError_VRInputError_None;
}

// ---------------------------------------------------------------- skeletal input (SteamVR's 31-bone hand)
// Bones come from MacVR's hand-tracking joints while a hand is tracked, else from a hand posed by the controller's
// trigger, grip and touch sensors. A SteamVR bone frame is the OpenXR joint frame times a constant basis change
// (+X along the finger on the left hand, -X on the right; the wrist adds a twist), the same mapping ALVR uses.
// Root = the hand's pose action (the grip), as SteamVR's root sits on the controller pose.
typedef struct { float x, y, z; } V3;
typedef struct { Quat q; V3 p; } Xf;
static V3 q_rot(Quat q, V3 v) { Quat r = q_mul(q_mul(q, (Quat){v.x, v.y, v.z, 0}), q_conj(q)); return (V3){r.x, r.y, r.z}; }
static Xf xf_mul(Xf a, Xf b) { V3 t = q_rot(a.q, b.p); return (Xf){q_mul(a.q, b.q), {a.p.x + t.x, a.p.y + t.y, a.p.z + t.z}}; }
static Xf xf_inv(Xf a) { Quat c = q_conj(a.q); V3 t = q_rot(c, a.p); return (Xf){c, {-t.x, -t.y, -t.z}}; }
static Xf xf_of(VR4Pose p) { return (Xf){{p.qx, p.qy, p.qz, p.qx || p.qy || p.qz || p.qw ? p.qw : 1}, {p.px, p.py, p.pz}}; }
static Quat q_axis(float ax, float ay, float az, float angle) { float s = sinf(angle / 2); return (Quat){ax * s, ay * s, az * s, cosf(angle / 2)}; }
/// Rotation whose -Z points along `dir` and whose +Y is as close to `up` as possible.
static Quat q_look(V3 dir, V3 up) {
    float l = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    V3 z = {-dir.x / l, -dir.y / l, -dir.z / l};
    V3 x = {up.y * z.z - up.z * z.y, up.z * z.x - up.x * z.z, up.x * z.y - up.y * z.x};
    l = sqrtf(x.x * x.x + x.y * x.y + x.z * x.z); x = (V3){x.x / l, x.y / l, x.z / l};
    V3 y = {z.y * x.z - z.z * x.y, z.z * x.x - z.x * x.z, z.x * x.y - z.y * x.x};
    float t = x.x + y.y + z.z, s;
    if (t > 0) { s = sqrtf(t + 1) * 2; return (Quat){(y.z - z.y) / s, (z.x - x.z) / s, (x.y - y.x) / s, s / 4}; }
    if (x.x > y.y && x.x > z.z) { s = sqrtf(1 + x.x - y.y - z.z) * 2; return (Quat){s / 4, (y.x + x.y) / s, (z.x + x.z) / s, (y.z - z.y) / s}; }
    if (y.y > z.z) { s = sqrtf(1 + y.y - x.x - z.z) * 2; return (Quat){(y.x + x.y) / s, s / 4, (z.y + y.z) / s, (z.x - x.z) / s}; }
    s = sqrtf(1 + z.z - x.x - y.y) * 2; return (Quat){(z.x + x.z) / s, (z.y + y.z) / s, s / 4, (x.y - y.x) / s};
}
enum { NBONES = 31 };
static const BoneIndex_t boneParent[NBONES] = {-1, 0, 1, 2, 3, 4, 1, 6, 7, 8, 9, 1, 11, 12, 13, 14, 1, 16, 17, 18, 19, 1, 21, 22, 23, 24, 0, 0, 0, 0, 0};
static const char *boneName[NBONES] = {"Root", "wrist_?", "finger_thumb_0_?", "finger_thumb_1_?", "finger_thumb_2_?", "finger_thumb_?_end",
    "finger_index_meta_?", "finger_index_0_?", "finger_index_1_?", "finger_index_2_?", "finger_index_?_end",
    "finger_middle_meta_?", "finger_middle_0_?", "finger_middle_1_?", "finger_middle_2_?", "finger_middle_?_end",
    "finger_ring_meta_?", "finger_ring_0_?", "finger_ring_1_?", "finger_ring_2_?", "finger_ring_?_end",
    "finger_pinky_meta_?", "finger_pinky_0_?", "finger_pinky_1_?", "finger_pinky_2_?", "finger_pinky_?_end",
    "finger_thumb_?_aux", "finger_index_?_aux", "finger_middle_?_aux", "finger_ring_?_aux", "finger_pinky_?_aux"};

/// A left hand (OpenXR joints, grip space) holding a controller, fingers curled by c[0..4] (0 open .. 1 fist).
/// The right hand is its mirror image across the grip's YZ plane.
static void synth_joints(const float c[5], int right, Xf j[26]) {
    static const float base[5][3] = {{0.022f, -0.012f, 0.032f}, {0.010f, 0, 0.040f}, {0.002f, 0, 0.040f}, {-0.008f, 0, 0.038f}, {-0.016f, -0.002f, 0.034f}};
    static const float knuckle[5][3] = {{0.040f, -0.020f, 0.002f}, {0.024f, 0.004f, -0.030f}, {0.004f, 0.006f, -0.034f}, {-0.016f, 0.004f, -0.030f}, {-0.034f, 0, -0.022f}};
    static const float len[5][3] = {{0.032f, 0.028f, 0}, {0.040f, 0.024f, 0.022f}, {0.044f, 0.028f, 0.024f}, {0.040f, 0.026f, 0.023f}, {0.032f, 0.019f, 0.020f}};
    static const float flex[5][3] = {{0.7f, 1.0f, 0}, {1.4f, 1.75f, 1.2f}, {1.4f, 1.75f, 1.2f}, {1.4f, 1.75f, 1.2f}, {1.4f, 1.75f, 1.2f}};
    Xf pj[26];   // palm space: palm at the origin, fingers toward -Z, back of the hand +Y, thumb toward +X
    pj[0] = (Xf){{0, 0, 0, 1}, {0, 0, 0}};
    pj[1] = (Xf){{0, 0, 0, 1}, {0, 0, 0.05f}};
    for (int f = 0; f < 5; f++) {
        int m = sxr_meta(f), n = sxr_tip(f) - m;   // joints after the metacarpal: thumb 3, fingers 4
        V3 b = {base[f][0], base[f][1], base[f][2]}, k = {knuckle[f][0], knuckle[f][1], knuckle[f][2]};
        Quat o = q_look((V3){k.x - b.x, k.y - b.y, k.z - b.z}, (V3){0, 1, 0});
        if (!f) o = q_mul(o, q_axis(0, 1, 0, 0.5f * c[0]));   // the thumb swings across the palm
        pj[m] = (Xf){o, b};
        V3 p = k;
        for (int s = 0; s < n; s++) {
            if (s < 3) o = q_mul(o, q_axis(1, 0, 0, -flex[f][s] * c[f]));   // curl toward the palm (-Y)
            pj[m + 1 + s] = (Xf){o, p};
            if (s < n - 1) { V3 d = q_rot(o, (V3){0, 0, -len[f][s]}); p = (V3){p.x + d.x, p.y + d.y, p.z + d.z}; }
        }
    }
    // palm in grip space: facing +X (into the handle), fingers wrapping toward +X, back of the hand toward -X
    static const Xf palm = {{-0.5f, 0.5f, 0.5f, 0.5f}, {-0.035f, 0.025f, 0}};
    for (int i = 0; i < 26; i++) {
        j[i] = xf_mul(palm, pj[i]);
        if (right) { j[i].p.x = -j[i].p.x; j[i].q.y = -j[i].q.y; j[i].q.z = -j[i].q.z; }
    }
}
/// Finger curls (thumb..little) a Touch controller's sensors imply.
static void controller_curls(const VR4Hand *h, float c[5]) {
    c[0] = h->buttons & (VR4_BTN_THUMB_TOUCH | VR4_BTN_STICK_TOUCH | VR4_BTN_A | VR4_BTN_B | VR4_BTN_X | VR4_BTN_Y | VR4_BTN_STICK_CLICK) ? 0.6f : 0.1f;
    c[1] = h->buttons & VR4_BTN_TRIGGER_TOUCH ? 0.45f + 0.55f * h->trigger : 0.6f * h->trigger;   // lifted off the trigger: pointing
    c[2] = c[3] = c[4] = 0.15f + 0.85f * h->squeeze;
}
/// Model-space bones of `hand` (31). Real joints while tracked, else the controller pose; `limit` caps curls at
/// the controller's handle (VRSkeletalMotionRange_WithController).
static void hand_bones(int hand, int limit, const float *curls, VRBoneTransform_t out[NBONES]) {
    Xf j[26];
    if (!curls && joints[hand].tracked) {
        Xf root = xf_inv(xf_of(track.hand[hand].grip));
        for (int i = 0; i < 26; i++) j[i] = xf_mul(root, xf_of(joints[hand].joint[i]));
    } else {
        float c[5];
        if (curls) memcpy(c, curls, sizeof c); else controller_curls(&track.hand[hand], c);
        for (int f = 0; f < 5 && limit; f++) c[f] = fminf(c[f], 0.75f);
        synth_joints(c, hand, j);
    }
    static const Quat K[2] = {{0.70710678f, 0, -0.70710678f, 0}, {0, -0.70710678f, 0, 0.70710678f}}, F = {0.5f, -0.5f, 0.5f, 0.5f};
    float ws = world_scale();
    for (int b = 0; b < NBONES; b++) {
        Xf m = b == 0 ? (Xf){{0, 0, 0, 1}, {0, 0, 0}} : j[b < 26 ? b : sxr_tip(b - 26) - 1];   // aux bones = the distal joints
        if (b) m.q = q_mul(m.q, b == 1 ? q_mul(F, K[hand]) : K[hand]);
        out[b] = (VRBoneTransform_t){{{m.p.x / ws, m.p.y / ws, m.p.z / ws, 1}}, {m.q.w, m.q.x, m.q.y, m.q.z}};
    }
}
static Xf bone_xf(VRBoneTransform_t b) { return (Xf){{b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w}, {b.position.v[0], b.position.v[1], b.position.v[2]}}; }
static void to_parent_space(VRBoneTransform_t t[NBONES]) {
    VRBoneTransform_t m[NBONES]; memcpy(m, t, sizeof m);
    for (int b = 1; b < NBONES; b++) {
        Xf l = xf_mul(xf_inv(bone_xf(m[boneParent[b]])), bone_xf(m[b]));
        t[b] = (VRBoneTransform_t){{{l.p.x, l.p.y, l.p.z, 1}}, {l.q.w, l.q.x, l.q.y, l.q.z}};
    }
}
static void to_model_space(VRBoneTransform_t t[NBONES]) {   // parents come first
    for (int b = 1; b < NBONES; b++) {
        Xf m = xf_mul(bone_xf(t[boneParent[b]]), bone_xf(t[b]));
        t[b] = (VRBoneTransform_t){{{m.p.x, m.p.y, m.p.z, 1}}, {m.q.w, m.q.x, m.q.y, m.q.z}};
    }
}
static Action *skeleton(VRActionHandle_t h, EVRInputError *err) {
    Action *a = action(h);
    *err = !a ? EVRInputError_VRInputError_InvalidHandle : strcmp(a->type, "skeleton") || a->hand < 0 ? EVRInputError_VRInputError_WrongType : 0;
    return *err ? NULL : a;
}
static EVRInputError GetSkeletalActionData(VRActionHandle_t h, InputSkeletalActionData_t *d, uint32_t size) {
    EVRInputError err; Action *a = skeleton(h, &err);
    if (!a) return err;
    memset(d, 0, size);
    d->bActive = joints[a->hand].tracked || (track.hand[a->hand].flags & VR4_HAND_ACTIVE);
    d->activeOrigin = (VRInputValueHandle_t)a->hand + 1;
    return 0;
}
static ETrackedControllerRole dominantHand = ETrackedControllerRole_TrackedControllerRole_RightHand;
static EVRInputError GetDominantHand(ETrackedControllerRole *r) { *r = dominantHand; return 0; }
static EVRInputError SetDominantHand(ETrackedControllerRole r) { dominantHand = r; return 0; }
static EVRInputError GetBoneCount(VRActionHandle_t h, uint32_t *n) {
    EVRInputError err; if (!skeleton(h, &err)) return err;
    *n = NBONES; return 0;
}
static EVRInputError GetBoneHierarchy(VRActionHandle_t h, BoneIndex_t *parents, uint32_t n) {
    EVRInputError err; if (!skeleton(h, &err)) return err;
    if (n != NBONES) return EVRInputError_VRInputError_InvalidBoneCount;
    memcpy(parents, boneParent, sizeof boneParent); return 0;
}
static EVRInputError GetBoneName(VRActionHandle_t h, BoneIndex_t b, char *name, uint32_t size) {
    EVRInputError err; Action *a = skeleton(h, &err);
    if (!a) return err;
    if (b < 0 || b >= NBONES) return EVRInputError_VRInputError_InvalidBoneIndex;
    if (strlen(boneName[b]) + 1 > size) return EVRInputError_VRInputError_BufferTooSmall;
    strcpy(name, boneName[b]);
    char *q = strchr(name, '?'); if (q) *q = a->hand ? 'r' : 'l';
    return 0;
}
static EVRInputError GetSkeletalReferenceTransforms(VRActionHandle_t h, EVRSkeletalTransformSpace space, EVRSkeletalReferencePose pose, VRBoneTransform_t *t, uint32_t n) {
    EVRInputError err; Action *a = skeleton(h, &err);
    if (!a) return err;
    if (n != NBONES) return EVRInputError_VRInputError_InvalidBoneCount;
    static const float open[5] = {0}, fist[5] = {1, 1, 1, 1, 1}, grip[5] = {0.6f, 0.75f, 0.75f, 0.75f, 0.75f};
    hand_bones(a->hand, 0, pose == EVRSkeletalReferencePose_VRSkeletalReferencePose_Fist ? fist :
                           pose == EVRSkeletalReferencePose_VRSkeletalReferencePose_GripLimit ? grip : open, t);
    if (space == EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Parent) to_parent_space(t);
    return 0;
}
static EVRInputError GetSkeletalTrackingLevel(VRActionHandle_t h, EVRSkeletalTrackingLevel *l) {
    EVRInputError err; Action *a = skeleton(h, &err);
    if (!a) return err;
    *l = joints[a->hand].tracked ? EVRSkeletalTrackingLevel_VRSkeletalTracking_Full : EVRSkeletalTrackingLevel_VRSkeletalTracking_Partial;   // Touch: capacitive fingers
    return 0;
}
static EVRInputError GetSkeletalBoneData(VRActionHandle_t h, EVRSkeletalTransformSpace space, EVRSkeletalMotionRange range, VRBoneTransform_t *t, uint32_t n) {
    EVRInputError err; Action *a = skeleton(h, &err);
    if (!a) return err;
    if (n != NBONES) return EVRInputError_VRInputError_InvalidBoneCount;
    hand_bones(a->hand, range == EVRSkeletalMotionRange_VRSkeletalMotionRange_WithController, NULL, t);
    if (space == EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Parent) to_parent_space(t);
    return 0;
}
static EVRInputError GetSkeletalSummaryData(VRActionHandle_t h, EVRSummaryType type, VRSkeletalSummaryData_t *d) {
    (void)type;
    EVRInputError err; Action *a = skeleton(h, &err);
    if (!a) return err;
    memset(d, 0, sizeof *d);
    const VR4HandJoints *j = &joints[a->hand];
    if (!j->tracked) { controller_curls(&track.hand[a->hand], d->flFingerCurl); return 0; }
    V3 dir[5];   // proximal bone directions, for the splay between neighbouring fingers
    for (int f = 0; f < 5; f++) {
        d->flFingerCurl[f] = sxr_curl(j, f);
        VR4Pose p = j->joint[sxr_meta(f) + 1], q = j->joint[sxr_meta(f) + 2];
        float l = fmaxf(sxr_dist(p, q), 1e-5f);
        dir[f] = (V3){(q.px - p.px) / l, (q.py - p.py) / l, (q.pz - p.pz) / l};
    }
    for (int f = 0; f < 4; f++) {
        float c = dir[f].x * dir[f + 1].x + dir[f].y * dir[f + 1].y + dir[f].z * dir[f + 1].z;
        d->flFingerSplay[f] = fminf(1, acosf(fmaxf(-1, fminf(1, c))) / (f ? 0.35f : 0.6f));   // full splay ~20 deg (thumb ~35)
    }
    return 0;
}
// Compressed bone data: our own format (only DecompressSkeletalBoneData reads it): a tag, then the parent-space bones.
enum { BONE_TAG = 0x42585253 };   // "SRXB"
static EVRInputError GetSkeletalBoneDataCompressed(VRActionHandle_t h, EVRSkeletalMotionRange range, void *buf, uint32_t size, uint32_t *needed) {
    uint32_t n = 4 + sizeof(VRBoneTransform_t) * NBONES;
    if (needed) *needed = n;
    VRBoneTransform_t t[NBONES];
    EVRInputError err = GetSkeletalBoneData(h, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Parent, range, t, NBONES);
    if (err) return err;
    if (!buf || size < n) return EVRInputError_VRInputError_BufferTooSmall;
    uint32_t tag = BONE_TAG; memcpy(buf, &tag, 4); memcpy((char *)buf + 4, t, sizeof t);
    return 0;
}
static EVRInputError DecompressSkeletalBoneData(void *buf, uint32_t size, EVRSkeletalTransformSpace space, VRBoneTransform_t *t, uint32_t n) {
    uint32_t tag = 0;
    if (!buf || size < 4 + sizeof(VRBoneTransform_t) * NBONES || (memcpy(&tag, buf, 4), tag != BONE_TAG)) return EVRInputError_VRInputError_InvalidCompressedData;
    if (n != NBONES) return EVRInputError_VRInputError_InvalidBoneCount;
    memcpy(t, (char *)buf + 4, sizeof(VRBoneTransform_t) * NBONES);
    if (space == EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model) to_model_space(t);
    return 0;
}

// ---------------------------------------------------------------- IVRSystem
static uint32_t eye_dim(uint32_t base) {
    float s = shm && shm->render_scale > 0.01f ? shm->render_scale : 1;
    uint32_t v = ((uint32_t)(base * s) + 16) / 32 * 32;
    return v < 128 ? 128 : v > 4096 ? 4096 : v;
}
static void GetRecommendedRenderTargetSize(uint32_t *w, uint32_t *h) {
    *w = eye_dim(shm && shm->eye_w ? shm->eye_w : 1440);
    *h = eye_dim(shm && shm->eye_h ? shm->eye_h : 1584);
}
static VR4Fov eye_fov(EVREye e) {
    read_tracking();
    VR4Fov f = track.eye[e == EVREye_Eye_Right].fov;
    if (f.left == 0 && f.right == 0) f = (VR4Fov){-0.94f, 0.79f, 0.84f, -0.96f};   // Quest 2-ish until tracking arrives
    if (e == EVREye_Eye_Right && track.eye[1].fov.left == 0) f = (VR4Fov){-f.right, -f.left, f.up, f.down};
    return f;
}
static void GetProjectionRaw(EVREye e, float *l, float *r, float *t, float *b) {
    VR4Fov f = eye_fov(e);
    *l = tanf(f.left); *r = tanf(f.right); *t = tanf(f.down); *b = tanf(f.up);   // OpenVR's "top" is the down tangent
}
static HmdMatrix44_t GetProjectionMatrix(EVREye e, float n, float fa) {   // OpenGL clip space, like xr_linear.h
    VR4Fov f = eye_fov(e);
    float L = tanf(f.left), R = tanf(f.right), U = tanf(f.up), D = tanf(f.down), W = R - L, H = U - D;
    return (HmdMatrix44_t){{{2 / W, 0, (R + L) / W, 0}, {0, 2 / H, (U + D) / H, 0},
                            {0, 0, -(fa + n) / (fa - n), -2 * fa * n / (fa - n)}, {0, 0, -1, 0}}};
}
static HmdMatrix34_t GetEyeToHeadTransform(EVREye e) {
    read_tracking();
    VR4Pose eye = track.eye[e == EVREye_Eye_Right].pose;
    if (eye.qw == 0 && eye.qx == 0 && eye.qy == 0 && eye.qz == 0) {   // no tracking yet: 64 mm IPD
        HmdMatrix34_t m = {{{1, 0, 0, e == EVREye_Eye_Right ? 0.032f : -0.032f}, {0, 1, 0, 0}, {0, 0, 1, 0}}};
        return m;
    }
    return mul34(inv34(mat(track.head)), mat(eye));
}
static bool ComputeDistortion(EVREye e, float u, float v, DistortionCoordinates_t *d) {
    (void)e;
    d->rfRed[0] = d->rfGreen[0] = d->rfBlue[0] = u; d->rfRed[1] = d->rfGreen[1] = d->rfBlue[1] = v;
    return true;
}
static ETrackedDeviceClass GetTrackedDeviceClass(TrackedDeviceIndex_t i) {
    return i == 0 ? ETrackedDeviceClass_TrackedDeviceClass_HMD : i <= 2 ? ETrackedDeviceClass_TrackedDeviceClass_Controller : ETrackedDeviceClass_TrackedDeviceClass_Invalid;
}
static bool IsTrackedDeviceConnected(TrackedDeviceIndex_t i) { return i <= 2; }
static ETrackedControllerRole GetControllerRoleForTrackedDeviceIndex(TrackedDeviceIndex_t i) {
    return i == 1 ? ETrackedControllerRole_TrackedControllerRole_LeftHand : i == 2 ? ETrackedControllerRole_TrackedControllerRole_RightHand : ETrackedControllerRole_TrackedControllerRole_Invalid;
}
static TrackedDeviceIndex_t GetTrackedDeviceIndexForControllerRole(ETrackedControllerRole r) {
    return r == ETrackedControllerRole_TrackedControllerRole_LeftHand ? 1 : r == ETrackedControllerRole_TrackedControllerRole_RightHand ? 2 : k_unTrackedDeviceIndexInvalid;
}
static EDeviceActivityLevel GetTrackedDeviceActivityLevel(TrackedDeviceIndex_t i) { (void)i; return EDeviceActivityLevel_k_EDeviceActivityLevel_UserInteraction; }
static void GetDeviceToAbsoluteTrackingPose(ETrackingUniverseOrigin o, float t, TrackedDevicePose_t *p, uint32_t n) {
    (void)o; (void)t;
    read_tracking();
    for (uint32_t i = 0; i < n; i++) p[i] = device_pose((int)i);
}
static HmdMatrix34_t identity34(void) { return (HmdMatrix34_t){{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}}; }
static HmdMatrix34_t GetSeatedZeroPoseToStandingAbsoluteTrackingPose(void) { return identity34(); }

// Device 0 is the headset, 1/2 the Touch controllers; they report what SteamVR reports for a Quest 2 over Link. The
// shared memory carries no battery level: batteries read full and DeviceProvidesBatteryStatus is false.
static int prop_device(TrackedDeviceIndex_t i, ETrackedPropertyError *err) {
    if (err) *err = i > 2 ? ETrackedPropertyError_TrackedProp_InvalidDevice : ETrackedPropertyError_TrackedProp_Success;
    return i <= 2;
}
static void unknown(ETrackedPropertyError *err) { if (err) *err = ETrackedPropertyError_TrackedProp_UnknownProperty; }
static float GetFloatTrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    if (!prop_device(i, err)) return 0;
    switch ((int)p) {
    case ETrackedDeviceProperty_Prop_DeviceBatteryPercentage_Float: return 1;
    case ETrackedDeviceProperty_Prop_DisplayFrequency_Float: if (i == 0) return fps(); break;
    case ETrackedDeviceProperty_Prop_SecondsFromVsyncToPhotons_Float: if (i == 0) return 0.011f; break;
    case ETrackedDeviceProperty_Prop_UserHeadToEyeDepthMeters_Float: if (i == 0) return 0; break;
    case ETrackedDeviceProperty_Prop_UserIpdMeters_Float:
        if (i == 0) { HmdMatrix34_t l = GetEyeToHeadTransform(EVREye_Eye_Left), r = GetEyeToHeadTransform(EVREye_Eye_Right); return fabsf(r.m[0][3] - l.m[0][3]); }
        break;
    }
    unknown(err); return 0;
}
static uint32_t GetStringTrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, char *v, uint32_t size, ETrackedPropertyError *err) {
    if (!prop_device(i, err)) return 0;
    const char *s = NULL;
    switch ((int)p) {
    case ETrackedDeviceProperty_Prop_TrackingSystemName_String: s = "oculus"; break;
    case ETrackedDeviceProperty_Prop_ManufacturerName_String: s = "Oculus"; break;
    case ETrackedDeviceProperty_Prop_ModelNumber_String: s = i == 0 ? "Oculus Quest2" : i == 1 ? "Oculus Quest2 (Left Controller)" : "Oculus Quest2 (Right Controller)"; break;
    case ETrackedDeviceProperty_Prop_SerialNumber_String: case ETrackedDeviceProperty_Prop_ManufacturerSerialNumber_String:
        s = i == 0 ? "MACVR-HMD" : i == 1 ? "MACVR-LEFT" : "MACVR-RIGHT"; break;
    case ETrackedDeviceProperty_Prop_RenderModelName_String: s = i == 0 ? "oculus_quest2" : i == 1 ? "oculus_quest2_controller_left" : "oculus_quest2_controller_right"; break;
    case ETrackedDeviceProperty_Prop_ControllerType_String: s = i == 0 ? "quest2_hmd" : "oculus_touch"; break;
    case ETrackedDeviceProperty_Prop_RegisteredDeviceType_String: s = i == 0 ? "oculus/MACVR_HMD" : i == 1 ? "oculus/MACVR_Controller_Left" : "oculus/MACVR_Controller_Right"; break;
    case ETrackedDeviceProperty_Prop_InputProfilePath_String: if (i) s = "{oculus}/input/touch_profile.json"; break;
    case ETrackedDeviceProperty_Prop_TrackingFirmwareVersion_String: case ETrackedDeviceProperty_Prop_DriverVersion_String: s = "SiliconXR"; break;
    case ETrackedDeviceProperty_Prop_HardwareRevision_String: s = "1"; break;
    }
    if (!s) { unknown(err); if (v && size) *v = 0; return 0; }
    uint32_t n = (uint32_t)strlen(s) + 1;
    if (!v || size < n) { if (err) *err = ETrackedPropertyError_TrackedProp_BufferTooSmall; return n; }
    memcpy(v, s, n);
    return n;
}
static bool GetBoolTrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    if (!prop_device(i, err)) return false;
    switch ((int)p) {
    case ETrackedDeviceProperty_Prop_DeviceIsWireless_Bool: return true;
    case ETrackedDeviceProperty_Prop_WillDriftInYaw_Bool: case ETrackedDeviceProperty_Prop_DeviceIsCharging_Bool:
    case ETrackedDeviceProperty_Prop_DeviceProvidesBatteryStatus_Bool: case ETrackedDeviceProperty_Prop_DeviceCanPowerOff_Bool:
    case ETrackedDeviceProperty_Prop_Identifiable_Bool: case ETrackedDeviceProperty_Prop_HasCamera_Bool: case ETrackedDeviceProperty_Prop_IsOnDesktop_Bool:
        return false;
    case ETrackedDeviceProperty_Prop_ContainsProximitySensor_Bool: case ETrackedDeviceProperty_Prop_HasDisplayComponent_Bool: return i == 0;
    case ETrackedDeviceProperty_Prop_HasControllerComponent_Bool: return i != 0;
    }
    unknown(err); return false;
}
static int32_t GetInt32TrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    if (!prop_device(i, err)) return 0;
    switch ((int)p) {
    case ETrackedDeviceProperty_Prop_DeviceClass_Int32: return GetTrackedDeviceClass(i);
    case ETrackedDeviceProperty_Prop_ControllerRoleHint_Int32: return GetControllerRoleForTrackedDeviceIndex(i);
    case ETrackedDeviceProperty_Prop_ControllerHandSelectionPriority_Int32: if (i) return 0; break;
    case ETrackedDeviceProperty_Prop_Axis0Type_Int32: if (i) return EVRControllerAxisType_k_eControllerAxis_Joystick; break;
    case ETrackedDeviceProperty_Prop_Axis1Type_Int32: case ETrackedDeviceProperty_Prop_Axis2Type_Int32: if (i) return EVRControllerAxisType_k_eControllerAxis_Trigger; break;
    }
    unknown(err); return 0;
}
static uint64_t GetUint64TrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    if (!prop_device(i, err)) return 0;
    switch ((int)p) {
    case ETrackedDeviceProperty_Prop_HardwareRevision_Uint64: case ETrackedDeviceProperty_Prop_FirmwareVersion_Uint64: return 1;
    case ETrackedDeviceProperty_Prop_CurrentUniverseId_Uint64: return 1;
    case ETrackedDeviceProperty_Prop_SupportedButtons_Uint64:
        return i == 0 ? 1ULL << EVRButtonId_k_EButton_ProximitySensor
                      : 1ULL << EVRButtonId_k_EButton_ApplicationMenu | 1ULL << EVRButtonId_k_EButton_Grip | 1ULL << EVRButtonId_k_EButton_A |
                        1ULL << EVRButtonId_k_EButton_SteamVR_Touchpad | 1ULL << EVRButtonId_k_EButton_SteamVR_Trigger | 1ULL << (EVRButtonId_k_EButton_SteamVR_Trigger + 1);
    }
    unknown(err); return 0;
}
/// The lens mask (siliconxr_shared.h) in the eye texture's 0..1 coordinates (top left origin). Vivecraft stencils
/// it out so the hidden corners are never shaded.
static HiddenAreaMesh_t GetHiddenAreaMesh(EVREye e, EHiddenAreaMeshType t) {
    (void)e;
    static HmdVector2_t verts[3][6 * SXR_MASK_N]; static uint32_t count[3]; static dispatch_once_t once;
    dispatch_once(&once, ^{
        float tri[2 * SXR_MASK_N][3][2], ring[SXR_MASK_N][2], rect[SXR_MASK_N][2];
        for (int type = 0; type < 2; type++) {   // 0 standard = hidden corners, 1 inverse = visible area
            int n = sxr_mask_triangles(type == 0, tri);
            for (int k = 0; k < n; k++) for (int v = 0; v < 3; v++) verts[type][3 * k + v] = (HmdVector2_t){{(tri[k][v][0] + 1) / 2, (1 - tri[k][v][1]) / 2}};
            count[type] = (uint32_t)n;
        }
        sxr_mask_ring(ring, rect);   // line loop: unTriangleCount is the vertex count
        for (int k = 0; k < SXR_MASK_N; k++) verts[2][k] = (HmdVector2_t){{(ring[k][0] + 1) / 2, (1 - ring[k][1]) / 2}};
        count[2] = SXR_MASK_N;
    });
    if (t < 0 || t > 2) return (HiddenAreaMesh_t){NULL, 0};
    return (HiddenAreaMesh_t){verts[t], count[t]};
}
/// Events: devices appear at start; a hand switching between its controller and hand tracking is a device update
/// (Vivecraft then re-reads the controller's aim transform); MacVR's menu opening and closing is the dashboard.
static bool PollNextEvent(struct VREvent_t *e, uint32_t size) {
    static uint32_t queue[16][2]; static int head, tail, started, tracked[2], dash;
    #define PUSH(type, dev) do { if (tail - head < 16) { queue[tail % 16][0] = (type); queue[tail % 16][1] = (dev); tail++; } } while (0)
    if (!started) {
        started = 1;
        for (uint32_t d = 0; d < 3; d++) PUSH(EVREventType_VREvent_TrackedDeviceActivated, d);
        if (shm) { tracked[0] = shm->hand_joints[0].tracked != 0; tracked[1] = shm->hand_joints[1].tracked != 0; dash = shm->input_blocked != 0; }
    }
    for (int h = 0; shm && h < 2; h++) {
        int t = shm->hand_joints[h].tracked != 0;
        if (t != tracked[h]) { tracked[h] = t; PUSH(EVREventType_VREvent_TrackedDeviceUpdated, (uint32_t)h + 1); }
    }
    if (shm && (shm->input_blocked != 0) != dash) {
        dash = !dash; PUSH(dash ? EVREventType_VREvent_DashboardActivated : EVREventType_VREvent_DashboardDeactivated, 0);
    }
    #undef PUSH
    if (head == tail || !e || size < 12) return false;
    memset(e, 0, size);
    e->eventType = queue[head % 16][0]; e->trackedDeviceIndex = queue[head % 16][1]; e->eventAgeSeconds = 0;
    head++;
    return true;
}
static bool ShouldApplicationPause(void) { return false; }
static bool IsInputAvailable(void) { return !(shm && shm->input_blocked); }

static bool GetControllerState(TrackedDeviceIndex_t i, VRControllerState_t *cs, uint32_t size) {
    if (i != 1 && i != 2) return false;
    if (!cs || size < sizeof(VRControllerState_t)) return false;
    read_tracking();
    if (shm && shm->input_blocked) {
        memset(cs, 0, sizeof(*cs));
        return true;
    }
    const VR4Hand *h = &track.hand[i - 1];
    memset(cs, 0, sizeof(*cs));
    cs->unPacketNum = (uint32_t)(track.time_ns / 1000000);
    uint64_t pressed = 0, touched = 0;
    if (h->flags & VR4_HAND_ACTIVE) {
        if (h->trigger > 0.5f) pressed |= (1ULL << EVRButtonId_k_EButton_SteamVR_Trigger) | (1ULL << EVRButtonId_k_EButton_Axis1);
        if (h->trigger > 0.05f || (h->buttons & VR4_BTN_TRIGGER_TOUCH)) touched |= (1ULL << EVRButtonId_k_EButton_SteamVR_Trigger) | (1ULL << EVRButtonId_k_EButton_Axis1);
        if (h->squeeze > 0.5f) pressed |= (1ULL << EVRButtonId_k_EButton_Grip) | (1ULL << EVRButtonId_k_EButton_Axis2);
        if (h->squeeze > 0.05f) touched |= (1ULL << EVRButtonId_k_EButton_Grip) | (1ULL << EVRButtonId_k_EButton_Axis2);
        if (h->buttons & VR4_BTN_STICK_CLICK) pressed |= (1ULL << EVRButtonId_k_EButton_SteamVR_Touchpad) | (1ULL << EVRButtonId_k_EButton_Axis0);
        if ((h->buttons & VR4_BTN_STICK_TOUCH) || fabsf(h->stick_x) > 0.1f || fabsf(h->stick_y) > 0.1f) touched |= (1ULL << EVRButtonId_k_EButton_SteamVR_Touchpad) | (1ULL << EVRButtonId_k_EButton_Axis0);
        if (i == 1) {
            if (h->buttons & VR4_BTN_X) pressed |= (1ULL << EVRButtonId_k_EButton_A);
            if (h->buttons & VR4_BTN_Y) pressed |= (1ULL << EVRButtonId_k_EButton_ApplicationMenu);
        } else {
            if (h->buttons & VR4_BTN_A) pressed |= (1ULL << EVRButtonId_k_EButton_A);
            if (h->buttons & VR4_BTN_B) pressed |= (1ULL << EVRButtonId_k_EButton_ApplicationMenu);
        }
        if (h->buttons & VR4_BTN_THUMB_TOUCH) touched |= (1ULL << EVRButtonId_k_EButton_A) | (1ULL << EVRButtonId_k_EButton_ApplicationMenu);
        if (h->buttons & VR4_BTN_MENU) pressed |= (1ULL << EVRButtonId_k_EButton_ApplicationMenu);
        cs->rAxis[0].x = h->stick_x;
        cs->rAxis[0].y = h->stick_y;
        cs->rAxis[1].x = h->trigger;
        cs->rAxis[2].x = h->squeeze;
    }
    cs->ulButtonPressed = pressed;
    cs->ulButtonTouched = touched;
    return true;
}

static bool GetControllerStateWithPose(ETrackingUniverseOrigin o, TrackedDeviceIndex_t i, VRControllerState_t *cs, uint32_t size, TrackedDevicePose_t *pose) {
    (void)o;
    if (!GetControllerState(i, cs, size)) return false;
    if (pose) {
        *pose = device_pose((int)i);
        fill_velocities(pose, 1);
    }
    return true;
}

static void TriggerHapticPulse(TrackedDeviceIndex_t i, uint32_t axis, unsigned short durUs) {
    (void)axis;
    if (i != 1 && i != 2) return;
    sxr_haptic(shm, (int)i - 1, 1, durUs > 0 ? (float)durUs / 1000000.0f : 0.02f, 160, 0);
}

/// MacVR shows this as the game's name (the window title, e.g. "Minecraft* 1.20.1"). AppKit is only touched on the
/// main thread, which is the render thread under GLFW's -XstartOnFirstThread.
static void publish_app_name(void) {
    if (!shm) return;
    NSString *title = nil;
    if ([NSThread isMainThread])
        for (NSWindow *w in NSApp.windows) if (w.isVisible && w.title.length) { title = w.title; break; }
    const char *name = title ? title.UTF8String : getprogname() && strcmp(getprogname(), "java") ? getprogname() : "Minecraft";
    if (strncmp(shm->app_name, name, sizeof shm->app_name - 1)) snprintf(shm->app_name, sizeof shm->app_name, "%s", name);
}

// ---------------------------------------------------------------- IVRCompositor
static VR4Pose renderHead;             // head pose handed out by the last WaitGetPoses
static uint64_t renderTime, nextVsync;
static GLuint readFbo, drawFbo, drawRbo; static uint32_t rboW, rboH;

static void SetTrackingSpace(ETrackingUniverseOrigin o) { trackingSpace = o; }
static ETrackingUniverseOrigin GetTrackingSpace(void) { return trackingSpace; }
static EVRCompositorError WaitGetPoses(TrackedDevicePose_t *render, uint32_t nr, TrackedDevicePose_t *game, uint32_t ng) {
    uint64_t period = (uint64_t)(1e9 / fps()), t = now_ns();   // pace the game to the headset refresh
    if (nextVsync > t && nextVsync - t <= period) { struct timespec ts = {0, (long)(nextVsync - t)}; nanosleep(&ts, NULL); t = nextVsync; }
    nextVsync = (nextVsync && t - nextVsync < period ? nextVsync : t) + period;
    read_tracking();
    renderHead = track.head; renderTime = track.time_ns;
    static uint64_t nameAt;
    if (t > nameAt) { @autoreleasepool { publish_app_name(); } nameAt = t + 2000000000ull; }   // titles change (world names)
    TrackedDevicePose_t p[3] = {device_pose(0), device_pose(1), device_pose(2)};
    fill_velocities(p, 3);
    for (uint32_t i = 0; render && i < nr; i++) render[i] = i < 3 ? p[i] : (TrackedDevicePose_t){0};
    for (uint32_t i = 0; game && i < ng; i++) game[i] = i < 3 ? p[i] : (TrackedDevicePose_t){0};
    if (shm) shm->runtime_heartbeat_ns = t;
    return EVRCompositorError_VRCompositorError_None;
}
static EVRCompositorError GetLastPoses(TrackedDevicePose_t *render, uint32_t nr, TrackedDevicePose_t *game, uint32_t ng) {
    for (uint32_t i = 0; render && i < nr; i++) render[i] = device_pose((int)i);
    for (uint32_t i = 0; game && i < ng; i++) game[i] = device_pose((int)i);
    return EVRCompositorError_VRCompositorError_None;
}

/// Reads an OpenGL eye texture into a side-by-side frame (top row first), flipping on the GPU with a blit.
/// GL 3+ contexts: each eye goes into a pixel buffer object without waiting, a fence follows the right eye, and a worker
/// thread with a context shared with the game's copies the frame into shm once the GPU is done, so the render thread
/// never stalls on the GPU. Legacy (2.1) contexts read back synchronously.
/// FadeToColor: the color animates from the current one to the target over the given time and is blended over the
/// submitted frames (the background fade only applies to SteamVR's own scene, so it is just reported back).
static struct { HmdColor_t from, to; uint64_t start, dur; } fade[2];   // [background]
static HmdColor_t fade_color(int bg) {
    uint64_t t = now_ns();
    float k = fade[bg].dur && t < fade[bg].start + fade[bg].dur ? (float)(t - fade[bg].start) / fade[bg].dur : 1;
    HmdColor_t a = fade[bg].from, b = fade[bg].to;
    return (HmdColor_t){a.r + (b.r - a.r) * k, a.g + (b.g - a.g) * k, a.b + (b.b - a.b) * k, a.a + (b.a - a.a) * k};
}
static void FadeToColor(float seconds, float r, float g, float b, float a, bool bg) {
    fade[bg != 0].from = fade_color(bg != 0); fade[bg != 0].to = (HmdColor_t){r, g, b, a};
    fade[bg != 0].start = now_ns(); fade[bg != 0].dur = seconds > 0 ? (uint64_t)(seconds * 1e9) : 0;
}
static void apply_fade(uint8_t *bgra, size_t pixels, HmdColor_t c) {
    if (c.a < 0.004f) return;
    uint32_t a = (uint32_t)(fminf(c.a, 1) * 256), na = 256 - a;
    uint32_t cb = (uint32_t)(fminf(c.b, 1) * 255) * a, cg = (uint32_t)(fminf(c.g, 1) * 255) * a, cr = (uint32_t)(fminf(c.r, 1) * 255) * a;
    for (size_t i = 0; i < pixels; i++, bgra += 4) {
        bgra[0] = (uint8_t)((bgra[0] * na + cb) >> 8); bgra[1] = (uint8_t)((bgra[1] * na + cg) >> 8); bgra[2] = (uint8_t)((bgra[2] * na + cr) >> 8);
    }
}

enum { NPBO = 3 };
typedef struct { GLuint pbo; size_t size; uint32_t w, h; VR4Pose pose[2]; uint64_t time; volatile int busy; } Pbo;
static Pbo pbos[NPBO]; static int pboCur = -1;   // slot being filled by this frame's Submits (-1: none / dropped)
static CGLContextObj gameCtx, workerCtx;
static dispatch_queue_t worker_queue(void) {
    static dispatch_queue_t q; static dispatch_once_t once;
    dispatch_once(&once, ^{ q = dispatch_queue_create("SiliconXR.readback", DISPATCH_QUEUE_SERIAL); });
    return q;
}
static void publish_frame(uint32_t buf, uint32_t w, uint32_t h, const VR4Pose pose[2], uint64_t time) {
    apply_fade(vr4_frame(shm, buf), (size_t)2 * w * h, fade_color(0));
    shm->frame_eye_pose[buf][0] = pose[0]; shm->frame_eye_pose[buf][1] = pose[1];
    shm->frame_w[buf] = 2 * w; shm->frame_h[buf] = h; shm->frame_rgba[buf] = 0;
    shm->frame_time_ns[buf] = time;
    shm->runtime_heartbeat_ns = now_ns();
    vr4_fence();
    shm->frame_seq++;
}
/// GL 3+ and a worker context sharing the current context's objects; 0 = read back synchronously.
static int async_gl(void) {
    CGLContextObj ctx = CGLGetCurrentContext();
    if (ctx == gameCtx) return workerCtx != NULL;
    dispatch_sync(worker_queue(), ^{});   // the old context's frames are done
    if (workerCtx) CGLDestroyContext(workerCtx);
    workerCtx = NULL; gameCtx = ctx; pboCur = -1;
    readFbo = drawFbo = drawRbo = 0; rboW = rboH = 0;   // framebuffers are per context: make new ones
    for (int k = 0; k < NPBO; k++) pbos[k] = (Pbo){0};   // they belonged to the old context
    const char *v = (const char *)glGetString(GL_VERSION);
    if (ctx && v && atoi(v) >= 3 && CGLCreateContext(CGLGetPixelFormat(ctx), ctx, &workerCtx) != kCGLNoError) workerCtx = NULL;
    logmsg("GL %s: %s readback", v ? v : "?", workerCtx ? "asynchronous" : "synchronous");
    return workerCtx != NULL;
}
static uint32_t frameW, frameH;
static EVRCompositorError Submit(EVREye eye, Texture_t *tex, VRTextureBounds_t *bounds, EVRSubmitFlags flags) {
    (void)flags;
    if (!shm || !tex) return EVRCompositorError_VRCompositorError_DoNotHaveFocus;
    if (tex->eType != ETextureType_TextureType_OpenGL) return EVRCompositorError_VRCompositorError_TextureUsesUnsupportedFormat;
    GLuint name = (GLuint)(uintptr_t)tex->handle;
    GLint prevRead, prevDraw, prevTex, prevRb, prevPack, prevAlign, prevPbo, scissor = glIsEnabled(GL_SCISSOR_TEST);
    glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &prevRead); glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &prevDraw);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &prevTex); glGetIntegerv(GL_RENDERBUFFER_BINDING, &prevRb);
    glGetIntegerv(GL_PACK_ROW_LENGTH, &prevPack); glGetIntegerv(GL_PACK_ALIGNMENT, &prevAlign); glGetIntegerv(GL_PIXEL_PACK_BUFFER_BINDING, &prevPbo);
    GLint tw = 0, th = 0;
    glBindTexture(GL_TEXTURE_2D, name);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &tw); glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &th);
    glBindTexture(GL_TEXTURE_2D, (GLuint)prevTex);
    if (tw <= 0 || th <= 0) return EVRCompositorError_VRCompositorError_InvalidTexture;
    VRTextureBounds_t b = bounds ? *bounds : (VRTextureBounds_t){0, 0, 1, 1};
    // GL textures: v = 0 is the top of the image, i.e. the last GL row. Inverted bounds flip it back.
    GLint sx0 = (GLint)(b.uMin * tw), sx1 = (GLint)(b.uMax * tw), sy0 = (GLint)((1 - b.vMax) * th), sy1 = (GLint)((1 - b.vMin) * th);
    uint32_t w = (uint32_t)abs(sx1 - sx0), h = (uint32_t)abs(sy1 - sy0);
    while (2ull * w * h * 4 > VR4_FRAME_MAX) { w = w * 7 / 8; h = h * 7 / 8; }   // stay inside the shm frame
    if (!w || !h) return EVRCompositorError_VRCompositorError_InvalidBounds;
    uint32_t e = eye == EVREye_Eye_Right;
    if (!e || frameW != w || frameH != h) { frameW = w; frameH = h; }
    int async = async_gl();
    Pbo *slot = NULL;
    if (async) {
        if (!e) {   // a new frame: a free buffer, else drop the frame (the GPU is several frames behind)
            pboCur = -1;
            for (int k = 0; k < NPBO && pboCur < 0; k++) if (!__atomic_load_n(&pbos[k].busy, __ATOMIC_ACQUIRE)) pboCur = k;
        }
        if (pboCur < 0 || (e && (pbos[pboCur].w != w || pbos[pboCur].h != h))) { pboCur = -1; return EVRCompositorError_VRCompositorError_None; }
        slot = &pbos[pboCur];
    }

    if (!readFbo) { glGenFramebuffers(1, &readFbo); glGenFramebuffers(1, &drawFbo); glGenRenderbuffers(1, &drawRbo); }
    if (rboW != w || rboH != h) {
        glBindRenderbuffer(GL_RENDERBUFFER, drawRbo); glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, (GLsizei)w, (GLsizei)h);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo); glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, drawRbo);
        rboW = w; rboH = h;
    }
    glBindFramebuffer(GL_READ_FRAMEBUFFER, readFbo);
    glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, name, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, drawFbo);
    glDisable(GL_SCISSOR_TEST);
    glBlitFramebuffer(sx0, sy0, sx1, sy1, 0, (GLint)h, (GLint)w, 0, GL_COLOR_BUFFER_BIT, w == (uint32_t)abs(sx1 - sx0) ? GL_NEAREST : GL_LINEAR);

    uint32_t buf = (shm->frame_seq + 1) % 2;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, drawFbo);
    glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)(2 * w)); glPixelStorei(GL_PACK_ALIGNMENT, 4);
    if (slot) {
        size_t size = (size_t)2 * w * h * 4;
        if (!slot->pbo) glGenBuffers(1, &slot->pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, slot->pbo);
        if (slot->size != size) { glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)size, NULL, GL_STREAM_READ); slot->size = size; }
        slot->w = w; slot->h = h; slot->pose[e] = renderHead; slot->time = renderTime;
        glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, (void *)(uintptr_t)(e * w * 4));
    } else {
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, vr4_frame(shm, buf) + (size_t)e * w * 4);
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)prevRead); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)prevDraw);
    glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)prevRb); glBindBuffer(GL_PIXEL_PACK_BUFFER, (GLuint)prevPbo);
    glPixelStorei(GL_PACK_ROW_LENGTH, prevPack); glPixelStorei(GL_PACK_ALIGNMENT, prevAlign);
    if (scissor) glEnable(GL_SCISSOR_TEST);

    if (!slot) {
        shm->frame_eye_pose[buf][e] = renderHead;
        if (e) { VR4Pose p[2] = {shm->frame_eye_pose[buf][0], renderHead}; publish_frame(buf, w, h, p, renderTime); }
    } else if (e) {   // frame complete: hand it to the worker
        GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        glFlush();   // the worker's context only sees the fence once it is flushed
        __atomic_store_n(&slot->busy, 1, __ATOMIC_RELEASE);
        pboCur = -1;
        CGLContextObj ctx = workerCtx;
        dispatch_async(worker_queue(), ^{
            CGLSetCurrentContext(ctx);
            if (glClientWaitSync(fence, 0, 1000000000ull) != GL_TIMEOUT_EXPIRED) {
                glBindBuffer(GL_PIXEL_PACK_BUFFER, slot->pbo);
                const void *src = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)slot->size, GL_MAP_READ_BIT);
                if (src) {
                    uint32_t fb = (shm->frame_seq + 1) % 2;
                    memcpy(vr4_frame(shm, fb), src, slot->size);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                    publish_frame(fb, slot->w, slot->h, slot->pose, slot->time);
                }
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            }
            glDeleteSync(fence);
            CGLSetCurrentContext(NULL);
            __atomic_store_n(&slot->busy, 0, __ATOMIC_RELEASE);
        });
    }
    return EVRCompositorError_VRCompositorError_None;
}
static EVRCompositorError SubmitWithArrayIndex(EVREye eye, Texture_t *tex, uint32_t layer, VRTextureBounds_t *bounds, EVRSubmitFlags flags) {
    if (layer) { logmsg("SubmitWithArrayIndex layer %u unsupported", layer); return EVRCompositorError_VRCompositorError_TextureUsesUnsupportedFormat; }
    return Submit(eye, tex, bounds, flags);
}
static void PostPresentHandoff(void) {}
static void ClearLastSubmittedFrame(void) {}
static float GetFrameTimeRemaining(void) { uint64_t t = now_ns(); return nextVsync > t ? (nextVsync - t) / 1e9f : 0; }
static bool CanRenderScene(void) { return true; }
static bool IsFullscreen(void) { return true; }
static void SuspendRendering(bool b) { (void)b; }
static uint32_t GetVulkanExtensionsRequired(char *v, uint32_t n) { if (v && n) *v = 0; return 1; }   // both Vulkan variants (2 args)
static uint32_t GetVulkanDeviceExtensionsRequired(void *pd, char *v, uint32_t n) { (void)pd; return GetVulkanExtensionsRequired(v, n); }
static bool IsMotionSmoothingEnabled(void) { return false; }
static EVRCompositorError GetLastPoseForTrackedDeviceIndex(TrackedDeviceIndex_t i, TrackedDevicePose_t *render, TrackedDevicePose_t *game) {
    TrackedDevicePose_t p = device_pose((int)i);
    fill_velocities(&p, 1);
    if (render) *render = p;
    if (game) *game = p;
    return EVRCompositorError_VRCompositorError_None;
}
static struct HmdColor_t GetCurrentFadeColor(bool bBackground) { return fade_color(bBackground != 0); }

// ---------------------------------------------------------------- IVRChaperone / IVRRenderModels / IVRSettings / IVRApplications
// Play area: 2 x 2 m around the standing origin, in game units (world scale), as the OpenXR stage bounds. The walls
// are 2.43 m high (SteamVR's default). MacVR draws no bounds, so they are never visible.
static ChaperoneCalibrationState GetCalibrationState(void) { return ChaperoneCalibrationState_OK; }
static bool GetPlayAreaSize(float *x, float *z) { if (x) *x = 2 / world_scale(); if (z) *z = 2 / world_scale(); return true; }
static bool GetPlayAreaRect(HmdQuad_t *q) {
    float c[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}}, s = 1 / world_scale();
    for (int i = 0; q && i < 4; i++) q->vCorners[i] = (HmdVector3_t){{c[i][0] * s, 0, c[i][1] * s}};
    return q != NULL;
}
static bool GetCollisionBounds(HmdQuad_t *quads, uint32_t *n) {   // one wall per side of the play area
    uint32_t cap = n ? *n : 0;
    if (n) *n = 4;
    HmdQuad_t r;
    if (!quads || cap < 4 || !GetPlayAreaRect(&r)) return false;
    for (int i = 0; i < 4; i++) {
        HmdVector3_t a = r.vCorners[i], b = r.vCorners[(i + 1) % 4], ah = a, bh = b;
        ah.v[1] = bh.v[1] = 2.43f / world_scale();
        quads[i] = (HmdQuad_t){{a, b, bh, ah}};
    }
    return true;
}
static bool GetStandingZeroPose(HmdMatrix34_t *m) { if (m) *m = (HmdMatrix34_t){{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}}; return m != NULL; }
static bool AreBoundsVisible(void) { return false; }
static void GetBoundsColor(HmdColor_t *colors, int n, float fade, HmdColor_t *camera) {
    (void)fade;
    for (int i = 0; colors && i < n; i++) colors[i] = (HmdColor_t){0, 0.6f, 1, 1};   // SteamVR's default cyan
    if (camera) *camera = (HmdColor_t){0, 0.6f, 1, 1};
}

static uint32_t GetRenderModelCount(void) { return 2; }
static uint64_t GetComponentButtonMask(char *model, char *comp) { (void)model; (void)comp; return 0; }
/// "handgrip" = the grip pose itself, "tip" = the aim pose in grip space (where Vivecraft aims and shoots from).
static bool GetComponentStateForDevicePath(char *model, char *comp, VRInputValueHandle_t dev, RenderModel_ControllerMode_State_t *st, RenderModel_ComponentState_t *cs) {
    (void)model; (void)st;
    if ((dev != 1 && dev != 2) || !cs) return false;
    read_tracking();
    const VR4Hand *h = &track.hand[dev - 1];
    HmdMatrix34_t local = !strcmp(comp, "tip") && hand_valid((int)dev - 1) ? mul34(inv34(mat(h->grip)), mat(h->aim)) : identity34();
    if (!strcmp(comp, "tip") && !hand_valid((int)dev - 1)) { local.m[1][3] = -0.03f; local.m[2][3] = -0.05f; }
    cs->mTrackingToComponentLocal = cs->mTrackingToComponentRenderModel = local;
    cs->uProperties = EVRComponentProperty_VRComponentProperty_IsStatic | EVRComponentProperty_VRComponentProperty_IsVisible;
    return true;
}

static float GetFloat(char *section, char *key, EVRSettingsError *err) {
    if (err) *err = EVRSettingsError_VRSettingsError_None;
    if (!strcmp(key, "supersampleScale")) return 1;
    (void)section;
    return 0;
}
static bool GetBool(char *section, char *key, EVRSettingsError *err) { (void)section; (void)key; if (err) *err = 0; return false; }
static int32_t GetInt32(char *section, char *key, EVRSettingsError *err) { (void)section; (void)key; if (err) *err = 0; return 0; }
static void GetString(char *section, char *key, char *v, uint32_t n, EVRSettingsError *err) { (void)section; (void)key; if (v && n) *v = 0; if (err) *err = 0; }

static EVRApplicationError AddApplicationManifest(char *path, bool temp) { (void)path; (void)temp; return EVRApplicationError_VRApplicationError_None; }
static bool IsApplicationInstalled(char *key) { (void)key; return true; }
static EVRApplicationError IdentifyApplication(uint32_t pid, char *key) { (void)pid; logmsg("app %s", key); return EVRApplicationError_VRApplicationError_None; }
static char *GetApplicationsErrorNameFromEnum(EVRApplicationError e) { return e ? "VRApplicationError_Unknown" : "VRApplicationError_None"; }

// ---------------------------------------------------------------- IVROverlay
// Overlays are bookkept (keys, names, flags, colors, sizes, transforms, visibility, textures) so apps that create them
// get consistent answers, and ComputeOverlayIntersection hit-tests them. MacVR has no system keyboard or message box:
// ShowKeyboard fails with RequestFailed, so apps (Vivecraft) use their own keyboards.
// ponytail: overlay textures are not drawn into the frames; composite them in Submit if an app needs it.
typedef struct {
    int used, visible; char key[256], name[256]; uint32_t flags, sortOrder; float color[3], alpha, texelAspect, width, curvature, pitch;
    EColorSpace colorSpace; VRTextureBounds_t bounds; VROverlayTransformType ttype; ETrackingUniverseOrigin origin; HmdMatrix34_t xf;
    TrackedDeviceIndex_t device; Texture_t tex; int hasTex; VROverlayInputMethod input; HmdVector2_t mouseScale;
} Overlay;
static Overlay overlays[64];
static Overlay *ov(VROverlayHandle_t h) { return h >= 1 && h <= 64 && overlays[h - 1].used ? &overlays[h - 1] : NULL; }
#define OV(h) Overlay *o = ov(h); if (!o) return EVROverlayError_VROverlayError_UnknownOverlay
static EVROverlayError FindOverlay(char *key, VROverlayHandle_t *h) {
    for (int i = 0; i < 64; i++) if (overlays[i].used && !strcmp(overlays[i].key, key)) { *h = (VROverlayHandle_t)i + 1; return 0; }
    *h = 0; return EVROverlayError_VROverlayError_UnknownOverlay;
}
static EVROverlayError CreateOverlay(char *key, char *name, VROverlayHandle_t *h) {
    *h = 0;
    if (strlen(key) >= 256) return EVROverlayError_VROverlayError_KeyTooLong;
    if (strlen(name) >= 256) return EVROverlayError_VROverlayError_NameTooLong;
    VROverlayHandle_t found;
    if (FindOverlay(key, &found) == 0) return EVROverlayError_VROverlayError_KeyInUse;
    for (int i = 0; i < 64; i++) if (!overlays[i].used) {
        overlays[i] = (Overlay){.used = 1, .color = {1, 1, 1}, .alpha = 1, .texelAspect = 1, .width = 1, .bounds = {0, 0, 1, 1},
                                .ttype = VROverlayTransformType_VROverlayTransform_Absolute, .origin = ETrackingUniverseOrigin_TrackingUniverseStanding,
                                .xf = {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}}, .mouseScale = {{1, 1}}};
        snprintf(overlays[i].key, 256, "%s", key); snprintf(overlays[i].name, 256, "%s", name);
        *h = (VROverlayHandle_t)i + 1; return 0;
    }
    return EVROverlayError_VROverlayError_OverlayLimitExceeded;
}
static EVROverlayError DestroyOverlay(VROverlayHandle_t h) { OV(h); o->used = 0; return 0; }
static uint32_t copy_str(const char *s, char *v, uint32_t size) { uint32_t n = (uint32_t)strlen(s) + 1; if (v && size >= n) memcpy(v, s, n); return n; }
static uint32_t GetOverlayKey(VROverlayHandle_t h, char *v, uint32_t size, EVROverlayError *err) {
    Overlay *o = ov(h); if (err) *err = o ? 0 : EVROverlayError_VROverlayError_UnknownOverlay; return o ? copy_str(o->key, v, size) : 0;
}
static uint32_t GetOverlayName(VROverlayHandle_t h, char *v, uint32_t size, EVROverlayError *err) {
    Overlay *o = ov(h); if (err) *err = o ? 0 : EVROverlayError_VROverlayError_UnknownOverlay; return o ? copy_str(o->name, v, size) : 0;
}
static EVROverlayError SetOverlayName(VROverlayHandle_t h, char *name) { OV(h); snprintf(o->name, 256, "%s", name); return 0; }
static char *GetOverlayErrorNameFromEnum(EVROverlayError e) {
    switch ((int)e) {
    case 0: return "VROverlayError_None"; case 10: return "VROverlayError_UnknownOverlay"; case 13: return "VROverlayError_OverlayLimitExceeded";
    case 17: return "VROverlayError_KeyInUse"; case 18: return "VROverlayError_WrongTransformType"; case 23: return "VROverlayError_RequestFailed";
    default: return "VROverlayError_Unknown";
    }
}
static EVROverlayError SetOverlayFlag(VROverlayHandle_t h, VROverlayFlags f, bool on) {
    OV(h); if ((unsigned)f > 31) return EVROverlayError_VROverlayError_InvalidParameter;
    o->flags = on ? o->flags | 1u << f : o->flags & ~(1u << f); return 0;
}
static EVROverlayError GetOverlayFlag(VROverlayHandle_t h, VROverlayFlags f, bool *on) { OV(h); *on = (unsigned)f < 32 && (o->flags >> f & 1); return 0; }
static EVROverlayError GetOverlayFlags(VROverlayHandle_t h, uint32_t *f) { OV(h); *f = o->flags; return 0; }
static EVROverlayError SetOverlayColor(VROverlayHandle_t h, float r, float g, float b) { OV(h); o->color[0] = r; o->color[1] = g; o->color[2] = b; return 0; }
static EVROverlayError GetOverlayColor(VROverlayHandle_t h, float *r, float *g, float *b) { OV(h); *r = o->color[0]; *g = o->color[1]; *b = o->color[2]; return 0; }
#define SETGET(Name, field, T) \
    static EVROverlayError SetOverlay##Name(VROverlayHandle_t h, T v) { OV(h); o->field = v; return 0; } \
    static EVROverlayError GetOverlay##Name(VROverlayHandle_t h, T *v) { OV(h); if (v) *v = o->field; return 0; }
SETGET(Alpha, alpha, float) SETGET(TexelAspect, texelAspect, float) SETGET(SortOrder, sortOrder, uint32_t)
SETGET(Curvature, curvature, float) SETGET(PreCurvePitch, pitch, float) SETGET(TextureColorSpace, colorSpace, EColorSpace)
SETGET(InputMethod, input, VROverlayInputMethod)
#undef SETGET
static EVROverlayError SetOverlayWidthInMeters(VROverlayHandle_t h, float w) { OV(h); if (w <= 0) return EVROverlayError_VROverlayError_InvalidParameter; o->width = w; return 0; }
static EVROverlayError GetOverlayWidthInMeters(VROverlayHandle_t h, float *w) { OV(h); *w = o->width; return 0; }
static EVROverlayError SetOverlayTextureBounds(VROverlayHandle_t h, VRTextureBounds_t *b) { OV(h); o->bounds = *b; return 0; }
static EVROverlayError GetOverlayTextureBounds(VROverlayHandle_t h, VRTextureBounds_t *b) { OV(h); *b = o->bounds; return 0; }
static EVROverlayError GetOverlayTransformType(VROverlayHandle_t h, VROverlayTransformType *t) { OV(h); *t = o->ttype; return 0; }
static EVROverlayError SetOverlayTransformAbsolute(VROverlayHandle_t h, ETrackingUniverseOrigin origin, HmdMatrix34_t *m) {
    OV(h); o->ttype = VROverlayTransformType_VROverlayTransform_Absolute; o->origin = origin; o->xf = *m; return 0;
}
static EVROverlayError GetOverlayTransformAbsolute(VROverlayHandle_t h, ETrackingUniverseOrigin *origin, HmdMatrix34_t *m) {
    OV(h); if (o->ttype != VROverlayTransformType_VROverlayTransform_Absolute) return EVROverlayError_VROverlayError_WrongTransformType;
    *origin = o->origin; *m = o->xf; return 0;
}
static EVROverlayError SetOverlayTransformTrackedDeviceRelative(VROverlayHandle_t h, TrackedDeviceIndex_t d, HmdMatrix34_t *m) {
    OV(h); if (d > 2) return EVROverlayError_VROverlayError_InvalidTrackedDevice;
    o->ttype = VROverlayTransformType_VROverlayTransform_TrackedDeviceRelative; o->device = d; o->xf = *m; return 0;
}
static EVROverlayError GetOverlayTransformTrackedDeviceRelative(VROverlayHandle_t h, TrackedDeviceIndex_t *d, HmdMatrix34_t *m) {
    OV(h); if (o->ttype != VROverlayTransformType_VROverlayTransform_TrackedDeviceRelative) return EVROverlayError_VROverlayError_WrongTransformType;
    *d = o->device; *m = o->xf; return 0;
}
static EVROverlayError ShowOverlay(VROverlayHandle_t h) { OV(h); o->visible = 1; return 0; }
static EVROverlayError HideOverlay(VROverlayHandle_t h) { OV(h); o->visible = 0; return 0; }
static bool IsOverlayVisible(VROverlayHandle_t h) { Overlay *o = ov(h); return o && o->visible; }
static bool PollNextOverlayEvent(VROverlayHandle_t h, struct VREvent_t *e, uint32_t size) { (void)h; (void)e; (void)size; return false; }
static EVROverlayError SetOverlayMouseScale(VROverlayHandle_t h, HmdVector2_t *s) { OV(h); o->mouseScale = *s; return 0; }
static EVROverlayError GetOverlayMouseScale(VROverlayHandle_t h, HmdVector2_t *s) { OV(h); *s = o->mouseScale; return 0; }
static EVROverlayError SetOverlayTexture(VROverlayHandle_t h, Texture_t *t) { OV(h); if (!t) return EVROverlayError_VROverlayError_InvalidTexture; o->tex = *t; o->hasTex = 1; return 0; }
static EVROverlayError ClearOverlayTexture(VROverlayHandle_t h) { OV(h); o->hasTex = 0; return 0; }
/// Overlay -> tracking space: absolute, or relative to a device's current pose.
static HmdMatrix34_t overlay_xf(Overlay *o) {
    if (o->ttype != VROverlayTransformType_VROverlayTransform_TrackedDeviceRelative) return o->xf;
    read_tracking();
    return mul34(mat(o->device == 0 ? track.head : track.hand[o->device - 1].grip), o->xf);
}
/// Ray (tracking space) against the overlay's quad: width x width * aspect of its texture bounds, facing +Z.
static bool ComputeOverlayIntersection(VROverlayHandle_t h, VROverlayIntersectionParams_t *p, VROverlayIntersectionResults_t *r) {
    Overlay *o = ov(h);
    if (!o || !o->visible) return false;
    HmdMatrix34_t m = overlay_xf(o), inv = inv34(m);
    float s[3], d[3];
    for (int i = 0; i < 3; i++) {
        s[i] = inv.m[i][0] * p->vSource.v[0] + inv.m[i][1] * p->vSource.v[1] + inv.m[i][2] * p->vSource.v[2] + inv.m[i][3];
        d[i] = inv.m[i][0] * p->vDirection.v[0] + inv.m[i][1] * p->vDirection.v[1] + inv.m[i][2] * p->vDirection.v[2];
    }
    if (fabsf(d[2]) < 1e-6f) return false;
    float t = -s[2] / d[2], x = s[0] + t * d[0], y = s[1] + t * d[1];
    float bw = fabsf(o->bounds.uMax - o->bounds.uMin), bh = fabsf(o->bounds.vMax - o->bounds.vMin);
    float hgt = o->width * (bw > 0 ? bh / bw : 1) * o->texelAspect;   // texture aspect is not known: square texels on the bounds
    if (t < 0 || fabsf(x) > o->width / 2 || fabsf(y) > hgt / 2) return false;
    for (int i = 0; i < 3; i++) {
        r->vPoint.v[i] = m.m[i][0] * x + m.m[i][1] * y + m.m[i][3];
        r->vNormal.v[i] = m.m[i][2];
    }
    r->vUVs = (HmdVector2_t){{x / o->width + 0.5f, y / hgt + 0.5f}};
    r->fDistance = t * sqrtf(p->vDirection.v[0] * p->vDirection.v[0] + p->vDirection.v[1] * p->vDirection.v[1] + p->vDirection.v[2] * p->vDirection.v[2]);
    return true;
}
static EVROverlayError CreateDashboardOverlay(char *key, char *name, VROverlayHandle_t *main, VROverlayHandle_t *thumb) {
    char tk[300]; snprintf(tk, sizeof tk, "%s.thumbnail", key);
    EVROverlayError e = CreateOverlay(key, name, main);
    if (!e && (e = CreateOverlay(tk, name, thumb))) { DestroyOverlay(*main); *main = 0; }
    return e;
}
static bool IsDashboardVisible(void) { return shm && shm->input_blocked; }   // MacVR's menu
static bool IsActiveDashboardOverlay(VROverlayHandle_t h) { (void)h; return false; }
static TrackedDeviceIndex_t GetPrimaryDashboardDevice(void) { return k_unTrackedDeviceIndexInvalid; }
static EVROverlayError ShowKeyboard(EGamepadTextInputMode m, EGamepadTextInputLineMode l, uint32_t f, char *d, uint32_t n, char *t, uint64_t u) {
    (void)m; (void)l; (void)f; (void)d; (void)n; (void)t; (void)u; return EVROverlayError_VROverlayError_RequestFailed;
}
static EVROverlayError ShowKeyboardForOverlay(VROverlayHandle_t h, EGamepadTextInputMode m, EGamepadTextInputLineMode l, uint32_t f, char *d, uint32_t n, char *t, uint64_t u) {
    (void)h; return ShowKeyboard(m, l, f, d, n, t, u);
}
static uint32_t GetKeyboardText(char *t, uint32_t n) { if (t && n) *t = 0; return 1; }
static VRMessageOverlayResponse ShowMessageOverlay(char *t, char *c, char *b0, char *b1, char *b2, char *b3) {
    logmsg("message overlay: %s: %s", c ? c : "", t ? t : ""); (void)b0; (void)b1; (void)b2; (void)b3;
    return VRMessageOverlayResponse_CouldntFindSystemOverlay;
}

// ---------------------------------------------------------------- tables
static uintptr_t stub(void) { return 0; }   // every slot we don't implement: returns 0 / false / None
#define FILL(t) for (size_t _i = 0; _i < sizeof(t) / sizeof(void *); _i++) ((void **)&(t))[_i] = (void *)stub
static struct VR_IVRSystem_FnTable sys;
static struct VR_IVRCompositor_FnTable comp;
static struct VR_IVRInput_FnTable input;
static struct VR_IVRChaperone_FnTable chap;
static struct VR_IVRChaperoneSetup_FnTable chapSetup;
static struct VR_IVRRenderModels_FnTable models;
static struct VR_IVRSettings_FnTable settings;
static struct VR_IVRApplications_FnTable apps;
static struct VR_IVROverlay_FnTable overlay;
static void *comp028[sizeof comp / sizeof(void *) + 1];   // IVRCompositor_028 (OpenVR 2.x) = 027 + SubmitWithArrayIndex after Submit

static void build_tables(void) {
    FILL(sys); FILL(comp); FILL(input); FILL(chap); FILL(chapSetup); FILL(models); FILL(settings); FILL(apps); FILL(overlay);
    sys.GetRecommendedRenderTargetSize = GetRecommendedRenderTargetSize; sys.GetProjectionMatrix = GetProjectionMatrix;
    sys.GetProjectionRaw = GetProjectionRaw; sys.ComputeDistortion = ComputeDistortion; sys.GetEyeToHeadTransform = GetEyeToHeadTransform;
    sys.GetDeviceToAbsoluteTrackingPose = GetDeviceToAbsoluteTrackingPose;
    sys.GetSeatedZeroPoseToStandingAbsoluteTrackingPose = GetSeatedZeroPoseToStandingAbsoluteTrackingPose;
    sys.GetRawZeroPoseToStandingAbsoluteTrackingPose = GetSeatedZeroPoseToStandingAbsoluteTrackingPose;
    sys.GetTrackedDeviceIndexForControllerRole = GetTrackedDeviceIndexForControllerRole;
    sys.GetControllerRoleForTrackedDeviceIndex = GetControllerRoleForTrackedDeviceIndex;
    sys.GetTrackedDeviceClass = GetTrackedDeviceClass; sys.IsTrackedDeviceConnected = IsTrackedDeviceConnected;
    sys.GetTrackedDeviceActivityLevel = GetTrackedDeviceActivityLevel;
    sys.GetBoolTrackedDeviceProperty = GetBoolTrackedDeviceProperty; sys.GetFloatTrackedDeviceProperty = GetFloatTrackedDeviceProperty;
    sys.GetInt32TrackedDeviceProperty = GetInt32TrackedDeviceProperty; sys.GetUint64TrackedDeviceProperty = GetUint64TrackedDeviceProperty;
    sys.GetStringTrackedDeviceProperty = GetStringTrackedDeviceProperty;
    sys.PollNextEvent = PollNextEvent; sys.GetHiddenAreaMesh = GetHiddenAreaMesh;
    sys.GetControllerState = GetControllerState; sys.GetControllerStateWithPose = GetControllerStateWithPose;
    sys.TriggerHapticPulse = TriggerHapticPulse;
    sys.ShouldApplicationPause = ShouldApplicationPause; sys.IsInputAvailable = IsInputAvailable;

    comp.SetTrackingSpace = SetTrackingSpace; comp.GetTrackingSpace = GetTrackingSpace; comp.WaitGetPoses = WaitGetPoses;
    comp.GetLastPoses = GetLastPoses; comp.GetLastPoseForTrackedDeviceIndex = GetLastPoseForTrackedDeviceIndex;
    comp.Submit = Submit; comp.PostPresentHandoff = PostPresentHandoff;
    comp.ClearLastSubmittedFrame = ClearLastSubmittedFrame; comp.GetFrameTimeRemaining = GetFrameTimeRemaining;
    comp.FadeToColor = FadeToColor; comp.GetCurrentFadeColor = GetCurrentFadeColor;
    comp.CanRenderScene = CanRenderScene; comp.IsFullscreen = IsFullscreen; comp.SuspendRendering = SuspendRendering;
    comp.GetVulkanInstanceExtensionsRequired = GetVulkanExtensionsRequired;
    comp.GetVulkanDeviceExtensionsRequired = (void *)GetVulkanDeviceExtensionsRequired;
    comp.IsMotionSmoothingEnabled = IsMotionSmoothingEnabled;

    input.SetActionManifestPath = SetActionManifestPath; input.GetActionSetHandle = GetActionSetHandle;
    input.GetActionHandle = GetActionHandle; input.GetInputSourceHandle = GetInputSourceHandle;
    input.UpdateActionState = UpdateActionState; input.GetDigitalActionData = GetDigitalActionData;
    input.GetAnalogActionData = GetAnalogActionData; input.GetPoseActionDataForNextFrame = GetPoseActionDataForNextFrame;
    input.GetActionOrigins = GetActionOrigins; input.GetOriginLocalizedName = GetOriginLocalizedName;
    input.GetOriginTrackedDeviceInfo = GetOriginTrackedDeviceInfo; input.TriggerHapticVibrationAction = TriggerHapticVibrationAction;
    input.GetPoseActionDataRelativeToNow = GetPoseActionDataRelativeToNow;
    input.GetSkeletalActionData = GetSkeletalActionData; input.GetDominantHand = GetDominantHand; input.SetDominantHand = SetDominantHand;
    input.GetBoneCount = GetBoneCount; input.GetBoneHierarchy = GetBoneHierarchy; input.GetBoneName = GetBoneName;
    input.GetSkeletalReferenceTransforms = GetSkeletalReferenceTransforms; input.GetSkeletalTrackingLevel = GetSkeletalTrackingLevel;
    input.GetSkeletalBoneData = GetSkeletalBoneData; input.GetSkeletalSummaryData = GetSkeletalSummaryData;
    input.GetSkeletalBoneDataCompressed = GetSkeletalBoneDataCompressed; input.DecompressSkeletalBoneData = DecompressSkeletalBoneData;

    chap.GetCalibrationState = GetCalibrationState; chap.GetPlayAreaSize = GetPlayAreaSize;
    chap.GetPlayAreaRect = GetPlayAreaRect; chap.AreBoundsVisible = AreBoundsVisible; chap.GetBoundsColor = GetBoundsColor;
    chapSetup.GetWorkingPlayAreaSize = GetPlayAreaSize; chapSetup.GetWorkingPlayAreaRect = GetPlayAreaRect;
    chapSetup.GetWorkingCollisionBoundsInfo = GetCollisionBounds; chapSetup.GetLiveCollisionBoundsInfo = GetCollisionBounds;
    chapSetup.GetWorkingSeatedZeroPoseToRawTrackingPose = GetStandingZeroPose; chapSetup.GetWorkingStandingZeroPoseToRawTrackingPose = GetStandingZeroPose;
    chapSetup.GetLiveSeatedZeroPoseToRawTrackingPose = GetStandingZeroPose;
    models.GetRenderModelCount = GetRenderModelCount; models.GetComponentButtonMask = GetComponentButtonMask;
    models.GetComponentStateForDevicePath = GetComponentStateForDevicePath;
    settings.GetFloat = GetFloat; settings.GetBool = GetBool; settings.GetInt32 = GetInt32; settings.GetString = GetString;
    size_t at = offsetof(struct VR_IVRCompositor_FnTable, Submit) / sizeof(void *) + 1;
    memcpy(comp028, &comp, at * sizeof(void *));
    comp028[at] = (void *)SubmitWithArrayIndex;
    memcpy(comp028 + at + 1, (void **)&comp + at, sizeof comp - at * sizeof(void *));
    apps.AddApplicationManifest = AddApplicationManifest; apps.IsApplicationInstalled = IsApplicationInstalled;
    apps.IdentifyApplication = IdentifyApplication; apps.GetApplicationsErrorNameFromEnum = GetApplicationsErrorNameFromEnum;
    #define O(n) overlay.n = n;
    O(FindOverlay) O(CreateOverlay) O(DestroyOverlay) O(GetOverlayKey) O(GetOverlayName) O(SetOverlayName) O(GetOverlayErrorNameFromEnum)
    O(SetOverlayFlag) O(GetOverlayFlag) O(GetOverlayFlags) O(SetOverlayColor) O(GetOverlayColor) O(SetOverlayAlpha) O(GetOverlayAlpha)
    O(SetOverlayTexelAspect) O(GetOverlayTexelAspect) O(SetOverlaySortOrder) O(GetOverlaySortOrder) O(SetOverlayWidthInMeters)
    O(GetOverlayWidthInMeters) O(SetOverlayCurvature) O(GetOverlayCurvature) O(SetOverlayPreCurvePitch) O(GetOverlayPreCurvePitch)
    O(SetOverlayTextureColorSpace) O(GetOverlayTextureColorSpace) O(SetOverlayTextureBounds) O(GetOverlayTextureBounds)
    O(GetOverlayTransformType) O(SetOverlayTransformAbsolute) O(GetOverlayTransformAbsolute) O(SetOverlayTransformTrackedDeviceRelative)
    O(GetOverlayTransformTrackedDeviceRelative) O(ShowOverlay) O(HideOverlay) O(IsOverlayVisible) O(PollNextOverlayEvent)
    O(GetOverlayInputMethod) O(SetOverlayInputMethod) O(GetOverlayMouseScale) O(SetOverlayMouseScale) O(ComputeOverlayIntersection)
    O(SetOverlayTexture) O(ClearOverlayTexture) O(CreateDashboardOverlay) O(IsDashboardVisible) O(IsActiveDashboardOverlay)
    O(GetPrimaryDashboardDevice) O(ShowKeyboard) O(ShowKeyboardForOverlay) O(GetKeyboardText) O(ShowMessageOverlay)
    #undef O
}

// ---------------------------------------------------------------- exported VR_* entry points
static intptr_t initToken;
EXPORT intptr_t VR_InitInternal2(EVRInitError *err, EVRApplicationType type, const char *startupInfo) {
    (void)startupInfo;
    if (!shm) {
        const char *path = getenv("VR4MAC_SHM") ? getenv("VR4MAC_SHM") : VR4_SHM_PATH_MAC;   // override for tests
        int fd = open(path, O_RDWR);
        void *p = fd >= 0 ? mmap(NULL, VR4_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0) : MAP_FAILED;
        if (fd >= 0) close(fd);
        if (p != MAP_FAILED && ((VR4Shm *)p)->magic == VR4_SHM_MAGIC && ((VR4Shm *)p)->version == VR4_SHM_VERSION) shm = p;
        else if (p != MAP_FAILED) munmap(p, VR4_SHM_SIZE);
    }
    if (!shm) {
        logmsg("init failed: MacVR shared memory not found (is MacVR running?)");
        if (err) *err = EVRInitError_VRInitError_Init_HmdNotFound;
        return 0;
    }
    build_tables();
    @autoreleasepool { publish_app_name(); }
    logmsg("init ok, app type %d, eye %ux%u @ %.0f Hz", type, shm->eye_w, shm->eye_h, fps());
    if (err) *err = EVRInitError_VRInitError_None;
    return ++initToken;
}
EXPORT intptr_t VR_InitInternal(EVRInitError *err, EVRApplicationType type) { return VR_InitInternal2(err, type, NULL); }
EXPORT void VR_ShutdownInternal(void) { logmsg("shutdown"); }
EXPORT bool VR_IsHmdPresent(void) { return access(VR4_SHM_PATH_MAC, F_OK) == 0; }
EXPORT bool VR_IsRuntimeInstalled(void) { return true; }
EXPORT const char *VR_RuntimePath(void) { return "/Applications/MacVR.app"; }
EXPORT bool VR_GetRuntimePath(char *buf, uint32_t size, uint32_t *needed) {
    const char *p = VR_RuntimePath(); uint32_t n = (uint32_t)strlen(p) + 1;
    if (needed) *needed = n;
    if (!buf || size < n) return false;
    memcpy(buf, p, n); return true;
}
EXPORT intptr_t VR_GetInitToken(void) { return initToken; }
EXPORT const char *VR_GetVRInitErrorAsSymbol(EVRInitError e) { return e == 0 ? "VRInitError_None" : e == 108 ? "VRInitError_Init_HmdNotFound" : "VRInitError_Unknown"; }
EXPORT const char *VR_GetVRInitErrorAsEnglishDescription(EVRInitError e) {
    return e == 0 ? "No error" : e == 108 ? "MacVR is not running. Open MacVR and connect your headset, then enable VR again." : "MacVR OpenVR error";
}
static const struct { const char *name; void *table; } interfaces[] = {
    {"IVRSystem_022", &sys}, {"IVRSystem_021", &sys}, {"IVRSystem_020", &sys}, {"IVRSystem_019", &sys},
    {"IVRCompositor_028", comp028}, {"IVRCompositor_027", &comp}, {"IVRCompositor_026", &comp},
    {"IVRCompositor_022", &comp}, {"IVRCompositor_021", &comp}, {"IVRCompositor_020", &comp},
    {"IVRCompositor_019", &comp}, {"IVRCompositor_018", &comp}, {"IVRCompositor_017", &comp},
    {"IVRInput_010", &input}, {"IVRInput_007", &input}, {"IVRInput_006", &input}, {"IVRInput_005", &input}, {"IVRInput_004", &input},
    {"IVRChaperone_004", &chap}, {"IVRChaperone_003", &chap},
    {"IVRChaperoneSetup_006", &chapSetup}, {"IVRChaperoneSetup_005", &chapSetup},
    {"IVRRenderModels_006", &models}, {"IVRRenderModels_005", &models},
    {"IVRSettings_003", &settings}, {"IVRSettings_002", &settings}, {"IVRSettings_001", &settings},
    {"IVROverlay_026", &overlay},
    {"IVRApplications_007", &apps}, {"IVRApplications_006", &apps}, {"IVRApplications_005", &apps},
};
EXPORT bool VR_IsInterfaceVersionValid(const char *v) {
    for (size_t i = 0; i < sizeof interfaces / sizeof *interfaces; i++) if (!strcmp(v, interfaces[i].name)) return true;
    return false;
}
EXPORT void *VR_GetGenericInterface(const char *v, EVRInitError *err) {
    const char *name = !strncmp(v, "FnTable:", 8) ? v + 8 : v;
    for (size_t i = 0; i < sizeof interfaces / sizeof *interfaces; i++)
        if (!strcmp(name, interfaces[i].name)) { if (err) *err = EVRInitError_VRInitError_None; return interfaces[i].table; }
    logmsg("interface %s not provided", v);
    if (err) *err = EVRInitError_VRInitError_Init_InterfaceNotFound;
    return NULL;
}
