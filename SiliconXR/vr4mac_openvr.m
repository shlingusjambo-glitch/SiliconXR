// MacVR native OpenVR runtime: libopenvr_api.dylib for macOS apps that speak OpenVR directly (Vivecraft via
// LWJGL). Implements the "FnTable:" C interfaces LWJGL binds to, backed by the MacVR shared memory
// (common/vr4mac.h): poses + buttons come from the Mac app, OpenGL eye textures are read back into the
// side-by-side frame buffers the Mac app streams to the headset. Input follows the app's SteamVR action
// manifest and its oculus_touch default bindings.
#define GL_SILENCE_DEPRECATION
#import <Foundation/Foundation.h>
#include <OpenGL/gl3.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <stdarg.h>
#include <unistd.h>
#include "openvr_capi.h"
#include "../common/vr4mac.h"

#define EXPORT __attribute__((visibility("default")))

_Static_assert(sizeof(InputDigitalActionData_t) == 24 && sizeof(InputAnalogActionData_t) == 48, "LWJGL layout");
_Static_assert(sizeof(InputPoseActionData_t) == 96 && sizeof(TrackedDevicePose_t) == 80, "LWJGL layout");
_Static_assert(sizeof(InputOriginInfo_t) == 144 && sizeof(VRActiveActionSet_t) == 32, "LWJGL layout");

static VR4Shm *shm;
static VR4Tracking track;              // latest snapshot (WaitGetPoses / UpdateActionState)
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

static void read_tracking(void) {      // seqlock read of the Mac app's latest sample
    for (int tries = 0; shm && tries < 100; tries++) {
        uint32_t s1 = shm->track_seq; vr4_fence();
        if (s1 & 1) continue;
        VR4Tracking t = shm->track; vr4_fence();
        if (shm->track_seq == s1) { track = t; lastSeq = s1; return; }
    }
}

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

// ---------------------------------------------------------------- devices: 0 = HMD, 1 = left hand, 2 = right hand
static int hand_valid(int h) { return (track.hand[h].flags & VR4_HAND_POSE_VALID) != 0; }
static TrackedDevicePose_t prevPose[3]; static uint64_t prevPoseT;
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
    double dt = prevPoseT && track.time_ns > prevPoseT ? (track.time_ns - prevPoseT) / 1e9 : 0;
    for (int i = 0; i < n && i < 3; i++) {
        if (dt > 0 && dt < 0.2)
            for (int k = 0; k < 3; k++) p[i].vVelocity.v[k] = (float)((p[i].mDeviceToAbsoluteTracking.m[k][3] - prevPose[i].mDeviceToAbsoluteTracking.m[k][3]) / dt);
        else p[i].vVelocity = prevPose[i].vVelocity;
    }
    if (track.time_ns != prevPoseT) { for (int i = 0; i < n && i < 3; i++) prevPose[i] = p[i]; prevPoseT = track.time_ns; }
}

// ---------------------------------------------------------------- input: manifest + oculus_touch bindings
enum { MODE_BUTTON, MODE_TRIGGER, MODE_JOYSTICK, MODE_TOGGLE, MODE_SCROLL };
typedef struct { char name[128]; char type[16]; int hand; float x, y, px, py; int state, prevState, origin; } Action;   // hand: pose/haptic actions
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
            if (a >= 0) actions[a].hand = hand_of(p[@"path"]);
        }
        for (NSDictionary *p in s[@"haptics"]) {
            int a = add_action([p[@"output"] UTF8String], "vibration");
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
    return EVRInputError_VRInputError_None;
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
    (void)start;
    Action *a = action(h);
    if (!a) return EVRInputError_VRInputError_InvalidHandle;
    int hand = a->hand >= 0 ? a->hand : r == 1 ? 0 : r == 2 ? 1 : -1;
    if (hand < 0 || !shm) return EVRInputError_VRInputError_None;
    shm->haptic = (VR4Haptics){(uint8_t)hand, amp, dur > 0 ? dur : 0.02f, freq > 0 ? freq : 0};
    vr4_fence(); shm->haptic_seq++;
    return EVRInputError_VRInputError_None;
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

static float GetFloatTrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    if (err) *err = ETrackedPropertyError_TrackedProp_Success;
    if (p == ETrackedDeviceProperty_Prop_DisplayFrequency_Float) return fps();
    if (p == ETrackedDeviceProperty_Prop_UserIpdMeters_Float) {
        HmdMatrix34_t l = GetEyeToHeadTransform(EVREye_Eye_Left), r = GetEyeToHeadTransform(EVREye_Eye_Right);
        return fabsf(r.m[0][3] - l.m[0][3]);
    }
    if (p == ETrackedDeviceProperty_Prop_SecondsFromVsyncToPhotons_Float) return 0.011f;
    (void)i;
    if (err) *err = ETrackedPropertyError_TrackedProp_UnknownProperty;
    return 0;
}
static uint32_t GetStringTrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, char *v, uint32_t size, ETrackedPropertyError *err) {
    const char *s = NULL;
    if (i > 2) { if (err) *err = ETrackedPropertyError_TrackedProp_InvalidDevice; return 0; }
    switch ((int)p) {
    case ETrackedDeviceProperty_Prop_TrackingSystemName_String: s = "oculus"; break;
    case ETrackedDeviceProperty_Prop_ManufacturerName_String: s = "Oculus"; break;
    case ETrackedDeviceProperty_Prop_ModelNumber_String: s = i == 0 ? "Oculus Quest2" : i == 1 ? "Oculus Quest2 (Left Controller)" : "Oculus Quest2 (Right Controller)"; break;
    case ETrackedDeviceProperty_Prop_SerialNumber_String: s = i == 0 ? "MACVR-HMD" : i == 1 ? "MACVR-LEFT" : "MACVR-RIGHT"; break;
    case ETrackedDeviceProperty_Prop_RenderModelName_String: s = i == 0 ? "oculus_quest2" : i == 1 ? "oculus_quest2_controller_left" : "oculus_quest2_controller_right"; break;
    case ETrackedDeviceProperty_Prop_ControllerType_String: s = i == 0 ? "quest2_hmd" : "oculus_touch"; break;
    }
    if (!s) { if (err) *err = ETrackedPropertyError_TrackedProp_UnknownProperty; if (v && size) *v = 0; return 0; }
    uint32_t n = (uint32_t)strlen(s) + 1;
    if (!v || size < n) { if (err) *err = ETrackedPropertyError_TrackedProp_BufferTooSmall; return n; }
    memcpy(v, s, n);
    if (err) *err = ETrackedPropertyError_TrackedProp_Success;
    return n;
}
static bool GetBoolTrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    (void)i; (void)p; if (err) *err = ETrackedPropertyError_TrackedProp_UnknownProperty; return false;
}
static int32_t GetInt32TrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    if (p == ETrackedDeviceProperty_Prop_ControllerRoleHint_Int32) { if (err) *err = 0; return GetControllerRoleForTrackedDeviceIndex(i); }
    if (err) *err = ETrackedPropertyError_TrackedProp_UnknownProperty; return 0;
}
static uint64_t GetUint64TrackedDeviceProperty(TrackedDeviceIndex_t i, ETrackedDeviceProperty p, ETrackedPropertyError *err) {
    (void)i; (void)p; if (err) *err = ETrackedPropertyError_TrackedProp_UnknownProperty; return 0;
}
static HiddenAreaMesh_t GetHiddenAreaMesh(EVREye e, EHiddenAreaMeshType t) {
    (void)e; (void)t;
    static HmdVector2_t degenerate[3];   // one empty triangle: nothing hidden, but never a NULL buffer
    return (HiddenAreaMesh_t){degenerate, 1};
}
static bool PollNextEvent(struct VREvent_t *e, uint32_t size) { (void)e; (void)size; return false; }
static bool ShouldApplicationPause(void) { return false; }
static bool IsInputAvailable(void) { return !(shm && shm->input_blocked); }

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

/// Reads an OpenGL eye texture into this frame's side-by-side shm buffer (top row first), flipping on the GPU
/// with a blit. Publishes the frame after the right eye.
// ponytail: synchronous glReadPixels (one GPU sync per eye); move to a PBO ring if Minecraft frame times suffer.
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
    if (eye == EVREye_Eye_Left || frameW != w || frameH != h) { frameW = w; frameH = h; }

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

    uint32_t buf = (shm->frame_seq + 1) % 2, e = eye == EVREye_Eye_Right;
    glBindFramebuffer(GL_READ_FRAMEBUFFER, drawFbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
    glPixelStorei(GL_PACK_ROW_LENGTH, (GLint)(2 * w)); glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, vr4_frame(shm, buf) + (size_t)e * w * 4);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)prevRead); glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)prevDraw);
    glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)prevRb); glBindBuffer(GL_PIXEL_PACK_BUFFER, (GLuint)prevPbo);
    glPixelStorei(GL_PACK_ROW_LENGTH, prevPack); glPixelStorei(GL_PACK_ALIGNMENT, prevAlign);
    if (scissor) glEnable(GL_SCISSOR_TEST);

    shm->frame_eye_pose[buf][e] = renderHead;
    if (e) {
        shm->frame_w[buf] = 2 * w; shm->frame_h[buf] = h; shm->frame_rgba[buf] = 0;
        shm->frame_time_ns[buf] = renderTime;
        shm->runtime_heartbeat_ns = now_ns();
        vr4_fence();
        shm->frame_seq++;
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

// ---------------------------------------------------------------- IVRChaperone / IVRRenderModels / IVRSettings / IVRApplications
static ChaperoneCalibrationState GetCalibrationState(void) { return ChaperoneCalibrationState_OK; }
static bool GetPlayAreaSize(float *x, float *z) { *x = 2; *z = 2; return true; }
static bool GetPlayAreaRect(HmdQuad_t *q) {
    float c[4][2] = {{-1, -1}, {1, -1}, {1, 1}, {-1, 1}};
    for (int i = 0; i < 4; i++) q->vCorners[i] = (HmdVector3_t){{c[i][0], 0, c[i][1]}};
    return true;
}
static bool AreBoundsVisible(void) { return false; }

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

// ---------------------------------------------------------------- tables
static uintptr_t stub(void) { return 0; }   // every slot we don't implement: returns 0 / false / None
#define FILL(t) for (size_t _i = 0; _i < sizeof(t) / sizeof(void *); _i++) ((void **)&(t))[_i] = (void *)stub
static struct VR_IVRSystem_FnTable sys;
static struct VR_IVRCompositor_FnTable comp;
static struct VR_IVRInput_FnTable input;
static struct VR_IVRChaperone_FnTable chap;
static struct VR_IVRRenderModels_FnTable models;
static struct VR_IVRSettings_FnTable settings;
static struct VR_IVRApplications_FnTable apps;
static void *comp028[sizeof comp / sizeof(void *) + 1];   // IVRCompositor_028 (OpenVR 2.x) = 027 + SubmitWithArrayIndex after Submit

static void build_tables(void) {
    FILL(sys); FILL(comp); FILL(input); FILL(chap); FILL(models); FILL(settings); FILL(apps);
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
    sys.ShouldApplicationPause = ShouldApplicationPause; sys.IsInputAvailable = IsInputAvailable;

    comp.SetTrackingSpace = SetTrackingSpace; comp.GetTrackingSpace = GetTrackingSpace; comp.WaitGetPoses = WaitGetPoses;
    comp.GetLastPoses = GetLastPoses; comp.Submit = Submit; comp.PostPresentHandoff = PostPresentHandoff;
    comp.ClearLastSubmittedFrame = ClearLastSubmittedFrame; comp.GetFrameTimeRemaining = GetFrameTimeRemaining;
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

    chap.GetCalibrationState = GetCalibrationState; chap.GetPlayAreaSize = GetPlayAreaSize;
    chap.GetPlayAreaRect = GetPlayAreaRect; chap.AreBoundsVisible = AreBoundsVisible;
    models.GetRenderModelCount = GetRenderModelCount; models.GetComponentButtonMask = GetComponentButtonMask;
    models.GetComponentStateForDevicePath = GetComponentStateForDevicePath;
    settings.GetFloat = GetFloat; settings.GetBool = GetBool; settings.GetInt32 = GetInt32; settings.GetString = GetString;
    size_t at = offsetof(struct VR_IVRCompositor_FnTable, Submit) / sizeof(void *) + 1;
    memcpy(comp028, &comp, at * sizeof(void *));
    comp028[at] = (void *)SubmitWithArrayIndex;
    memcpy(comp028 + at + 1, (void **)&comp + at, sizeof comp - at * sizeof(void *));
    apps.AddApplicationManifest = AddApplicationManifest; apps.IsApplicationInstalled = IsApplicationInstalled;
    apps.IdentifyApplication = IdentifyApplication; apps.GetApplicationsErrorNameFromEnum = GetApplicationsErrorNameFromEnum;
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
    snprintf(shm->app_name, sizeof shm->app_name, "%s", getprogname() && strcmp(getprogname(), "java") ? getprogname() : "Minecraft");
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
    {"IVRSystem_022", &sys}, {"IVRCompositor_028", comp028}, {"IVRCompositor_027", &comp}, {"IVRCompositor_026", &comp}, {"IVRInput_010", &input},
    {"IVRChaperone_004", &chap}, {"IVRRenderModels_006", &models}, {"IVRSettings_003", &settings}, {"IVRApplications_007", &apps},
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
