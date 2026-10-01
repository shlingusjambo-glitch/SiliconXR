// SiliconXR OpenXR runtime: libsiliconxr_openxr.dylib for games that run natively on macOS. It is the Mac twin of
// WineXR (runtime/vr4mac_openxr.c), and the session, space, action and pacing logic is kept the same. Poses and input come from the MacVR app
// through shared memory (common/vr4mac.h); Metal eye images are read back into the frames the Mac app streams.
// Graphics: XR_KHR_metal_enable. Built without ARC (Metal objects live in C structs).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>
#define XR_USE_GRAPHICS_API_METAL
#define XR_USE_TIMESPEC
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_loader_negotiation.h>
#include <openxr/openxr_reflection.h>
#include "../../common/vr4mac.h"

#define EXPORT __attribute__((visibility("default")))
#define MAXN 256
#define NRING 3   // readback buffers in flight (GPU copies finish asynchronously)

static VR4Shm *shm;
static FILE *logfile;
static void logmsg(const char *fmt, ...) {
    if (!logfile) logfile = fopen("/tmp/vr4mac/siliconxr_openxr.log", "a");
    if (!logfile) return;
    va_list a; va_start(a, fmt); vfprintf(logfile, fmt, a); va_end(a); fputc('\n', logfile); fflush(logfile);
}

static float current_render_scale(void) {
    return (shm && shm->render_scale > 0.01f) ? shm->render_scale : 1.0f;
}
static float current_world_scale(void) {
    return (shm && shm->world_scale > 0.01f) ? shm->world_scale : 1.0f;
}

// ---------------------------------------------------------------- math
typedef XrQuaternionf Q; typedef XrVector3f V;
static Q qmul(Q a, Q b) {
    return (Q){a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
               a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
static Q qconj(Q a) { return (Q){-a.x, -a.y, -a.z, a.w}; }
static V qrot(Q q, V v) { Q r = qmul(qmul(q, (Q){v.x, v.y, v.z, 0}), qconj(q)); return (V){r.x, r.y, r.z}; }
static XrPosef pmul(XrPosef a, XrPosef b) {   // b expressed in a's frame -> a's parent frame
    V t = qrot(a.orientation, b.position);
    return (XrPosef){qmul(a.orientation, b.orientation), {a.position.x + t.x, a.position.y + t.y, a.position.z + t.z}};
}
static inline V vcross(V a, V b) { return (V){a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static inline V vadd(V a, V b) { return (V){a.x + b.x, a.y + b.y, a.z + b.z}; }
static inline V vsub(V a, V b) { return (V){a.x - b.x, a.y - b.y, a.z - b.z}; }
static XrPosef pinv(XrPosef a) {
    Q c = qconj(a.orientation); V t = qrot(c, a.position);
    return (XrPosef){c, {-t.x, -t.y, -t.z}};
}
static XrPosef xp(VR4Pose p) {
    float ws = current_world_scale();
    return (XrPosef){{p.qx, p.qy, p.qz, p.qw}, {p.px / ws, p.py / ws, p.pz / ws}};
}
static const XrPosef IDENT = {{0, 0, 0, 1}, {0, 0, 0}};

// ---------------------------------------------------------------- paths
static char *paths[4096]; static int npaths;
static XrPath intern(const char *s) {
    for (int i = 0; i < npaths; i++) if (!strcmp(paths[i], s)) return i + 1;
    if (npaths == 4096) return XR_NULL_PATH;
    paths[npaths] = strdup(s);
    return ++npaths;
}
static const char *pstr(XrPath p) { return p >= 1 && p <= (XrPath)npaths ? paths[p - 1] : ""; }

// ---------------------------------------------------------------- objects
typedef struct { int hand; char comp[64]; } Binding;
typedef struct { uint32_t gen; float cur[2], prev[2]; } ActionHistory;
typedef struct Action { XrActionType type; char name[64]; XrPath sub[8]; int nsub; Binding b[16]; int nb; ActionHistory hist[3]; } Action;
typedef struct { char name[64]; } ActionSet;
typedef struct { XrPath profile; Action *action; XrPath binding; } Suggestion;
typedef struct {
    XrSession session; int ref; XrReferenceSpaceType type; Action *action; XrPath sub; XrPosef offset;
} Space;
typedef struct {
    id<MTLTexture> img[3]; int count, acquired, released; MTLPixelFormat fmt; uint32_t w, h, array, mips;
} Swapchain;
typedef struct {
    id<MTLCommandQueue> queue; int running, focused, exitRequested;
    id<MTLBuffer> ring[NRING]; volatile int busy[NRING]; size_t ringSize;   // readback buffers, busy while the GPU/copy owns them
    XrPosef localOrigin; V localOriginRaw; XrPath profile;
    uint64_t lastPublished;   // displayTime of the last actually-published frame (duplicate-submit skip)
} Session;

static Suggestion sugg[1024]; static int nsugg;
static VR4Tracking track;           // input snapshot (hands/buttons), refreshed by xrSyncActions
static VR4Tracking frameTrack;      // head/eyes of the latest xrWaitFrame
static VR4Tracking frameRing[4]; static int frameRingN;   // recent xrWaitFrame snapshots, for pipelined apps
static uint32_t lastSeq;
static uint32_t syncGen;   // bumped by xrSyncActions
static XrEventDataBuffer events[64]; static int evHead, evTail;
static Session *theSession;
static char appName[128];
static int64_t qpcOffsetNs;         // XrTime - CLOCK_MONOTONIC time

static int64_t qpc_ns(void) { return (int64_t)clock_gettime_nsec_np(CLOCK_MONOTONIC); }
static dispatch_queue_t publish_queue(void);

static void push_state(XrSessionState st) {
    if (evTail - evHead >= 64) evHead = evTail - 63;
    XrEventDataSessionStateChanged *e = (XrEventDataSessionStateChanged *)&events[evTail % 64];
    memset(e, 0, sizeof(XrEventDataBuffer));
    e->type = XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED; e->session = (XrSession)theSession; e->state = st;
    e->time = qpc_ns() + qpcOffsetNs;
    evTail++;
}

/// Head/eye snapshot handed out by the xrWaitFrame that predicted `t` (falls back to the latest).
static const VR4Tracking *frame_for(XrTime t) {
    for (int i = 0; i < 4; i++) if (frameRing[i].time_ns == (uint64_t)t && t) return &frameRing[i];
    return &frameTrack;
}
static void update_velocities(const VR4Tracking *t);
static int read_tracking(void) {   // seqlock read of the Mac app's latest tracking sample
    for (int tries = 0; tries < 100; tries++) {
        uint32_t s1 = shm->track_seq; vr4_fence();
        if (s1 & 1) continue;
        VR4Tracking t = shm->track; vr4_fence();
        if (shm->track_seq == s1) { int fresh = s1 != lastSeq; track = t; lastSeq = s1; if (fresh) update_velocities(&track); return fresh; }
    }
    return 0;
}

// ---------------------------------------------------------------- input mapping
static int btn(const VR4Hand *h, uint32_t bit) { return (h->buttons & bit) != 0; }
/// Scalar value of an input component path suffix like "trigger/value" or "a/click".
static float comp_value(const VR4Hand *h, const char *c) {
    #define IS(p) (!strncmp(c, p, strlen(p)))
    if (IS("trigger/touch")) return btn(h, VR4_BTN_TRIGGER_TOUCH);
    if (IS("trigger") || IS("select")) return h->trigger;
    if (IS("squeeze") || IS("grip/value") || IS("grip/click") || IS("grip/force") || IS("squeeze/force")) return h->squeeze;
    if (IS("thumbstick/x") || IS("trackpad/x") || IS("joystick/x")) return h->stick_x;
    if (IS("thumbstick/y") || IS("trackpad/y") || IS("joystick/y")) return h->stick_y;
    if (IS("thumbstick/click") || IS("trackpad/click") || IS("joystick/click")) return btn(h, VR4_BTN_STICK_CLICK);
    if (IS("thumbstick/touch") || IS("trackpad/touch") || IS("joystick/touch")) return btn(h, VR4_BTN_STICK_TOUCH);
    if (IS("thumbrest/touch")) return btn(h, VR4_BTN_THUMB_TOUCH);
    if (IS("a/click")) return btn(h, VR4_BTN_A);
    if (IS("b/click")) return btn(h, VR4_BTN_B);
    if (IS("x/click")) return btn(h, VR4_BTN_X);
    if (IS("y/click")) return btn(h, VR4_BTN_Y);
    if (IS("a/touch") || IS("b/touch") || IS("x/touch") || IS("y/touch")) return btn(h, VR4_BTN_THUMB_TOUCH);
    if (IS("menu/click") || IS("system/click") || IS("menu") || IS("system")) return btn(h, VR4_BTN_MENU);
    return 0;
    #undef IS
}
static int is_pose_comp(const char *c) { return !strcmp(c, "grip/pose") || !strcmp(c, "aim/pose") || !strcmp(c, "palm_ext/pose") || !strcmp(c, "grip_surface/pose"); }
static int parse_binding(const char *path, Binding *b) {   // "/user/hand/left/input/trigger/value"
    if (!strncmp(path, "/user/hand/left/", 16)) b->hand = 0;
    else if (!strncmp(path, "/user/hand/right/", 17)) b->hand = 1;
    else return 0;
    const char *c = strstr(path, "/input/");
    if (c) c += 7; else if ((c = strstr(path, "/output/"))) c += 8; else return 0;
    snprintf(b->comp, sizeof b->comp, "%s", c);
    return 1;
}
static int sub_matches(XrPath sub, int hand) {
    if (sub == XR_NULL_PATH) return 1;
    return !strcmp(pstr(sub), hand ? "/user/hand/right" : "/user/hand/left");
}
static XrPosef hand_pose(int hand, const char *comp, int *valid) {
    const VR4Hand *h = &track.hand[hand];
    *valid = (h->flags & VR4_HAND_POSE_VALID) != 0;
    return xp(!strcmp(comp, "aim/pose") ? h->aim : h->grip);
}

// ---------------------------------------------------------------- spaces
typedef enum {
    TRACK_SLOT_HEAD = 0,
    TRACK_SLOT_HAND_LEFT_AIM,
    TRACK_SLOT_HAND_LEFT_GRIP,
    TRACK_SLOT_HAND_RIGHT_AIM,
    TRACK_SLOT_HAND_RIGHT_GRIP,
    TRACK_SLOT_COUNT
} TrackSlot;

typedef struct {
    XrPosef lastPose;
    uint64_t lastTimeNs;
    V linearVel;
    V angularVel;
    int valid;
} VelocityTracker;

static VelocityTracker velTrackers[TRACK_SLOT_COUNT];

static void update_velocities(const VR4Tracking *t) {
    if (!t || t->time_ns == 0) return;
    for (int slot = 0; slot < TRACK_SLOT_COUNT; slot++) {
        VelocityTracker *vt = &velTrackers[slot];
        int slotValid = 0;
        VR4Pose rawPose = {0};
        switch (slot) {
            case TRACK_SLOT_HEAD:
                rawPose = t->head;
                slotValid = 1;
                break;
            case TRACK_SLOT_HAND_LEFT_AIM:
                if (t->hand[0].flags & VR4_HAND_POSE_VALID) { rawPose = t->hand[0].aim; slotValid = 1; }
                break;
            case TRACK_SLOT_HAND_LEFT_GRIP:
                if (t->hand[0].flags & VR4_HAND_POSE_VALID) { rawPose = t->hand[0].grip; slotValid = 1; }
                break;
            case TRACK_SLOT_HAND_RIGHT_AIM:
                if (t->hand[1].flags & VR4_HAND_POSE_VALID) { rawPose = t->hand[1].aim; slotValid = 1; }
                break;
            case TRACK_SLOT_HAND_RIGHT_GRIP:
                if (t->hand[1].flags & VR4_HAND_POSE_VALID) { rawPose = t->hand[1].grip; slotValid = 1; }
                break;
            default: break;
        }
        if (!slotValid) {
            vt->valid = 0;
            vt->lastTimeNs = 0;
            vt->linearVel = (V){0, 0, 0};
            vt->angularVel = (V){0, 0, 0};
            continue;
        }
        XrPosef p = xp(rawPose);
        if (vt->valid && vt->lastTimeNs > 0 && t->time_ns > vt->lastTimeNs) {
            double dt = (double)(t->time_ns - vt->lastTimeNs) * 1e-9;
            if (dt >= 0.002 && dt <= 0.15) {
                V instLinear = {
                    (p.position.x - vt->lastPose.position.x) / (float)dt,
                    (p.position.y - vt->lastPose.position.y) / (float)dt,
                    (p.position.z - vt->lastPose.position.z) / (float)dt
                };
                Q qrel = qmul(p.orientation, qconj(vt->lastPose.orientation));
                if (qrel.w < 0.0f) qrel = (Q){-qrel.x, -qrel.y, -qrel.z, -qrel.w};
                if (qrel.w > 1.0f) qrel.w = 1.0f;
                float angle = 2.0f * acosf(qrel.w);
                float sinHalf = sqrtf(fmaxf(0.0f, 1.0f - qrel.w * qrel.w));
                V instAngular = {0, 0, 0};
                if (sinHalf > 1e-4f && angle > 1e-4f) {
                    float factor = (angle / (float)dt) / sinHalf;
                    instAngular = (V){qrel.x * factor, qrel.y * factor, qrel.z * factor};
                }
                float vMag = sqrtf(instLinear.x * instLinear.x + instLinear.y * instLinear.y + instLinear.z * instLinear.z);
                if (vMag > 40.0f) {
                    float s = 40.0f / vMag;
                    instLinear = (V){instLinear.x * s, instLinear.y * s, instLinear.z * s};
                }
                float wMag = sqrtf(instAngular.x * instAngular.x + instAngular.y * instAngular.y + instAngular.z * instAngular.z);
                if (wMag > 100.0f) {
                    float s = 100.0f / wMag;
                    instAngular = (V){instAngular.x * s, instAngular.y * s, instAngular.z * s};
                }
                if (vt->linearVel.x != 0 || vt->linearVel.y != 0 || vt->linearVel.z != 0) {
                    vt->linearVel.x = 0.75f * instLinear.x + 0.25f * vt->linearVel.x;
                    vt->linearVel.y = 0.75f * instLinear.y + 0.25f * vt->linearVel.y;
                    vt->linearVel.z = 0.75f * instLinear.z + 0.25f * vt->linearVel.z;
                    vt->angularVel.x = 0.75f * instAngular.x + 0.25f * vt->angularVel.x;
                    vt->angularVel.y = 0.75f * instAngular.y + 0.25f * vt->angularVel.y;
                    vt->angularVel.z = 0.75f * instAngular.z + 0.25f * vt->angularVel.z;
                } else {
                    vt->linearVel = instLinear;
                    vt->angularVel = instAngular;
                }
            }
        }
        vt->lastPose = p;
        vt->lastTimeNs = t->time_ns;
        vt->valid = 1;
    }
}

static XrPosef current_local_origin(const Session *s) {
    if (!s) return IDENT;
    float ws = current_world_scale();
    XrPosef o = s->localOrigin;
    o.position = (V){s->localOriginRaw.x / ws, s->localOriginRaw.y / ws, s->localOriginRaw.z / ws};
    return o;
}

static XrPosef space_in_stage(Space *s, int *valid) {
    *valid = 1;
    if (s->ref) switch (s->type) {
        case XR_REFERENCE_SPACE_TYPE_VIEW: return pmul(xp(frameTrack.head), s->offset);
        case XR_REFERENCE_SPACE_TYPE_LOCAL: return pmul(current_local_origin((Session *)s->session), s->offset);
        case XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR: {
            XrPosef o = current_local_origin((Session *)s->session); o.position.y = 0; return pmul(o, s->offset);
        }
        default: return s->offset;
    }
    for (int i = 0; i < s->action->nb; i++) {
        Binding *b = &s->action->b[i];
        if (is_pose_comp(b->comp) && sub_matches(s->sub, b->hand)) return pmul(hand_pose(b->hand, b->comp, valid), s->offset);
    }
    *valid = 0;
    return IDENT;
}

static void space_velocity_in_stage(Space *s, V *outLinear, V *outAngular, int *valid) {
    *valid = 1;
    *outLinear = (V){0, 0, 0};
    *outAngular = (V){0, 0, 0};
    if (!s) { *valid = 0; return; }
    if (s->ref) {
        switch (s->type) {
            case XR_REFERENCE_SPACE_TYPE_STAGE:
            case XR_REFERENCE_SPACE_TYPE_LOCAL:
            case XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR:
                *valid = 1;
                return;
            case XR_REFERENCE_SPACE_TYPE_VIEW: {
                VelocityTracker *vt = &velTrackers[TRACK_SLOT_HEAD];
                *valid = vt->valid;
                if (!vt->valid) return;
                XrPosef headPose = xp(frameTrack.head);
                V r = qrot(headPose.orientation, s->offset.position);
                *outLinear = vadd(vt->linearVel, vcross(vt->angularVel, r));
                *outAngular = vt->angularVel;
                return;
            }
            default:
                *valid = 0;
                return;
        }
    }
    for (int i = 0; i < s->action->nb; i++) {
        Binding *b = &s->action->b[i];
        if (is_pose_comp(b->comp) && sub_matches(s->sub, b->hand)) {
            int isAim = !strcmp(b->comp, "aim/pose");
            TrackSlot slot = (b->hand == 0) ? (isAim ? TRACK_SLOT_HAND_LEFT_AIM : TRACK_SLOT_HAND_LEFT_GRIP)
                                            : (isAim ? TRACK_SLOT_HAND_RIGHT_AIM : TRACK_SLOT_HAND_RIGHT_GRIP);
            VelocityTracker *vt = &velTrackers[slot];
            *valid = vt->valid;
            if (!vt->valid) return;
            int hvalid;
            XrPosef hp = hand_pose(b->hand, b->comp, &hvalid);
            if (!hvalid) { *valid = 0; return; }
            V r = qrot(hp.orientation, s->offset.position);
            *outLinear = vadd(vt->linearVel, vcross(vt->angularVel, r));
            *outAngular = vt->angularVel;
            return;
        }
    }
    *valid = 0;
}

static void set_local_origin(Session *s) {   // LOCAL = head position at start, yaw only
    Q q = {track.head.qx, track.head.qy, track.head.qz, track.head.qw};
    V f = qrot(q, (V){0, 0, -1});
    float yaw = atan2f(-f.x, -f.z);
    s->localOrigin.orientation = (Q){0, sinf(yaw / 2), 0, cosf(yaw / 2)};
    s->localOriginRaw = (V){track.head.px, track.head.py, track.head.pz};
    if (track.time_ns == 0) s->localOriginRaw.y = 1.6f;
    float ws = current_world_scale();
    s->localOrigin.position = (V){s->localOriginRaw.x / ws, s->localOriginRaw.y / ws, s->localOriginRaw.z / ws};
}

// ---------------------------------------------------------------- instance
#define FILL_ARRAY(cap, countOut, arr, n, ...) do { \
    if (countOut) *(countOut) = (n); \
    if ((cap) == 0) return XR_SUCCESS; \
    if ((cap) < (n)) return XR_ERROR_SIZE_INSUFFICIENT; \
    for (uint32_t i_ = 0; i_ < (n); i_++) { __VA_ARGS__; } } while (0)

// Compatibility: games refuse to start or wait forever without some of these (BONELAB blocks on XR_FB_display_refresh_rate).
// Depth / cylinder layers are accepted and ignored; the visibility mask is empty; LOCAL_FLOOR is the local origin on the floor.
static const char *exts[] = {XR_KHR_METAL_ENABLE_EXTENSION_NAME, XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME,
                             XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME, XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME,
                             XR_KHR_COMPOSITION_LAYER_CYLINDER_EXTENSION_NAME, XR_KHR_VISIBILITY_MASK_EXTENSION_NAME,
                             XR_EXT_LOCAL_FLOOR_EXTENSION_NAME, XR_FB_COLOR_SPACE_EXTENSION_NAME,
                             XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME,
                             XR_EXT_DEBUG_UTILS_EXTENSION_NAME};   // no XR_EXT_palm_pose: we only have grip/aim, and a fake palm (= grip) made OpenComposite rotate/shift the hands
static const uint32_t extVer[] = {XR_KHR_metal_enable_SPEC_VERSION, XR_KHR_convert_timespec_time_SPEC_VERSION,
                                  XR_FB_display_refresh_rate_SPEC_VERSION, XR_KHR_composition_layer_depth_SPEC_VERSION,
                                  XR_KHR_composition_layer_cylinder_SPEC_VERSION, XR_KHR_visibility_mask_SPEC_VERSION,
                                  XR_EXT_local_floor_SPEC_VERSION, XR_FB_color_space_SPEC_VERSION,
                                  XR_KHR_composition_layer_color_scale_bias_SPEC_VERSION,
                                  XR_EXT_debug_utils_SPEC_VERSION};
#define NEXTS (sizeof exts / sizeof *exts)

static XrResult XRAPI_CALL xrEnumerateApiLayerProperties_(uint32_t cap, uint32_t *n, XrApiLayerProperties *p) { (void)cap; (void)p; *n = 0; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrEnumerateInstanceExtensionProperties_(const char *layer, uint32_t cap, uint32_t *n, XrExtensionProperties *p) {
    (void)layer;
    FILL_ARRAY(cap, n, p, (uint32_t)NEXTS, { strcpy(p[i_].extensionName, exts[i_]); p[i_].extensionVersion = extVer[i_]; });
    return XR_SUCCESS;
}

static XrResult XRAPI_CALL xrCreateInstance_(const XrInstanceCreateInfo *ci, XrInstance *out) {
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        int ok = 0;
        for (size_t j = 0; j < NEXTS; j++) ok |= !strcmp(ci->enabledExtensionNames[i], exts[j]);
        if (!ok) { logmsg("unsupported extension %s", ci->enabledExtensionNames[i]); return XR_ERROR_EXTENSION_NOT_PRESENT; }
    }
    if (!shm) {
        const char *path = getenv("VR4MAC_SHM") ? getenv("VR4MAC_SHM") : VR4_SHM_PATH_MAC;   // override for tests
        int fd = open(path, O_RDWR);
        if (fd < 0) { logmsg("MacVR app not running (no %s)", path); return XR_ERROR_RUNTIME_UNAVAILABLE; }
        void *m = mmap(NULL, VR4_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        shm = m == MAP_FAILED ? NULL : (VR4Shm *)m;
        if (!shm || shm->magic != VR4_SHM_MAGIC || shm->version != VR4_SHM_VERSION) {
            logmsg("bad shared memory");
            if (shm) munmap(shm, VR4_SHM_SIZE);
            shm = NULL; return XR_ERROR_RUNTIME_UNAVAILABLE;
        }
    }
    snprintf(appName, sizeof appName, "%s", ci->applicationInfo.applicationName);
    snprintf(shm->app_name, sizeof shm->app_name, "%s", appName);
    logmsg("xrCreateInstance app=%s engine=%s", ci->applicationInfo.applicationName, ci->applicationInfo.engineName);
    *out = (XrInstance)(uintptr_t)1;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroyInstance_(XrInstance i) { (void)i; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrGetInstanceProperties_(XrInstance i, XrInstanceProperties *p) {
    (void)i; p->runtimeVersion = XR_MAKE_VERSION(0, 2, 0); strcpy(p->runtimeName, "SiliconXR"); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrPollEvent_(XrInstance i, XrEventDataBuffer *e) {
    (void)i;
    Session *s = theSession;
    if (s && s->running && !s->exitRequested) {   // dashboard open on the Mac side = app loses input focus
        int focused = !shm->input_blocked;
        if (focused != s->focused) { s->focused = focused; push_state(focused ? XR_SESSION_STATE_FOCUSED : XR_SESSION_STATE_VISIBLE); }
    }
    if (evHead == evTail) return XR_EVENT_UNAVAILABLE;
    memcpy(e, &events[evHead++ % 64], sizeof *e);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrResultToString_(XrInstance i, XrResult r, char buf[XR_MAX_RESULT_STRING_SIZE]) {
    (void)i;
    #define RS(name, val) case name: snprintf(buf, XR_MAX_RESULT_STRING_SIZE, "%s", #name); break;
    switch (r) { XR_LIST_ENUM_XrResult(RS) default: snprintf(buf, XR_MAX_RESULT_STRING_SIZE, "XR_UNKNOWN_%d", r); }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrStructureTypeToString_(XrInstance i, XrStructureType t, char buf[XR_MAX_STRUCTURE_NAME_SIZE]) {
    (void)i;
    #undef RS
    #define RS(name, val) case name: snprintf(buf, XR_MAX_STRUCTURE_NAME_SIZE, "%s", #name); break;
    switch (t) { XR_LIST_ENUM_XrStructureType(RS) default: snprintf(buf, XR_MAX_STRUCTURE_NAME_SIZE, "XR_UNKNOWN_STRUCTURE_TYPE_%d", t); }
    #undef RS
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrStringToPath_(XrInstance i, const char *s, XrPath *p) { (void)i; *p = intern(s); return *p ? XR_SUCCESS : XR_ERROR_PATH_COUNT_EXCEEDED; }
static XrResult XRAPI_CALL xrPathToString_(XrInstance i, XrPath p, uint32_t cap, uint32_t *n, char *buf) {
    (void)i;
    if (p < 1 || p > (XrPath)npaths) return XR_ERROR_PATH_INVALID;
    const char *s = pstr(p); uint32_t len = (uint32_t)strlen(s) + 1;
    *n = len;
    if (!cap) return XR_SUCCESS;
    if (cap < len) return XR_ERROR_SIZE_INSUFFICIENT;
    memcpy(buf, s, len);
    return XR_SUCCESS;
}

// ---------------------------------------------------------------- system
static XrResult XRAPI_CALL xrGetSystem_(XrInstance i, const XrSystemGetInfo *gi, XrSystemId *id) {
    (void)i;
    if (gi->formFactor != XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY) return XR_ERROR_FORM_FACTOR_UNSUPPORTED;
    *id = 1; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetSystemProperties_(XrInstance i, XrSystemId id, XrSystemProperties *p) {
    (void)i; (void)id;
    p->systemId = 1; p->vendorId = 0x2833;
    strcpy(p->systemName, "MacVR Quest");
    p->graphicsProperties.maxSwapchainImageWidth = 4096; p->graphicsProperties.maxSwapchainImageHeight = 4096;
    p->graphicsProperties.maxLayerCount = XR_MIN_COMPOSITION_LAYERS_SUPPORTED;
    p->trackingProperties.orientationTracking = XR_TRUE; p->trackingProperties.positionTracking = XR_TRUE;
    for (XrBaseOutStructure *next = (XrBaseOutStructure *)p->next; next; next = next->next) {
        if (next->type == XR_TYPE_SYSTEM_COLOR_SPACE_PROPERTIES_FB) {
            ((XrSystemColorSpacePropertiesFB *)next)->colorSpace = XR_COLOR_SPACE_QUEST_FB;
        }
    }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEnumerateEnvironmentBlendModes_(XrInstance i, XrSystemId id, XrViewConfigurationType v, uint32_t cap, uint32_t *n, XrEnvironmentBlendMode *m) {
    (void)i; (void)id; (void)v; FILL_ARRAY(cap, n, m, 1, m[i_] = XR_ENVIRONMENT_BLEND_MODE_OPAQUE); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEnumerateViewConfigurations_(XrInstance i, XrSystemId id, uint32_t cap, uint32_t *n, XrViewConfigurationType *t) {
    (void)i; (void)id; FILL_ARRAY(cap, n, t, 1, t[i_] = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetViewConfigurationProperties_(XrInstance i, XrSystemId id, XrViewConfigurationType t, XrViewConfigurationProperties *p) {
    (void)i; (void)id;
    if (t != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO) return XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED;
    p->viewConfigurationType = t; p->fovMutable = XR_FALSE; return XR_SUCCESS;
}
static uint32_t eye_w(void) {
    uint32_t base = (shm && shm->eye_w) ? shm->eye_w : 1440;
    float s = current_render_scale();
    uint32_t w = (uint32_t)((float)base * s);
    uint32_t rounded = ((w + 16) / 32) * 32;
    return rounded < 128 ? 128 : (rounded > 4096 ? 4096 : rounded);
}
static uint32_t eye_h(void) {
    uint32_t base = (shm && shm->eye_h) ? shm->eye_h : 1584;
    float s = current_render_scale();
    uint32_t h = (uint32_t)((float)base * s);
    uint32_t rounded = ((h + 16) / 32) * 32;
    return rounded < 128 ? 128 : (rounded > 4096 ? 4096 : rounded);
}
static XrResult XRAPI_CALL xrEnumerateViewConfigurationViews_(XrInstance i, XrSystemId id, XrViewConfigurationType t, uint32_t cap, uint32_t *n, XrViewConfigurationView *v) {
    (void)i; (void)id;
    if (t != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO) return XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED;
    FILL_ARRAY(cap, n, v, 2, {
        v[i_].recommendedImageRectWidth = eye_w(); v[i_].recommendedImageRectHeight = eye_h();
        v[i_].maxImageRectWidth = 4096; v[i_].maxImageRectHeight = 4096;
        v[i_].recommendedSwapchainSampleCount = 1; v[i_].maxSwapchainSampleCount = 1; });
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetMetalGraphicsRequirementsKHR_(XrInstance i, XrSystemId id, XrGraphicsRequirementsMetalKHR *r) {
    (void)i; (void)id;
    r->metalDevice = MTLCreateSystemDefaultDevice();   // +1, owned by the caller per the extension spec
    return r->metalDevice ? XR_SUCCESS : XR_ERROR_RUNTIME_FAILURE;
}
static XrResult XRAPI_CALL xrConvertTimespecTimeToTimeKHR_(XrInstance i, const struct timespec *ts, XrTime *t) {
    (void)i; *t = (XrTime)((int64_t)ts->tv_sec * 1000000000LL + ts->tv_nsec) + qpcOffsetNs; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrConvertTimeToTimespecTimeKHR_(XrInstance i, XrTime t, struct timespec *ts) {
    (void)i; int64_t ns = (int64_t)t - qpcOffsetNs; ts->tv_sec = ns / 1000000000LL; ts->tv_nsec = ns % 1000000000LL; return XR_SUCCESS;
}

// ---------------------------------------------------------------- session
static XrResult XRAPI_CALL xrCreateSession_(XrInstance i, const XrSessionCreateInfo *ci, XrSession *out) {
    (void)i;
    const XrGraphicsBindingMetalKHR *gb = NULL;
    for (const XrBaseInStructure *b = ci->next; b; b = b->next)
        if (b->type == XR_TYPE_GRAPHICS_BINDING_METAL_KHR) gb = (const XrGraphicsBindingMetalKHR *)b;
    if (!gb || !gb->commandQueue) { logmsg("xrCreateSession: only Metal is supported"); return XR_ERROR_GRAPHICS_DEVICE_INVALID; }
    Session *s = calloc(1, sizeof *s);
    s->queue = [(id<MTLCommandQueue>)gb->commandQueue retain];
    read_tracking(); frameTrack = track; set_local_origin(s);
    theSession = s;
    *out = (XrSession)s;
    push_state(XR_SESSION_STATE_IDLE); push_state(XR_SESSION_STATE_READY);
    logmsg("xrCreateSession ok, eye %ux%u", eye_w(), eye_h());
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroySession_(XrSession h) {
    Session *s = (Session *)h;
    id<MTLCommandBuffer> drain = [s->queue commandBuffer]; [drain commit]; [drain waitUntilCompleted];   // queue is in order
    dispatch_sync(publish_queue(), ^{});                                                                  // and pending publishes
    for (int k = 0; k < NRING; k++) [s->ring[k] release];
    [s->queue release];
    if (theSession == s) theSession = NULL;
    free(s);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrBeginSession_(XrSession h, const XrSessionBeginInfo *bi) {
    Session *s = (Session *)h; (void)bi;
    s->running = 1; s->focused = 1;
    push_state(XR_SESSION_STATE_SYNCHRONIZED); push_state(XR_SESSION_STATE_VISIBLE); push_state(XR_SESSION_STATE_FOCUSED);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEndSession_(XrSession h) {
    Session *s = (Session *)h;
    s->running = 0;
    push_state(XR_SESSION_STATE_IDLE); push_state(XR_SESSION_STATE_EXITING);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrRequestExitSession_(XrSession h) {
    Session *s = (Session *)h;
    s->exitRequested = 1;
    push_state(XR_SESSION_STATE_VISIBLE); push_state(XR_SESSION_STATE_SYNCHRONIZED); push_state(XR_SESSION_STATE_STOPPING);
    return XR_SUCCESS;
}

// ---------------------------------------------------------------- spaces
static XrResult XRAPI_CALL xrEnumerateReferenceSpaces_(XrSession h, uint32_t cap, uint32_t *n, XrReferenceSpaceType *t) {
    (void)h;
    static const XrReferenceSpaceType all[] = {XR_REFERENCE_SPACE_TYPE_VIEW, XR_REFERENCE_SPACE_TYPE_LOCAL,
                                               XR_REFERENCE_SPACE_TYPE_STAGE, XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR};
    FILL_ARRAY(cap, n, t, 4, t[i_] = all[i_]);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrCreateReferenceSpace_(XrSession h, const XrReferenceSpaceCreateInfo *ci, XrSpace *out) {
    if (ci->referenceSpaceType != XR_REFERENCE_SPACE_TYPE_VIEW && ci->referenceSpaceType != XR_REFERENCE_SPACE_TYPE_LOCAL &&
        ci->referenceSpaceType != XR_REFERENCE_SPACE_TYPE_STAGE && ci->referenceSpaceType != XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR) {
        logmsg("reference space %d unsupported", ci->referenceSpaceType); return XR_ERROR_REFERENCE_SPACE_UNSUPPORTED;
    }
    Space *s = calloc(1, sizeof *s);
    s->session = h; s->ref = 1; s->type = ci->referenceSpaceType; s->offset = ci->poseInReferenceSpace;
    *out = (XrSpace)s; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetReferenceSpaceBoundsRect_(XrSession h, XrReferenceSpaceType t, XrExtent2Df *b) {
    (void)h;
    if (t != XR_REFERENCE_SPACE_TYPE_STAGE) { b->width = b->height = 0; return XR_SPACE_BOUNDS_UNAVAILABLE; }
    float ws = current_world_scale();
    b->width = 2.0f / ws; b->height = 2.0f / ws; return XR_SUCCESS;   // fixed 2x2 m play area scaled to stage
}
static XrResult XRAPI_CALL xrCreateActionSpace_(XrSession h, const XrActionSpaceCreateInfo *ci, XrSpace *out) {
    Space *s = calloc(1, sizeof *s);
    s->session = h; s->action = (Action *)ci->action; s->sub = ci->subactionPath; s->offset = ci->poseInActionSpace;
    *out = (XrSpace)s; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrLocateSpace_(XrSpace sp, XrSpace base, XrTime t, XrSpaceLocation *loc) {
    (void)t;
    int v1, v2;
    XrPosef a = space_in_stage((Space *)sp, &v1), b = space_in_stage((Space *)base, &v2);
    loc->pose = pmul(pinv(b), a);
    loc->locationFlags = v1 && v2 ? XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT |
                                    XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT : 0;
    for (XrBaseOutStructure *n = loc->next; n; n = n->next)
        if (n->type == XR_TYPE_SPACE_VELOCITY) {
            XrSpaceVelocity *sv = (XrSpaceVelocity *)n;
            V va, wa, vb, wb;
            int vvel1 = 0, vvel2 = 0;
            space_velocity_in_stage((Space *)sp, &va, &wa, &vvel1);
            space_velocity_in_stage((Space *)base, &vb, &wb, &vvel2);
            if (v1 && v2 && vvel1 && vvel2) {
                V relLin = vsub(va, vb);
                V relAng = vsub(wa, wb);
                sv->linearVelocity = qrot(qconj(b.orientation), relLin);
                sv->angularVelocity = qrot(qconj(b.orientation), relAng);
                sv->velocityFlags = XR_SPACE_VELOCITY_LINEAR_VALID_BIT | XR_SPACE_VELOCITY_ANGULAR_VALID_BIT;
            } else {
                sv->velocityFlags = 0;
                sv->linearVelocity = (XrVector3f){0, 0, 0};
                sv->angularVelocity = (XrVector3f){0, 0, 0};
            }
        }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroySpace_(XrSpace s) { free(s); return XR_SUCCESS; }

// ---------------------------------------------------------------- frames
static int64_t endSum, copySum;   // pacing log: time inside xrEndFrame / in the CPU readback copy
static XrResult XRAPI_CALL xrWaitFrame_(XrSession h, const XrFrameWaitInfo *wi, XrFrameState *fs) {
    (void)wi; (void)h;
    float fps = shm->fps > 0 ? shm->fps : 72;
    int64_t period = (int64_t)(1e9 / fps), start = qpc_ns();
    static int64_t lastExit, statStart, waitSum, workSum; static int statN;
    // Pace the game to the headset: block until the next tracking sample, but never past one period (+2 ms jitter)
    // after the previous frame was released. The deadline used to be 2 periods from *now*, so stale tracking (headset
    // asleep while linked) stacked the full timeout on top of the game's own frame time: 6 ms game -> 29 fps.
    int64_t deadline = (lastExit ? lastExit : start) + period + 2000000;
    while (!read_tracking()) {
        if (qpc_ns() > deadline) break;
        usleep(100);
    }
    frameTrack = track;
    frameRing[frameRingN++ % 4] = track;
    int64_t now = qpc_ns();
    if (shm->client_connected && track.time_ns) qpcOffsetNs = (int64_t)track.time_ns - now - period;
    // Connected: the exact tracking sample time, since the Quest matches video to its pose history by it. Monotonic
    // synthesis only while disconnected; never latch across timelines (045fd16 regression: navy screen).
    static int64_t lastPredicted = 0;
    int64_t predicted;
    if (shm->client_connected && track.time_ns) predicted = (int64_t)track.time_ns;
    else { predicted = now + qpcOffsetNs + period; if (predicted <= lastPredicted) predicted = lastPredicted + period; }
    lastPredicted = predicted;
    fs->predictedDisplayTime = (XrTime)predicted;
    fs->predictedDisplayPeriod = period;
    fs->shouldRender = XR_TRUE;
    // pacing diagnostics every 5 s: time blocked here vs time the game spends between frames (render+submit+present)
    int64_t exitNs = qpc_ns();
    if (lastExit) { waitSum += exitNs - start; workSum += start - lastExit; statN++; }
    lastExit = exitNs;
    if (!statStart) statStart = exitNs;
    if (exitNs - statStart > 5000000000LL && statN) {
        logmsg("pacing: %.1f fps, wait %.2f ms, game %.2f ms/frame (endframe %.2f, copy %.2f), connected %u", statN * 1e9 / (exitNs - statStart),
               waitSum / 1e6 / statN, workSum / 1e6 / statN, endSum / 1e6 / statN, copySum / 1e6 / statN, shm->client_connected);
        statStart = exitNs; waitSum = workSum = endSum = copySum = 0; statN = 0;
    }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrBeginFrame_(XrSession h, const XrFrameBeginInfo *bi) { (void)h; (void)bi; return XR_SUCCESS; }

static XrResult XRAPI_CALL xrLocateViews_(XrSession h, const XrViewLocateInfo *li, XrViewState *vs, uint32_t cap, uint32_t *n, XrView *views) {
    (void)h;
    int valid;
    XrPosef base = space_in_stage((Space *)li->space, &valid);
    const VR4Tracking *ft = frame_for(li->displayTime);
    vs->viewStateFlags = valid ? (XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT |
                                  XR_VIEW_STATE_ORIENTATION_TRACKED_BIT | XR_VIEW_STATE_POSITION_TRACKED_BIT) : 0;
    FILL_ARRAY(cap, n, views, 2, {
        VR4Eye d = {{i_ ? 0.032f : -0.032f, 1.6f, 0, 0, 0, 0, 1}, {-0.8f, 0.8f, 0.8f, -0.8f}};   // no headset yet
        const VR4Eye *e = ft->time_ns ? &ft->eye[i_] : &d;
        views[i_].pose = pmul(pinv(base), xp(e->pose));
        views[i_].fov = (XrFovf){e->fov.left, e->fov.right, e->fov.up, e->fov.down};
    });
    return XR_SUCCESS;
}

static Swapchain *find_sc(XrSwapchain s) { return (Swapchain *)s; }
static int is_rgba(MTLPixelFormat f) { return f == MTLPixelFormatRGBA8Unorm || f == MTLPixelFormatRGBA8Unorm_sRGB; }

// ---------------------------------------------------------------- Metal readback
// Eye images are blitted into a shared ring buffer on the app's own queue (after its rendering, queues run in order),
// then copied into the shm frame on a serial queue when the GPU is done. The game thread never waits.
typedef struct { VR4Pose p[2]; } Pose2;
static dispatch_queue_t publish_queue(void) {
    static dispatch_queue_t q; static dispatch_once_t once;
    dispatch_once(&once, ^{ q = dispatch_queue_create("SiliconXR.publish", DISPATCH_QUEUE_SERIAL); });
    return q;
}
static int ring_slot(Session *s, size_t size) {
    if (s->ringSize != size) {   // (re)allocate once every buffer is idle
        for (int k = 0; k < NRING; k++) if (__atomic_load_n(&s->busy[k], __ATOMIC_ACQUIRE)) return -1;
        for (int k = 0; k < NRING; k++) {
            [s->ring[k] release];
            s->ring[k] = [s->queue.device newBufferWithLength:size options:MTLResourceStorageModeShared];
            if (!s->ring[k]) { s->ringSize = 0; return -1; }
        }
        s->ringSize = size;
    }
    for (int k = 0; k < NRING; k++) if (!__atomic_load_n(&s->busy[k], __ATOMIC_ACQUIRE)) return k;
    return -1;   // GPU still owns every buffer: drop this frame rather than block
}
/// Copies `n` w x h regions into one ring buffer (`row` bytes per row, region e at byte e*w*4), then runs `done` with
/// the bytes on the publish queue. Returns 0 if no buffer was free.
static int readback(Session *s, int n, id<MTLTexture> const *tex, const uint32_t *slice, const MTLOrigin *org, uint32_t w, uint32_t h,
                    size_t row, void (^done)(const uint8_t *)) {
    int k = ring_slot(s, row * h);
    if (k < 0) return 0;
    id<MTLCommandBuffer> cb = [s->queue commandBuffer];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    for (int e = 0; e < n; e++)
        [bl copyFromTexture:tex[e] sourceSlice:slice[e] sourceLevel:0 sourceOrigin:org[e] sourceSize:MTLSizeMake(w, h, 1)
                   toBuffer:s->ring[k] destinationOffset:(NSUInteger)e * w * 4 destinationBytesPerRow:row destinationBytesPerImage:row * h];
    [bl endEncoding];
    __atomic_store_n(&s->busy[k], 1, __ATOMIC_RELEASE);
    id<MTLBuffer> buf = s->ring[k]; volatile int *busy = &s->busy[k];
    void (^cp)(const uint8_t *) = [done copy];
    [cb addCompletedHandler:^(id<MTLCommandBuffer> c) {
        dispatch_async(publish_queue(), ^{
            if (c.status == MTLCommandBufferStatusCompleted) cp((const uint8_t *)buf.contents);
            __atomic_store_n(busy, 0, __ATOMIC_RELEASE);
            [cp release];
        });
    }];
    [cb commit];
    return 1;
}
static void publish(uint32_t buf, uint32_t fw, uint32_t fh, int rgba, uint64_t t, Pose2 pose) {
    shm->frame_w[buf] = fw; shm->frame_h[buf] = fh; shm->frame_rgba[buf] = (uint32_t)rgba;
    shm->frame_time_ns[buf] = t;
    shm->frame_eye_pose[buf][0] = pose.p[0]; shm->frame_eye_pose[buf][1] = pose.p[1];
    shm->runtime_heartbeat_ns = (uint64_t)qpc_ns();
    vr4_fence();
    shm->frame_seq++;
}

static XrResult XRAPI_CALL xrEndFrameImpl(XrSession h, const XrFrameEndInfo *fi);
static XrResult XRAPI_CALL xrEndFrame_(XrSession h, const XrFrameEndInfo *fi) {
    int64_t endT0 = qpc_ns();
    XrResult r = xrEndFrameImpl(h, fi);
    endSum += qpc_ns() - endT0;
    return r;
}
// Fallback for apps that submit quad layers without a stereo projection (menus, media players):
// letterbox the first quad into both eyes so the headset shows something instead of a stale frame.
static XrResult XRAPI_CALL endFrameQuad(Session *s, const XrFrameEndInfo *fi) {
    const XrCompositionLayerQuad *q = NULL;
    for (uint32_t i = 0; i < fi->layerCount; i++)
        if (fi->layers[i] && fi->layers[i]->type == XR_TYPE_COMPOSITION_LAYER_QUAD) { q = (const XrCompositionLayerQuad *)fi->layers[i]; break; }
    if (!q) {
        static int warned_eq;
        for (uint32_t i = 0; i < fi->layerCount; i++)
            if (fi->layers[i] && fi->layers[i]->type == XR_TYPE_COMPOSITION_LAYER_EQUIRECT_KHR && !warned_eq++)
                logmsg("equirect layers are not composited yet");
        return XR_SUCCESS;
    }
    Swapchain *sc = find_sc(q->subImage.swapchain);
    if (!sc || sc->released < 0) return XR_SUCCESS;
    uint32_t sw = (uint32_t)q->subImage.imageRect.extent.width, sh = (uint32_t)q->subImage.imageRect.extent.height;
    int32_t ox = q->subImage.imageRect.offset.x, oy = q->subImage.imageRect.offset.y;
    if (!sw || !sh || ox < 0 || oy < 0 || (uint32_t)ox + sw > sc->w || (uint32_t)oy + sh > sc->h ||
        q->subImage.imageArrayIndex >= sc->array) return XR_SUCCESS;
    id<MTLTexture> tex[1] = {sc->img[sc->released]};
    uint32_t slice[1] = {q->subImage.imageArrayIndex};
    MTLOrigin org[1] = {MTLOriginMake((NSUInteger)ox, (NSUInteger)oy, 0)};
    int rgba = is_rgba(sc->fmt);
    Pose2 pose = {{frameTrack.head, frameTrack.head}};
    uint64_t t = (uint64_t)fi->displayTime;
    readback(s, 1, tex, slice, org, sw, sh, (size_t)sw * 4, ^(const uint8_t *src) {
        uint32_t dw = eye_w(), dh = eye_h();
        if (2ULL * dw * dh * 4ULL > (uint64_t)VR4_FRAME_MAX) return;
        uint32_t buf = (shm->frame_seq + 1) % 2;
        uint8_t *dst = vr4_frame(shm, buf);
        float scf = (float)dw / sw < (float)dh / sh ? (float)dw / sw : (float)dh / sh;
        uint32_t tw = (uint32_t)(sw * scf), th = (uint32_t)(sh * scf);
        if (!tw) tw = 1;
        if (!th) th = 1;
        if (tw > dw) tw = dw;
        if (th > dh) th = dh;
        uint32_t x0 = (dw - tw) / 2, y0 = (dh - th) / 2;
        memset(dst, 0, (size_t)2 * dw * dh * 4);
        for (uint32_t y = 0; y < th; y++) {
            const uint32_t *row = (const uint32_t *)(src + (size_t)(y * sh / th) * sw * 4);
            for (int e = 0; e < 2; e++) {
                uint32_t *out = (uint32_t *)(dst + ((size_t)(y0 + y) * 2 * dw + (size_t)e * dw + x0) * 4);
                for (uint32_t x = 0; x < tw; x++) out[x] = row[x * sw / tw];
            }
        }
        publish(buf, 2 * dw, dh, rgba, t, pose);
    });
    s->lastPublished = t;
    shm->runtime_heartbeat_ns = (uint64_t)qpc_ns();
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEndFrameImpl(XrSession h, const XrFrameEndInfo *fi) {
    Session *s = (Session *)h;
    if (!s || !s->queue || !fi) return XR_ERROR_HANDLE_INVALID;
    if (fi->displayTime && (uint64_t)fi->displayTime == s->lastPublished) {
        shm->runtime_heartbeat_ns = (uint64_t)qpc_ns();   // same frame re-submitted: no new pixels, stay alive
        return XR_SUCCESS;
    }
    const XrCompositionLayerProjection *proj = NULL;
    for (uint32_t i = 0; i < fi->layerCount; i++)
        if (fi->layers[i] && fi->layers[i]->type == XR_TYPE_COMPOSITION_LAYER_PROJECTION) { proj = (const XrCompositionLayerProjection *)fi->layers[i]; break; }
    static int layersLogged;
    if (layersLogged < 3 && fi->layerCount) {
        layersLogged++;
        for (uint32_t i = 0; i < fi->layerCount; i++) if (fi->layers[i]) logmsg("endframe layer %u type %d flags 0x%llx", i, fi->layers[i]->type, (unsigned long long)fi->layers[i]->layerFlags);
        if (proj) for (int e = 0; e < 2 && e < (int)proj->viewCount; e++) {
            const XrSwapchainSubImage *si = &proj->views[e].subImage;
            logmsg("  view %d sc %p rect %d,%d %dx%d array %u", e, (void *)si->swapchain, si->imageRect.offset.x, si->imageRect.offset.y,
                   si->imageRect.extent.width, si->imageRect.extent.height, si->imageArrayIndex);
        }
    }
    if (!proj || proj->viewCount < 2) return endFrameQuad(s, fi);   // no stereo projection: quad fallback or nothing

    Swapchain *sc[2] = { find_sc(proj->views[0].subImage.swapchain), find_sc(proj->views[1].subImage.swapchain) };
    if (!sc[0] || !sc[1]) {
        static int warned_sc; if (!warned_sc++) logmsg("xrEndFrame: invalid swapchain handle");
        return XR_ERROR_HANDLE_INVALID;
    }
    if (sc[0]->released < 0 || sc[1]->released < 0) return XR_SUCCESS;

    uint64_t w = (uint64_t)proj->views[0].subImage.imageRect.extent.width, hgt = (uint64_t)proj->views[0].subImage.imageRect.extent.height;
    if (!w || !hgt || (uint64_t)proj->views[1].subImage.imageRect.extent.width != w || (uint64_t)proj->views[1].subImage.imageRect.extent.height != hgt ||
        2ULL * w * hgt * 4ULL > (uint64_t)VR4_FRAME_MAX) {
        static int warned; if (!warned++) logmsg("unsupported eye rects %llux%llu / %dx%d", (unsigned long long)w, (unsigned long long)hgt, proj->views[1].subImage.imageRect.extent.width, proj->views[1].subImage.imageRect.extent.height);
        return XR_SUCCESS;
    }

    MTLPixelFormat fmt = sc[0]->fmt;
    if (sc[1]->fmt != fmt) {
        static int warned_fmt; if (!warned_fmt++) logmsg("mismatched swapchain formats between eyes: %d vs %d", sc[0]->fmt, sc[1]->fmt);
        return XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED;
    }

    // Bounds checking for rects and array indices against swapchain metadata
    for (int e = 0; e < 2; e++) {
        const XrSwapchainSubImage *si = &proj->views[e].subImage;
        if (si->imageArrayIndex >= sc[e]->array) {
            static int warned_arr; if (!warned_arr++) logmsg("eye %d imageArrayIndex %u >= %u", e, si->imageArrayIndex, sc[e]->array);
            return XR_ERROR_RUNTIME_FAILURE;
        }
        if (si->imageRect.offset.x < 0 || si->imageRect.offset.y < 0 ||
            (uint64_t)si->imageRect.offset.x + w > (uint64_t)sc[e]->w ||
            (uint64_t)si->imageRect.offset.y + hgt > (uint64_t)sc[e]->h) {
            static int warned_rect; if (!warned_rect++) logmsg("eye %d imageRect bounds overflow", e);
            return XR_ERROR_RUNTIME_FAILURE;
        }
    }

    id<MTLTexture> tex[2]; uint32_t slice[2]; MTLOrigin org[2]; Pose2 pose;
    for (int e = 0; e < 2; e++) {
        const XrSwapchainSubImage *si = &proj->views[e].subImage;
        tex[e] = sc[e]->img[sc[e]->released]; slice[e] = si->imageArrayIndex;
        org[e] = MTLOriginMake((NSUInteger)si->imageRect.offset.x, (NSUInteger)si->imageRect.offset.y, 0);
        pose.p[e] = (VR4Pose){proj->views[e].pose.position.x, proj->views[e].pose.position.y, proj->views[e].pose.position.z,
                              proj->views[e].pose.orientation.x, proj->views[e].pose.orientation.y, proj->views[e].pose.orientation.z, proj->views[e].pose.orientation.w};
    }
    int rgba = is_rgba(fmt);
    uint64_t t = (uint64_t)fi->displayTime;
    uint32_t fw = 2 * (uint32_t)w, fh = (uint32_t)hgt;
    int64_t copyT0 = qpc_ns();
    readback(s, 2, tex, slice, org, (uint32_t)w, (uint32_t)hgt, (size_t)fw * 4, ^(const uint8_t *src) {
        uint32_t buf = (shm->frame_seq + 1) % 2;
        memcpy(vr4_frame(shm, buf), src, (size_t)fw * fh * 4);   // channel order fixed up on the Mac (frame_rgba)
        publish(buf, fw, fh, rgba, t, pose);
    });
    copySum += qpc_ns() - copyT0;
    s->lastPublished = t;
    shm->runtime_heartbeat_ns = (uint64_t)qpc_ns();
    return XR_SUCCESS;
}

// ---------------------------------------------------------------- swapchains
static const int64_t formats[] = {MTLPixelFormatBGRA8Unorm_sRGB, MTLPixelFormatRGBA8Unorm_sRGB, MTLPixelFormatBGRA8Unorm, MTLPixelFormatRGBA8Unorm,
                                  MTLPixelFormatDepth32Float, MTLPixelFormatDepth16Unorm, MTLPixelFormatDepth32Float_Stencil8};
#define NFORMATS (sizeof formats / sizeof *formats)
static XrResult XRAPI_CALL xrEnumerateSwapchainFormats_(XrSession h, uint32_t cap, uint32_t *n, int64_t *f) {
    (void)h; FILL_ARRAY(cap, n, f, (uint32_t)NFORMATS, f[i_] = formats[i_]); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrCreateSwapchain_(XrSession h, const XrSwapchainCreateInfo *ci, XrSwapchain *out) {
    Session *s = (Session *)h;
    int ok = 0;
    for (size_t i = 0; i < NFORMATS; i++) ok |= formats[i] == ci->format;
    if (!ok) { logmsg("swapchain format %lld unsupported", (long long)ci->format); return XR_ERROR_SWAPCHAIN_FORMAT_UNSUPPORTED; }
    if (ci->sampleCount > 1) { logmsg("swapchain asks %u samples: unsupported (max 1)", ci->sampleCount); return XR_ERROR_FEATURE_UNSUPPORTED; }
    logmsg("swapchain fmt %lld %ux%u array %u mips %u faces %u usage 0x%llx flags 0x%llx", (long long)ci->format, ci->width, ci->height,
           ci->arraySize, ci->mipCount, ci->faceCount, (unsigned long long)ci->usageFlags, (unsigned long long)ci->createFlags);
    Swapchain *sc = calloc(1, sizeof *sc);
    sc->fmt = (MTLPixelFormat)ci->format; sc->w = ci->width; sc->h = ci->height; sc->array = ci->arraySize ? ci->arraySize : 1;
    sc->mips = ci->mipCount ? ci->mipCount : 1; sc->released = -1;
    sc->count = ci->createFlags & XR_SWAPCHAIN_CREATE_STATIC_IMAGE_BIT ? 1 : 3;
    MTLTextureDescriptor *d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:sc->fmt width:ci->width height:ci->height mipmapped:NO];
    d.textureType = sc->array > 1 ? MTLTextureType2DArray : MTLTextureType2D;
    d.arrayLength = sc->array; d.mipmapLevelCount = sc->mips; d.storageMode = MTLStorageModePrivate;
    d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
    if (ci->usageFlags & XR_SWAPCHAIN_USAGE_UNORDERED_ACCESS_BIT) d.usage |= MTLTextureUsageShaderWrite;
    if (ci->usageFlags & XR_SWAPCHAIN_USAGE_MUTABLE_FORMAT_BIT) d.usage |= MTLTextureUsagePixelFormatView;
    for (int i = 0; i < sc->count; i++)
        if (!(sc->img[i] = [s->queue.device newTextureWithDescriptor:d])) {
            logmsg("newTextureWithDescriptor failed");
            for (int k = 0; k < i; k++) [sc->img[k] release];
            free(sc); return XR_ERROR_RUNTIME_FAILURE;
        }
    *out = (XrSwapchain)sc;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroySwapchain_(XrSwapchain h) {
    Swapchain *sc = find_sc(h);
    for (int i = 0; i < sc->count; i++) [sc->img[i] release];   // a readback in flight keeps its own reference
    free(sc); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEnumerateSwapchainImages_(XrSwapchain h, uint32_t cap, uint32_t *n, XrSwapchainImageBaseHeader *imgs) {
    Swapchain *sc = find_sc(h);
    XrSwapchainImageMetalKHR *d = (XrSwapchainImageMetalKHR *)imgs;
    FILL_ARRAY(cap, n, d, (uint32_t)sc->count, d[i_].texture = sc->img[i_]);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrAcquireSwapchainImage_(XrSwapchain h, const XrSwapchainImageAcquireInfo *ai, uint32_t *idx) {
    (void)ai; Swapchain *sc = find_sc(h);
    sc->acquired = (sc->acquired + 1) % sc->count; *idx = sc->acquired; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrWaitSwapchainImage_(XrSwapchain h, const XrSwapchainImageWaitInfo *wi) { (void)h; (void)wi; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrReleaseSwapchainImage_(XrSwapchain h, const XrSwapchainImageReleaseInfo *ri) {
    (void)ri; Swapchain *sc = find_sc(h); sc->released = sc->acquired; return XR_SUCCESS;
}

// ---------------------------------------------------------------- actions
static XrResult XRAPI_CALL xrCreateActionSet_(XrInstance i, const XrActionSetCreateInfo *ci, XrActionSet *out) {
    (void)i; ActionSet *s = calloc(1, sizeof *s); snprintf(s->name, 64, "%s", ci->actionSetName); *out = (XrActionSet)s; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroyActionSet_(XrActionSet s) { free(s); return XR_SUCCESS; }
static XrResult XRAPI_CALL xrCreateAction_(XrActionSet set, const XrActionCreateInfo *ci, XrAction *out) {
    (void)set;
    Action *a = calloc(1, sizeof *a);
    a->type = ci->actionType; snprintf(a->name, 64, "%s", ci->actionName);
    a->nsub = ci->countSubactionPaths < 8 ? (int)ci->countSubactionPaths : 8;
    for (int i = 0; i < a->nsub; i++) a->sub[i] = ci->subactionPaths[i];
    *out = (XrAction)a; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroyAction_(XrAction a) { (void)a; return XR_SUCCESS; }   // may still be referenced by suggestions
static XrResult XRAPI_CALL xrSuggestInteractionProfileBindings_(XrInstance i, const XrInteractionProfileSuggestedBinding *sb) {
    (void)i;
    for (uint32_t k = 0; k < sb->countSuggestedBindings && nsugg < 1024; k++)
        sugg[nsugg++] = (Suggestion){sb->interactionProfile, (Action *)sb->suggestedBindings[k].action, sb->suggestedBindings[k].binding};
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrAttachSessionActionSets_(XrSession h, const XrSessionActionSetsAttachInfo *ai) {
    Session *s = (Session *)h; (void)ai;
    XrPath touch = intern("/interaction_profiles/oculus/touch_controller");
    XrPath touchPlus = intern("/interaction_profiles/meta/touch_controller_plus");
    XrPath touchPro = intern("/interaction_profiles/meta/touch_pro_controller");
    XrPath index = intern("/interaction_profiles/valve/index_controller");
    XrPath simple = intern("/interaction_profiles/khr/simple_controller");

    XrPath best = XR_NULL_PATH;
    for (int k = 0; k < nsugg; k++) {
        if (sugg[k].profile == touch) { best = touch; break; }
        if (sugg[k].profile == touchPlus && (!best || best == simple)) best = touchPlus;
        if (sugg[k].profile == touchPro && (!best || best == simple)) best = touchPro;
        if (sugg[k].profile == index && (!best || best == simple)) best = index;
        if (!best && sugg[k].profile != simple) best = sugg[k].profile;
    }
    s->profile = best ? best : (nsugg ? sugg[0].profile : touch);
    for (int k = 0; k < nsugg; k++) {
        Action *a = sugg[k].action;
        if (sugg[k].profile != s->profile || a->nb >= 16) continue;
        if (parse_binding(pstr(sugg[k].binding), &a->b[a->nb])) a->nb++;
    }
    logmsg("attached %d bindings, profile %s", nsugg, pstr(s->profile));
    if (evTail - evHead >= 64) evHead = evTail - 63;
    XrEventDataInteractionProfileChanged *e = (XrEventDataInteractionProfileChanged *)&events[evTail++ % 64];
    memset(e, 0, sizeof(XrEventDataBuffer));
    e->type = XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED; e->session = h;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetCurrentInteractionProfile_(XrSession h, XrPath user, XrInteractionProfileState *st) {
    Session *s = (Session *)h;
    const char *u = pstr(user);
    st->interactionProfile = !strcmp(u, "/user/hand/left") || !strcmp(u, "/user/hand/right") ? s->profile : XR_NULL_PATH;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrSyncActions_(XrSession h, const XrActionsSyncInfo *si) {
    (void)si; Session *s = (Session *)h;
    read_tracking();
    syncGen++;
    return s->focused ? XR_SUCCESS : XR_SESSION_NOT_FOCUSED;
}
static int active(Session *s) { return s->focused && !shm->input_blocked; }
static float action_value(Session *s, Action *a, XrPath sub, XrVector2f *v2, int *bound) {
    float best = 0; *bound = 0;
    if (v2) *v2 = (XrVector2f){0, 0};
    for (int i = 0; i < a->nb; i++) {
        Binding *b = &a->b[i];
        if (!sub_matches(sub, b->hand)) continue;
        *bound = 1;
        if (!active(s)) continue;
        const VR4Hand *hh = &track.hand[b->hand];
        if (v2 && (!strcmp(b->comp, "thumbstick") || !strcmp(b->comp, "trackpad") || !strcmp(b->comp, "joystick") ||
                   !strcmp(b->comp, "thumbstick/2d") || !strcmp(b->comp, "trackpad/2d") || !strcmp(b->comp, "joystick/2d"))) {
            if (fabsf(hh->stick_x) + fabsf(hh->stick_y) > fabsf(v2->x) + fabsf(v2->y)) *v2 = (XrVector2f){hh->stick_x, hh->stick_y};
            continue;
        }
        float v = comp_value(hh, b->comp);
        if (fabsf(v) > fabsf(best)) best = v;
    }
    return best;
}
static int slot(XrPath sub) { const char *p = pstr(sub); return !strcmp(p, "/user/hand/left") ? 0 : !strcmp(p, "/user/hand/right") ? 1 : 2; }
/// changedSinceLastSync: compares this sync's value with the previous sync's, stable across repeated reads.
static int changed(Action *a, XrPath sub, float x, float y) {
    ActionHistory *h = &a->hist[slot(sub)];
    if (h->gen != syncGen) { h->prev[0] = h->cur[0]; h->prev[1] = h->cur[1]; h->cur[0] = x; h->cur[1] = y; h->gen = syncGen; }
    return h->cur[0] != h->prev[0] || h->cur[1] != h->prev[1];
}
static XrResult XRAPI_CALL xrGetActionStateBoolean_(XrSession h, const XrActionStateGetInfo *gi, XrActionStateBoolean *st) {
    int bound; float v = action_value((Session *)h, (Action *)gi->action, gi->subactionPath, NULL, &bound);
    XrBool32 now = v > 0.5f;
    st->changedSinceLastSync = changed((Action *)gi->action, gi->subactionPath, (float)now, 0);
    st->currentState = now; st->isActive = bound; st->lastChangeTime = 0;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetActionStateFloat_(XrSession h, const XrActionStateGetInfo *gi, XrActionStateFloat *st) {
    int bound; float v = action_value((Session *)h, (Action *)gi->action, gi->subactionPath, NULL, &bound);
    st->changedSinceLastSync = changed((Action *)gi->action, gi->subactionPath, v, 0); st->currentState = v; st->isActive = bound; st->lastChangeTime = 0;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetActionStateVector2f_(XrSession h, const XrActionStateGetInfo *gi, XrActionStateVector2f *st) {
    int bound; XrVector2f v; action_value((Session *)h, (Action *)gi->action, gi->subactionPath, &v, &bound);
    st->changedSinceLastSync = changed((Action *)gi->action, gi->subactionPath, v.x, v.y);
    st->currentState = v; st->isActive = bound; st->lastChangeTime = 0;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetActionStatePose_(XrSession h, const XrActionStateGetInfo *gi, XrActionStatePose *st) {
    (void)h; Action *a = (Action *)gi->action;
    st->isActive = XR_FALSE;
    for (int i = 0; i < a->nb; i++)
        if (is_pose_comp(a->b[i].comp) && sub_matches(gi->subactionPath, a->b[i].hand)) {
            if (track.hand[a->b[i].hand].flags & VR4_HAND_ACTIVE) st->isActive = XR_TRUE;
        }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEnumerateBoundSourcesForAction_(XrSession h, const XrBoundSourcesForActionEnumerateInfo *ei, uint32_t cap, uint32_t *n, XrPath *out) {
    (void)h; Action *a = (Action *)ei->action;
    char buf[160];
    FILL_ARRAY(cap, n, out, (uint32_t)a->nb, {
        snprintf(buf, sizeof buf, "/user/hand/%s/input/%s", a->b[i_].hand ? "right" : "left", a->b[i_].comp);
        out[i_] = intern(buf); });
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetInputSourceLocalizedName_(XrSession h, const XrInputSourceLocalizedNameGetInfo *gi, uint32_t cap, uint32_t *n, char *buf) {
    (void)h;
    const char *s = pstr(gi->sourcePath); uint32_t len = (uint32_t)strlen(s) + 1;
    *n = len;
    if (!cap) return XR_SUCCESS;
    if (cap < len) return XR_ERROR_SIZE_INSUFFICIENT;
    memcpy(buf, s, len); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrApplyHapticFeedback_(XrSession h, const XrHapticActionInfo *hi, const XrHapticBaseHeader *hb) {
    (void)h;
    if (hb->type != XR_TYPE_HAPTIC_VIBRATION) return XR_SUCCESS;
    const XrHapticVibration *v = (const XrHapticVibration *)hb;
    Action *a = (Action *)hi->action;
    for (int i = 0; i < a->nb; i++) if (sub_matches(hi->subactionPath, a->b[i].hand)) {
        float dur = v->duration <= 0 ? 0.02f : (float)v->duration / 1e9f;
        shm->haptic = (VR4Haptics){(uint8_t)a->b[i].hand, v->amplitude, dur, v->frequency > 0 ? v->frequency : 0};
        vr4_fence(); shm->haptic_seq++;
    }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrStopHapticFeedback_(XrSession h, const XrHapticActionInfo *hi) {
    (void)h;
    Action *a = (Action *)hi->action;
    for (int i = 0; i < a->nb; i++) if (sub_matches(hi->subactionPath, a->b[i].hand)) {
        shm->haptic = (VR4Haptics){(uint8_t)a->b[i].hand, 0.0f, 0.0f, 0.0f};
        vr4_fence(); shm->haptic_seq++;
    }
    return XR_SUCCESS;
}

// ---------------------------------------------------------------- compatibility extensions
static float current_hz(void) { return shm && shm->fps > 0 ? shm->fps : 72; }
static XrResult XRAPI_CALL xrEnumerateDisplayRefreshRatesFB_(XrSession h, uint32_t cap, uint32_t *n, float *rates) {
    (void)h; FILL_ARRAY(cap, n, rates, 1, rates[i_] = current_hz()); return XR_SUCCESS;   // the rate MacVR negotiated with the headset
}
static XrResult XRAPI_CALL xrGetDisplayRefreshRateFB_(XrSession h, float *rate) { (void)h; *rate = current_hz(); return XR_SUCCESS; }
static XrResult XRAPI_CALL xrRequestDisplayRefreshRateFB_(XrSession h, float rate) {
    (void)h; return rate == 0 || fabsf(rate - current_hz()) < 0.5f ? XR_SUCCESS : XR_ERROR_DISPLAY_REFRESH_RATE_UNSUPPORTED_FB;
}
static XrResult XRAPI_CALL xrGetVisibilityMaskKHR_(XrSession h, XrViewConfigurationType t, uint32_t view, XrVisibilityMaskTypeKHR mt, XrVisibilityMaskKHR *m) {
    (void)h; (void)t; (void)view; (void)mt;   // no hidden-area mask: the whole eye buffer is visible
    m->vertexCountOutput = 0; m->indexCountOutput = 0; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEnumerateColorSpacesFB_(XrSession h, uint32_t cap, uint32_t *n, XrColorSpaceFB *cs) {
    (void)h;
    static const XrColorSpaceFB supported[] = {XR_COLOR_SPACE_QUEST_FB, XR_COLOR_SPACE_REC709_FB, XR_COLOR_SPACE_UNMANAGED_FB};
    FILL_ARRAY(cap, n, cs, 3, cs[i_] = supported[i_]);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrSetColorSpaceFB_(XrSession h, const XrColorSpaceFB cs) {
    (void)h;
    if (cs < XR_COLOR_SPACE_UNMANAGED_FB || cs > XR_COLOR_SPACE_ADOBE_RGB_FB) return XR_ERROR_COLOR_SPACE_UNSUPPORTED_FB;
    return XR_SUCCESS;
}

static XrResult XRAPI_CALL xrSetDebugUtilsObjectNameEXT_(XrInstance i, const XrDebugUtilsObjectNameInfoEXT *n) { (void)i; (void)n; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrCreateDebugUtilsMessengerEXT_(XrInstance i, const XrDebugUtilsMessengerCreateInfoEXT *ci, XrDebugUtilsMessengerEXT *out) {
    (void)i; (void)ci; *out = (XrDebugUtilsMessengerEXT)(uintptr_t)1; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroyDebugUtilsMessengerEXT_(XrDebugUtilsMessengerEXT m) { (void)m; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrSessionBeginDebugUtilsLabelRegionEXT_(XrSession s, const XrDebugUtilsLabelEXT *l) { (void)s; (void)l; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrSessionEndDebugUtilsLabelRegionEXT_(XrSession s) { (void)s; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrSessionInsertDebugUtilsLabelEXT_(XrSession s, const XrDebugUtilsLabelEXT *l) { (void)s; (void)l; return XR_SUCCESS; }

// ---------------------------------------------------------------- dispatch
static XrResult XRAPI_CALL xrGetInstanceProcAddr_(XrInstance inst, const char *name, PFN_xrVoidFunction *fn);
static const struct { const char *name; PFN_xrVoidFunction fn; } table[] = {
#define F(n) {#n, (PFN_xrVoidFunction)n##_},
    F(xrGetInstanceProcAddr) F(xrEnumerateApiLayerProperties) F(xrEnumerateInstanceExtensionProperties) F(xrCreateInstance)
    F(xrDestroyInstance) F(xrGetInstanceProperties) F(xrPollEvent) F(xrResultToString) F(xrStructureTypeToString)
    F(xrStringToPath) F(xrPathToString) F(xrGetSystem) F(xrGetSystemProperties) F(xrEnumerateEnvironmentBlendModes)
    F(xrEnumerateViewConfigurations) F(xrGetViewConfigurationProperties) F(xrEnumerateViewConfigurationViews)
    F(xrGetMetalGraphicsRequirementsKHR) F(xrConvertTimespecTimeToTimeKHR) F(xrConvertTimeToTimespecTimeKHR)
    F(xrCreateSession) F(xrDestroySession) F(xrBeginSession) F(xrEndSession) F(xrRequestExitSession)
    F(xrEnumerateReferenceSpaces) F(xrCreateReferenceSpace) F(xrGetReferenceSpaceBoundsRect) F(xrCreateActionSpace)
    F(xrLocateSpace) F(xrDestroySpace) F(xrWaitFrame) F(xrBeginFrame) F(xrEndFrame) F(xrLocateViews)
    F(xrEnumerateSwapchainFormats) F(xrCreateSwapchain) F(xrDestroySwapchain) F(xrEnumerateSwapchainImages)
    F(xrAcquireSwapchainImage) F(xrWaitSwapchainImage) F(xrReleaseSwapchainImage)
    F(xrCreateActionSet) F(xrDestroyActionSet) F(xrCreateAction) F(xrDestroyAction) F(xrSuggestInteractionProfileBindings)
    F(xrAttachSessionActionSets) F(xrGetCurrentInteractionProfile) F(xrSyncActions) F(xrGetActionStateBoolean)
    F(xrGetActionStateFloat) F(xrGetActionStateVector2f) F(xrGetActionStatePose) F(xrEnumerateBoundSourcesForAction)
    F(xrGetInputSourceLocalizedName) F(xrApplyHapticFeedback) F(xrStopHapticFeedback)
    F(xrEnumerateDisplayRefreshRatesFB) F(xrGetDisplayRefreshRateFB) F(xrRequestDisplayRefreshRateFB) F(xrGetVisibilityMaskKHR)
    F(xrEnumerateColorSpacesFB) F(xrSetColorSpaceFB)
    F(xrSetDebugUtilsObjectNameEXT) F(xrCreateDebugUtilsMessengerEXT) F(xrDestroyDebugUtilsMessengerEXT)
    F(xrSessionBeginDebugUtilsLabelRegionEXT) F(xrSessionEndDebugUtilsLabelRegionEXT) F(xrSessionInsertDebugUtilsLabelEXT)
#undef F
};
static XrResult XRAPI_CALL xrGetInstanceProcAddr_(XrInstance inst, const char *name, PFN_xrVoidFunction *fn) {
    (void)inst;
    for (size_t i = 0; i < sizeof table / sizeof table[0]; i++)
        if (!strcmp(name, table[i].name)) { *fn = table[i].fn; return XR_SUCCESS; }
    *fn = NULL;
    logmsg("unsupported function %s", name);
    return XR_ERROR_FUNCTION_UNSUPPORTED;
}

EXPORT XrResult XRAPI_CALL xrNegotiateLoaderRuntimeInterface(const XrNegotiateLoaderInfo *li, XrNegotiateRuntimeRequest *rr) {
    if (!li || !rr || li->structType != XR_LOADER_INTERFACE_STRUCT_LOADER_INFO || rr->structType != XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST ||
        li->minInterfaceVersion > XR_CURRENT_LOADER_RUNTIME_VERSION || li->maxInterfaceVersion < XR_CURRENT_LOADER_RUNTIME_VERSION)
        return XR_ERROR_INITIALIZATION_FAILED;
    rr->runtimeInterfaceVersion = XR_CURRENT_LOADER_RUNTIME_VERSION;
    rr->runtimeApiVersion = XR_MAKE_VERSION(1, 0, 0);
    rr->getInstanceProcAddr = xrGetInstanceProcAddr_;
    return XR_SUCCESS;
}
