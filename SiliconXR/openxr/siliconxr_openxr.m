// SiliconXR OpenXR runtime: libsiliconxr_openxr.dylib for games that run natively on macOS. It is the Mac twin of
// WineXR (runtime/vr4mac_openxr.c), and the session, space, action and pacing logic is kept the same. Poses and input come from the MacVR app
// through shared memory (common/vr4mac.h); Metal eye images are read back into the frames the Mac app streams.
// Graphics: XR_KHR_metal_enable. Built without ARC (Metal objects live in C structs).
// Quad, cylinder and equirect layers are composited on the GPU (composite()); hand tracking comes from the MacVR hands.
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
#include "../siliconxr_shared.h"

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
typedef struct { int hand, hp; char comp[64]; } Binding;   // hp: from the hand-interaction profile (used while that hand is tracked)
typedef struct { uint32_t gen; float cur[2], prev[2]; } ActionHistory;
typedef struct Action { XrActionType type; char name[64]; XrPath sub[8]; int nsub; Binding b[32]; int nb; ActionHistory hist[3]; } Action;
typedef struct { char name[64]; } ActionSet;
typedef struct { XrPath profile; Action *action; XrPath binding; } Suggestion;
typedef struct {
    XrSession session; int ref; XrReferenceSpaceType type; Action *action; XrPath sub; XrPosef offset;
} Space;
typedef struct {
    id<MTLTexture> img[3]; int count, acquired, released; MTLPixelFormat fmt; uint32_t w, h, array, mips;
    XrSwapchainStateFoveationFlagsFB fovFlags; XrFoveationProfileFB fovProfile;   // XR_FB_foveation: a hint, kept for xrGetSwapchainStateFB
} Swapchain;
typedef struct {
    id<MTLCommandQueue> queue; int running, focused, exitRequested;
    id<MTLBuffer> ring[NRING]; volatile int busy[NRING]; size_t ringSize;   // readback buffers, busy while the GPU/copy owns them
    XrPosef localOrigin; V localOriginRaw; XrPath profile;
    XrPath handProfile; int kind, handMode[2];   // hand-interaction profile (if suggested), controller kind, hand-tracked per hand
    int present, perfMetrics;                    // XR_EXT_user_presence state (-1 = not reported yet); XR_META_performance_metrics on
    id<MTLTexture> comp;                         // composition target for extra layers, both eyes side by side
    uint64_t lastPublished;   // displayTime of the last actually-published frame (duplicate-submit skip)
} Session;

static Suggestion sugg[1024]; static int nsugg;
static VR4Tracking track;           // input snapshot (hands/buttons), refreshed by xrSyncActions
static VR4HandJoints joints[2];     // hand-tracking joints of the same snapshot (joints[h].tracked = controller put down)
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

static XrEventDataBuffer *push_event(XrStructureType type) {
    if (evTail - evHead >= 64) evHead = evTail - 63;
    XrEventDataBuffer *e = &events[evTail++ % 64];
    memset(e, 0, sizeof *e);
    e->type = type;
    return e;
}
static void push_state(XrSessionState st) {
    XrEventDataSessionStateChanged *e = (XrEventDataSessionStateChanged *)push_event(XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED);
    e->session = (XrSession)theSession; e->state = st; e->time = qpc_ns() + qpcOffsetNs;
}

/// Head/eye snapshot handed out by the xrWaitFrame that predicted `t` (falls back to the latest).
static const VR4Tracking *frame_for(XrTime t) {
    for (int i = 0; i < 4; i++) if (frameRing[i].time_ns == (uint64_t)t && t) return &frameRing[i];
    return &frameTrack;
}
static void update_velocities(const VR4Tracking *t);
static VR4Eye frame_eye(const VR4Tracking *ft, int e);
static int read_tracking(void) {   // seqlock read of the Mac app's latest tracking sample
    int fresh = sxr_read(shm, &track, joints, &lastSeq);
    if (fresh) update_velocities(&track);
    return fresh;
}

// ---------------------------------------------------------------- input mapping
// Controller profiles we serve from the Touch controllers, best first (an app's best suggested one is bound). Unknown
// profiles rank just above simple_controller: their a/b/x/y/thumbstick/trigger/squeeze names still map.
enum { K_TOUCH, K_INDEX, K_HP, K_WMR, K_VIVE, K_SIMPLE, K_OTHER };
static const struct { const char *path; int kind; } profileTable[] = {
    {"/interaction_profiles/oculus/touch_controller", K_TOUCH},
    {"/interaction_profiles/meta/touch_controller_plus", K_TOUCH}, {"/interaction_profiles/meta/touch_plus_controller", K_TOUCH},
    {"/interaction_profiles/facebook/touch_controller_pro", K_TOUCH}, {"/interaction_profiles/meta/touch_pro_controller", K_TOUCH},
    {"/interaction_profiles/valve/index_controller", K_INDEX}, {"/interaction_profiles/hp/mixed_reality_controller", K_HP},
    {"/interaction_profiles/microsoft/motion_controller", K_WMR}, {"/interaction_profiles/htc/vive_controller", K_VIVE},
    {"/interaction_profiles/khr/simple_controller", K_SIMPLE},
};
#define NPROFILES (int)(sizeof profileTable / sizeof *profileTable)
#define HAND_PROFILE "/interaction_profiles/ext/hand_interaction_ext"
static int profile_index(XrPath p) {   // rank in profileTable; unknown controllers just above simple
    for (int i = 0; i < NPROFILES; i++) if (!strcmp(pstr(p), profileTable[i].path)) return 2 * i;
    return 2 * (NPROFILES - 1) - 1;
}
static int profile_kind(XrPath p) { int i = profile_index(p); return i & 1 ? K_OTHER : profileTable[i / 2].kind; }

static int btn(const VR4Hand *h, uint32_t bit) { return (h->buttons & bit) != 0; }
/// Scalar value of an input component path suffix like "trigger/value" or "a/click", remapped from Touch for every
/// controller profile: Index/HP a,b on the left hand are X,Y; the Vive trackpad is the thumbstick; menu on the right
/// hand of a Vive/WMR/simple controller is B (Touch has no right menu button). /click of an analog input is 0 or 1.
static float comp_value(const VR4Hand *h, int hand, const char *c, int kind) {
    #define IS(p) (!strncmp(c, p, strlen(p)))
    float v = 0;
    uint32_t A = hand ? VR4_BTN_A : VR4_BTN_X, B = hand ? VR4_BTN_B : VR4_BTN_Y;
    if (IS("trigger/touch") || IS("trigger/proximity")) return btn(h, VR4_BTN_TRIGGER_TOUCH);
    if (IS("thumb_meta/proximity") || IS("thumb_fb/proximity")) return btn(h, VR4_BTN_THUMB_TOUCH) || btn(h, VR4_BTN_STICK_TOUCH);
    if (IS("trigger/slide") || IS("thumbrest/force") || IS("stylus_fb/force")) return 0;   // no such sensors on Touch
    if (IS("trigger/force")) return h->trigger > 0.9f ? (h->trigger - 0.9f) * 10 : 0;     // pressure past a full pull
    if (IS("trigger") || IS("select")) v = h->trigger;                                     // value, click, curl
    else if (IS("squeeze") || IS("grip/")) v = h->squeeze;                                 // value, click, force
    else if (IS("trackpad") && kind != K_VIVE) return 0;                                  // Index/WMR pads: the stick is bound already
    else if (IS("thumbstick/x") || IS("trackpad/x") || IS("joystick/x")) return h->stick_x;
    else if (IS("thumbstick/y") || IS("trackpad/y") || IS("joystick/y")) return h->stick_y;
    else if (IS("thumbstick/click") || IS("trackpad/click") || IS("trackpad/force") || IS("joystick/click")) return btn(h, VR4_BTN_STICK_CLICK);
    else if (IS("thumbstick/touch") || IS("trackpad/touch") || IS("joystick/touch")) return btn(h, VR4_BTN_STICK_TOUCH);
    else if (IS("thumbrest/touch")) return btn(h, VR4_BTN_THUMB_TOUCH);
    else if (IS("a/touch") || IS("b/touch") || IS("x/touch") || IS("y/touch")) return btn(h, VR4_BTN_THUMB_TOUCH);
    else if (IS("a/") || IS("x/")) return btn(h, A);
    else if (IS("b/") || IS("y/")) return btn(h, B);
    else if (IS("menu") && hand && (kind == K_VIVE || kind == K_WMR || kind == K_SIMPLE)) return btn(h, B);
    else if (IS("menu") || IS("system")) return btn(h, VR4_BTN_MENU);
    return strstr(c, "/click") ? v > 0.5f : v;
    #undef IS
}
static int is_pose_comp(const char *c) {
    return !strcmp(c, "grip/pose") || !strcmp(c, "aim/pose") || !strcmp(c, "palm_ext/pose") || !strcmp(c, "grip_surface/pose") ||
           !strcmp(c, "pinch_ext/pose") || !strcmp(c, "poke_ext/pose");
}
/// Grasp (XR_EXT_hand_interaction grasp_ext) from how far the middle, ring and little fingers are curled.
static float grasp(int h) {
    float c = (sxr_curl(&joints[h], 2) + sxr_curl(&joints[h], 3) + sxr_curl(&joints[h], 4)) / 3;
    c = (c - 0.3f) / 0.45f;
    return c < 0 ? 0 : c > 1 ? 1 : c;
}
/// Values of the hand-interaction profile while the hand is tracked (MacVR sends the pinch as the trigger).
static float hand_value(int h, const char *c) {
    #define IS(p) (!strncmp(c, p, strlen(p)))
    const VR4Hand *v = &track.hand[h];
    if (IS("pinch_ext/value") || IS("aim_activate_ext/value")) return v->trigger;
    if (IS("pinch_ext/ready_ext") || IS("aim_activate_ext/ready_ext")) return (v->flags & VR4_HAND_PINCH_READY) || v->trigger > 0.5f;
    if (IS("grasp_ext/value")) return grasp(h);
    if (IS("grasp_ext/ready_ext")) return grasp(h) > 0.05f;
    return 0;
    #undef IS
}
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
    if (!strcmp(comp, "poke_ext/pose") || !strcmp(comp, "pinch_ext/pose")) {   // hand interaction: index tip / between thumb and index tips
        const VR4HandJoints *j = &joints[hand];
        *valid = j->tracked != 0;
        if (comp[1] == 'o') return xp(j->joint[XR_HAND_JOINT_INDEX_TIP_EXT]);
        VR4Pose t = j->joint[XR_HAND_JOINT_THUMB_TIP_EXT], i = j->joint[XR_HAND_JOINT_INDEX_TIP_EXT], p = h->aim;
        p.px = (t.px + i.px) / 2; p.py = (t.py + i.py) / 2; p.pz = (t.pz + i.pz) / 2;
        return xp(p);
    }
    return xp(!strcmp(comp, "aim/pose") ? h->aim : h->grip);
}
/// A binding counts while its hand is in its profile's mode: hand-interaction bindings while the hand is tracked.
static int live(const Session *s, const Binding *b) { return b->hp == s->handMode[b->hand]; }

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
        if (is_pose_comp(b->comp) && live((Session *)s->session, b) && sub_matches(s->sub, b->hand)) return pmul(hand_pose(b->hand, b->comp, valid), s->offset);
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
        if (is_pose_comp(b->comp) && live((Session *)s->session, b) && sub_matches(s->sub, b->hand)) {
            int isAim = !strcmp(b->comp, "aim/pose") || !strcmp(b->comp, "pinch_ext/pose");
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
// Depth layers, FB composition-layer settings, local dimming, foveation and performance levels are hints and ignored;
// LOCAL_FLOOR is the local origin on the floor. The Touch Plus/Pro, Index, Vive, HP, WMR and simple profiles are all
// fed from the Touch controllers (comp_value). No XR_EXT_palm_pose: a fake palm (= grip) made OpenComposite
// rotate/shift the hands.
#define E(n, v) {n, v}
static const struct { const char *name; uint32_t ver; } exts[] = {
    E(XR_KHR_METAL_ENABLE_EXTENSION_NAME, XR_KHR_metal_enable_SPEC_VERSION),
    E(XR_KHR_CONVERT_TIMESPEC_TIME_EXTENSION_NAME, XR_KHR_convert_timespec_time_SPEC_VERSION),
    E(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME, XR_FB_display_refresh_rate_SPEC_VERSION),
    E(XR_KHR_COMPOSITION_LAYER_DEPTH_EXTENSION_NAME, XR_KHR_composition_layer_depth_SPEC_VERSION),
    E(XR_KHR_COMPOSITION_LAYER_CYLINDER_EXTENSION_NAME, XR_KHR_composition_layer_cylinder_SPEC_VERSION),
    E(XR_KHR_COMPOSITION_LAYER_EQUIRECT2_EXTENSION_NAME, XR_KHR_composition_layer_equirect2_SPEC_VERSION),
    E(XR_KHR_VISIBILITY_MASK_EXTENSION_NAME, XR_KHR_visibility_mask_SPEC_VERSION),
    E(XR_EXT_LOCAL_FLOOR_EXTENSION_NAME, XR_EXT_local_floor_SPEC_VERSION),
    E(XR_FB_COLOR_SPACE_EXTENSION_NAME, XR_FB_color_space_SPEC_VERSION),
    E(XR_KHR_COMPOSITION_LAYER_COLOR_SCALE_BIAS_EXTENSION_NAME, XR_KHR_composition_layer_color_scale_bias_SPEC_VERSION),
    E(XR_EXT_DEBUG_UTILS_EXTENSION_NAME, XR_EXT_debug_utils_SPEC_VERSION),
    E(XR_EXT_HAND_TRACKING_EXTENSION_NAME, XR_EXT_hand_tracking_SPEC_VERSION),
    E(XR_FB_HAND_TRACKING_AIM_EXTENSION_NAME, XR_FB_hand_tracking_aim_SPEC_VERSION),
    E(XR_EXT_HAND_INTERACTION_EXTENSION_NAME, XR_EXT_hand_interaction_SPEC_VERSION),
    E(XR_FB_TOUCH_CONTROLLER_PRO_EXTENSION_NAME, XR_FB_touch_controller_pro_SPEC_VERSION),
    E(XR_META_TOUCH_CONTROLLER_PLUS_EXTENSION_NAME, XR_META_touch_controller_plus_SPEC_VERSION),
    E(XR_EXT_HP_MIXED_REALITY_CONTROLLER_EXTENSION_NAME, XR_EXT_hp_mixed_reality_controller_SPEC_VERSION),
    E(XR_KHR_LOCATE_SPACES_EXTENSION_NAME, XR_KHR_locate_spaces_SPEC_VERSION),
    E(XR_EXT_USER_PRESENCE_EXTENSION_NAME, XR_EXT_user_presence_SPEC_VERSION),
    E(XR_FB_HAPTIC_PCM_EXTENSION_NAME, XR_FB_haptic_pcm_SPEC_VERSION),
    E(XR_FB_HAPTIC_AMPLITUDE_ENVELOPE_EXTENSION_NAME, XR_FB_haptic_amplitude_envelope_SPEC_VERSION),
    E(XR_META_PERFORMANCE_METRICS_EXTENSION_NAME, XR_META_performance_metrics_SPEC_VERSION),
    E(XR_FB_SWAPCHAIN_UPDATE_STATE_EXTENSION_NAME, XR_FB_swapchain_update_state_SPEC_VERSION),
    E(XR_FB_FOVEATION_EXTENSION_NAME, XR_FB_foveation_SPEC_VERSION),
    E(XR_FB_FOVEATION_CONFIGURATION_EXTENSION_NAME, XR_FB_foveation_configuration_SPEC_VERSION),
    E(XR_FB_COMPOSITION_LAYER_SETTINGS_EXTENSION_NAME, XR_FB_composition_layer_settings_SPEC_VERSION),
    E(XR_META_LOCAL_DIMMING_EXTENSION_NAME, XR_META_local_dimming_SPEC_VERSION),
    E(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME, XR_EXT_performance_settings_SPEC_VERSION),
};
#undef E
#define NEXTS (sizeof exts / sizeof *exts)
static int extOn[NEXTS];   // enabled by the app at xrCreateInstance
static int enabled(const char *name) { for (size_t i = 0; i < NEXTS; i++) if (!strcmp(exts[i].name, name)) return extOn[i]; return 0; }

static XrResult XRAPI_CALL xrEnumerateApiLayerProperties_(uint32_t cap, uint32_t *n, XrApiLayerProperties *p) { (void)cap; (void)p; *n = 0; return XR_SUCCESS; }
static XrResult XRAPI_CALL xrEnumerateInstanceExtensionProperties_(const char *layer, uint32_t cap, uint32_t *n, XrExtensionProperties *p) {
    (void)layer;
    FILL_ARRAY(cap, n, p, (uint32_t)NEXTS, { strcpy(p[i_].extensionName, exts[i_].name); p[i_].extensionVersion = exts[i_].ver; });
    return XR_SUCCESS;
}

static XrResult XRAPI_CALL xrCreateInstance_(const XrInstanceCreateInfo *ci, XrInstance *out) {
    int on[NEXTS] = {0};
    for (uint32_t i = 0; i < ci->enabledExtensionCount; i++) {
        int ok = 0;
        for (size_t j = 0; j < NEXTS; j++) if (!strcmp(ci->enabledExtensionNames[i], exts[j].name)) ok = on[j] = 1;
        if (!ok) { logmsg("unsupported extension %s", ci->enabledExtensionNames[i]); return XR_ERROR_EXTENSION_NOT_PRESENT; }
    }
    memcpy(extOn, on, sizeof on);
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
static XrResult XRAPI_CALL xrDestroyInstance_(XrInstance i) { (void)i; nsugg = 0; return XR_SUCCESS; }   // suggestions are per instance
static XrResult XRAPI_CALL xrGetInstanceProperties_(XrInstance i, XrInstanceProperties *p) {
    (void)i; p->runtimeVersion = XR_MAKE_VERSION(1, 0, 0); strcpy(p->runtimeName, "SiliconXR"); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrPollEvent_(XrInstance i, XrEventDataBuffer *e) {
    (void)i;
    Session *s = theSession;
    if (s && s->running && !s->exitRequested) {   // dashboard open on the Mac side = app loses input focus
        int focused = !shm->input_blocked;
        if (focused != s->focused) { s->focused = focused; push_state(focused ? XR_SESSION_STATE_FOCUSED : XR_SESSION_STATE_VISIBLE); }
        int present = shm->client_connected != 0;   // XR_EXT_user_presence: the headset is linked to MacVR
        if (present != s->present && enabled(XR_EXT_USER_PRESENCE_EXTENSION_NAME)) {
            s->present = present;
            XrEventDataUserPresenceChangedEXT *u = (XrEventDataUserPresenceChangedEXT *)push_event(XR_TYPE_EVENT_DATA_USER_PRESENCE_CHANGED_EXT);
            u->session = (XrSession)s; u->isUserPresent = present;
        }
        static VR4Fov maskFov[2];   // the visibility mask follows the eye's field of view (it arrives with tracking)
        for (int v = 0; v < 2 && enabled(XR_KHR_VISIBILITY_MASK_EXTENSION_NAME); v++) {
            VR4Fov f = frame_eye(&frameTrack, v).fov;
            if (!memcmp(&f, &maskFov[v], sizeof f)) continue;
            maskFov[v] = f;
            XrEventDataVisibilityMaskChangedKHR *m = (XrEventDataVisibilityMaskChangedKHR *)push_event(XR_TYPE_EVENT_DATA_VISIBILITY_MASK_CHANGED_KHR);
            m->session = (XrSession)s; m->viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO; m->viewIndex = (uint32_t)v;
        }
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
        if (next->type == XR_TYPE_SYSTEM_COLOR_SPACE_PROPERTIES_FB) ((XrSystemColorSpacePropertiesFB *)next)->colorSpace = XR_COLOR_SPACE_QUEST_FB;
        if (next->type == XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT) ((XrSystemHandTrackingPropertiesEXT *)next)->supportsHandTracking = XR_TRUE;
        if (next->type == XR_TYPE_SYSTEM_USER_PRESENCE_PROPERTIES_EXT) ((XrSystemUserPresencePropertiesEXT *)next)->supportsUserPresence = XR_TRUE;
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
    s->queue = [(id<MTLCommandQueue>)gb->commandQueue retain]; s->present = -1;
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
    [s->comp release];
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
static float gameMs;              // XR_META_performance_metrics: smoothed game time per frame (ms)
static uint32_t droppedFrames;    // frames not streamed because every readback buffer was still busy
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
    if (lastExit) { waitSum += exitNs - start; workSum += start - lastExit; statN++; gameMs += ((start - lastExit) / 1e6f - gameMs) * 0.1f; }
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

static VR4Eye frame_eye(const VR4Tracking *ft, int e) {
    VR4Eye d = {{e ? 0.032f : -0.032f, 1.6f, 0, 0, 0, 0, 1}, {-0.8f, 0.8f, 0.8f, -0.8f}};   // no headset yet
    return ft->time_ns ? ft->eye[e] : d;
}
static XrResult XRAPI_CALL xrLocateViews_(XrSession h, const XrViewLocateInfo *li, XrViewState *vs, uint32_t cap, uint32_t *n, XrView *views) {
    (void)h;
    int valid;
    XrPosef base = space_in_stage((Space *)li->space, &valid);
    const VR4Tracking *ft = frame_for(li->displayTime);
    vs->viewStateFlags = valid ? (XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT |
                                  XR_VIEW_STATE_ORIENTATION_TRACKED_BIT | XR_VIEW_STATE_POSITION_TRACKED_BIT) : 0;
    FILL_ARRAY(cap, n, views, 2, {
        VR4Eye e = frame_eye(ft, (int)i_);
        views[i_].pose = pmul(pinv(base), xp(e.pose));
        views[i_].fov = (XrFovf){e.fov.left, e.fov.right, e.fov.up, e.fov.down};
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
/// the bytes on the publish queue. `cb` may already hold GPU work (the compositor's pass); nil makes a new one.
/// Returns 0 if no buffer was free (the frame is dropped and `cb` is never committed).
static int readback(Session *s, id<MTLCommandBuffer> cb, int n, id<MTLTexture> const *tex, const uint32_t *slice, const MTLOrigin *org,
                    uint32_t w, uint32_t h, size_t row, void (^done)(const uint8_t *)) {
    int k = ring_slot(s, row * h);
    if (k < 0) { droppedFrames++; return 0; }
    if (!cb) cb = [s->queue commandBuffer];
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
    XrResult r;
    @autoreleasepool { r = xrEndFrameImpl(h, fi); }   // Metal objects made per frame; the game thread may have no pool
    endSum += qpc_ns() - endT0;
    return r;
}
// ---------------------------------------------------------------- compositor
// Frames with more than one plain projection layer (quads, cylinders, equirect2, color scale/bias, or no projection
// at all: menus, video players) are drawn into s->comp, both eyes side by side, then read back like a projection.
// One full-screen pass per layer and eye: the fragment shader casts the pixel's view ray into the layer (quad plane,
// cylinder, sphere) and samples it; projection layers are sampled straight. Layer flags pick the blending.
static const char *shaderSrc =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct L { float4 tan, vp, r0, r1, r2, t, p, uv, scale, bias; int type, slice, pad0, pad1; };\n"
    "struct VO { float4 pos [[position]]; };\n"
    "vertex VO vs(uint id [[vertex_id]]) { VO o; float2 c = float2((id << 1) & 2, id & 2); o.pos = float4(c * 2 - 1, 0, 1); return o; }\n"
    "static bool hit(constant L &l, float2 px, thread float2 &uv) {\n"
    "    float2 f = (px - l.vp.xy) / l.vp.zw, q;\n"
    "    if (l.type == 3) { uv = l.uv.xy + f * l.uv.zw; return true; }\n"
    "    float3 d = float3(mix(l.tan.x, l.tan.y, f.x), mix(l.tan.z, l.tan.w, f.y), -1);\n"
    "    float3 o = l.t.xyz, r = float3(dot(l.r0.xyz, d), dot(l.r1.xyz, d), dot(l.r2.xyz, d));\n"
    "    if (l.type == 0) {\n"   // quad: its XY plane, size p.xy, visible from both sides
    "        if (abs(r.z) < 1e-6) return false;\n"
    "        float s = -o.z / r.z; if (s <= 0) return false;\n"
    "        float3 h = o + s * r; q = float2(h.x / l.p.x + 0.5, 0.5 - h.y / l.p.y);\n"
    "    } else if (l.type == 1) {\n"   // cylinder around +Y: radius p.x, arc p.y centered on -Z, aspect p.z
    "        float a = dot(r.xz, r.xz), b = dot(o.xz, r.xz), c = dot(o.xz, o.xz) - l.p.x * l.p.x, disc = b * b - a * c;\n"
    "        if (a < 1e-9 || disc < 0) return false;\n"
    "        float height = l.p.x * l.p.y / l.p.z; bool found = false;\n"
    "        for (int k = 0; k < 2 && !found; k++) {\n"
    "            float s = (-b + (k ? 1 : -1) * sqrt(disc)) / a; float3 h = o + s * r; float th = atan2(h.x, -h.z);\n"
    "            if (s > 0 && abs(th) <= l.p.y * 0.5 && abs(h.y) <= height * 0.5) { q = float2(th / l.p.y + 0.5, 0.5 - h.y / height); found = true; }\n"
    "        }\n"
    "        if (!found) return false;\n"
    "    } else {\n"   // equirect2: sphere radius p.x (0 = infinite), horizontal angle p.y, upper p.z / lower p.w latitude
    "        float3 v = r;\n"
    "        if (l.p.x > 0) {\n"
    "            float a = dot(r, r), b = dot(o, r), c = dot(o, o) - l.p.x * l.p.x, disc = b * b - a * c;\n"
    "            if (disc < 0) return false;\n"
    "            float s = (-b + sqrt(disc)) / a; if (s <= 0) return false;\n"
    "            v = o + s * r;\n"
    "        }\n"
    "        v = normalize(v); float lon = atan2(v.x, -v.z), lat = asin(clamp(v.y, -1.0, 1.0));\n"
    "        if (abs(lon) > l.p.y * 0.5 || lat > l.p.z || lat < l.p.w) return false;\n"
    "        q = float2(lon / l.p.y + 0.5, (l.p.z - lat) / (l.p.z - l.p.w));\n"
    "    }\n"
    "    if (any(q < 0) || any(q > 1)) return false;\n"
    "    uv = l.uv.xy + q * l.uv.zw; return true;\n"
    "}\n"
    "constexpr sampler sm(filter::linear, address::clamp_to_edge);\n"
    "fragment float4 fs2d(VO in [[stage_in]], constant L &l [[buffer(0)]], texture2d<float> t [[texture(0)]]) {\n"
    "    float2 uv; if (!hit(l, in.pos.xy, uv)) { discard_fragment(); return 0; }\n"
    "    return t.sample(sm, uv) * l.scale + l.bias;\n"
    "}\n"
    "fragment float4 fsArray(VO in [[stage_in]], constant L &l [[buffer(0)]], texture2d_array<float> t [[texture(0)]]) {\n"
    "    float2 uv; if (!hit(l, in.pos.xy, uv)) { discard_fragment(); return 0; }\n"
    "    return t.sample(sm, uv, l.slice) * l.scale + l.bias;\n"
    "}\n";
typedef struct { float tan[4], vp[4], r0[4], r1[4], r2[4], t[4], p[4], uv[4], scale[4], bias[4]; int32_t type, slice, pad[2]; } LayerU;
enum { L_QUAD, L_CYLINDER, L_EQUIRECT, L_PROJECTION };

/// Render pipeline for the composition target format, blend mode (0 opaque, 1 premultiplied, 2 unpremultiplied
/// alpha) and texture type (array or not). Built on first use and cached.
static id<MTLRenderPipelineState> pipeline(id<MTLDevice> dev, MTLPixelFormat fmt, int blend, int array) {
    static id<MTLDevice> libDev; static id<MTLLibrary> lib;
    static struct { MTLPixelFormat fmt; int blend, array; id<MTLRenderPipelineState> ps; } cache[24]; static int n;
    if (dev != libDev) {   // new GPU: start over
        for (int i = 0; i < n; i++) [cache[i].ps release];
        [lib release]; lib = nil; n = 0; libDev = dev;
        NSError *err = nil;
        lib = [dev newLibraryWithSource:@(shaderSrc) options:nil error:&err];
        if (!lib) logmsg("compositor shader: %s", err.localizedDescription.UTF8String);
    }
    for (int i = 0; i < n; i++) if (cache[i].fmt == fmt && cache[i].blend == blend && cache[i].array == array) return cache[i].ps;
    if (!lib || n == 24) return nil;
    MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
    id<MTLFunction> vs = [lib newFunctionWithName:@"vs"], fs = [lib newFunctionWithName:array ? @"fsArray" : @"fs2d"];
    d.vertexFunction = vs; d.fragmentFunction = fs;
    MTLRenderPipelineColorAttachmentDescriptor *c = d.colorAttachments[0];
    c.pixelFormat = fmt;
    if (blend) {
        c.blendingEnabled = YES;
        c.sourceRGBBlendFactor = blend == 2 ? MTLBlendFactorSourceAlpha : MTLBlendFactorOne;
        c.destinationRGBBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
        c.sourceAlphaBlendFactor = MTLBlendFactorOne; c.destinationAlphaBlendFactor = MTLBlendFactorOneMinusSourceAlpha;
    }
    NSError *err = nil;
    id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:d error:&err];
    if (!ps) logmsg("compositor pipeline: %s", err.localizedDescription.UTF8String);
    else { cache[n].fmt = fmt; cache[n].blend = blend; cache[n].array = array; cache[n++].ps = ps; }
    [vs release]; [fs release]; [d release];
    return ps;
}
static const XrCompositionLayerColorScaleBiasKHR *scale_bias(const XrCompositionLayerBaseHeader *l) {
    for (const XrBaseInStructure *n = l->next; n; n = n->next)
        if (n->type == XR_TYPE_COMPOSITION_LAYER_COLOR_SCALE_BIAS_KHR) return (const XrCompositionLayerColorScaleBiasKHR *)n;
    return NULL;
}
/// The released image of a layer's sub-image, if the rect and array index fit its swapchain.
static Swapchain *sub_image(const XrSwapchainSubImage *si) {
    Swapchain *sc = find_sc(si->swapchain);
    if (!sc || sc->released < 0 || si->imageRect.offset.x < 0 || si->imageRect.offset.y < 0 || si->imageRect.extent.width <= 0 ||
        si->imageRect.extent.height <= 0 || (uint32_t)(si->imageRect.offset.x + si->imageRect.extent.width) > sc->w ||
        (uint32_t)(si->imageRect.offset.y + si->imageRect.extent.height) > sc->h || si->imageArrayIndex >= sc->array) return NULL;
    return sc;
}
static int color_format(MTLPixelFormat f) {
    return f == MTLPixelFormatBGRA8Unorm_sRGB || f == MTLPixelFormatRGBA8Unorm_sRGB || f == MTLPixelFormatBGRA8Unorm || f == MTLPixelFormatRGBA8Unorm;
}
static XrResult composite(Session *s, const XrFrameEndInfo *fi, const XrCompositionLayerProjection *proj) {
    uint32_t w = eye_w(), h = eye_h();
    MTLPixelFormat fmt = MTLPixelFormatBGRA8Unorm_sRGB;
    if (proj && proj->viewCount >= 2) {   // the projection sets the frame size and format
        Swapchain *sc = sub_image(&proj->views[0].subImage);
        if (sc) { w = (uint32_t)proj->views[0].subImage.imageRect.extent.width; h = (uint32_t)proj->views[0].subImage.imageRect.extent.height; }
        if (sc && color_format(sc->fmt)) fmt = sc->fmt;
    } else proj = NULL;
    if (2ULL * w * h * 4ULL > (uint64_t)VR4_FRAME_MAX) return XR_SUCCESS;

    // Eyes in stage space: the projection's views if there is one, else this frame's tracked eyes.
    const VR4Tracking *ft = frame_for(fi->displayTime);
    int valid;
    XrPosef projSpace = proj ? space_in_stage((Space *)proj->space, &valid) : IDENT, eye[2];
    float tn[2][4]; Pose2 pose;
    for (int e = 0; e < 2; e++) {
        XrFovf fov;
        if (proj) {
            eye[e] = pmul(projSpace, proj->views[e].pose); fov = proj->views[e].fov;
            XrPosef v = proj->views[e].pose;
            pose.p[e] = (VR4Pose){v.position.x, v.position.y, v.position.z, v.orientation.x, v.orientation.y, v.orientation.z, v.orientation.w};
        } else {
            VR4Eye fe = frame_eye(ft, e);
            eye[e] = xp(fe.pose); fov = (XrFovf){fe.fov.left, fe.fov.right, fe.fov.up, fe.fov.down}; pose.p[e] = fe.pose;
        }
        tn[e][0] = tanf(fov.angleLeft); tn[e][1] = tanf(fov.angleRight); tn[e][2] = tanf(fov.angleUp); tn[e][3] = tanf(fov.angleDown);
    }

    typedef struct { LayerU u[2]; id<MTLTexture> tex[2]; id<MTLRenderPipelineState> ps[2]; } Draw;
    Draw draws[XR_MIN_COMPOSITION_LAYERS_SUPPORTED]; int nd = 0;
    id<MTLDevice> dev = s->queue.device;
    for (uint32_t i = 0; i < fi->layerCount && nd < XR_MIN_COMPOSITION_LAYERS_SUPPORTED; i++) {
        const XrCompositionLayerBaseHeader *l = fi->layers[i];
        if (!l) continue;
        const XrSwapchainSubImage *si[2] = {NULL, NULL}; XrPosef lp = IDENT; float p[4] = {0}; int type, eyes = 3;
        XrEyeVisibility vis = XR_EYE_VISIBILITY_BOTH;
        switch (l->type) {
        case XR_TYPE_COMPOSITION_LAYER_PROJECTION: {
            const XrCompositionLayerProjection *pl = (const XrCompositionLayerProjection *)l;
            if (pl->viewCount < 2) continue;
            type = L_PROJECTION; si[0] = &pl->views[0].subImage; si[1] = &pl->views[1].subImage; break;
        }
        case XR_TYPE_COMPOSITION_LAYER_QUAD: {
            const XrCompositionLayerQuad *q = (const XrCompositionLayerQuad *)l;
            type = L_QUAD; si[0] = si[1] = &q->subImage; lp = q->pose; vis = q->eyeVisibility;
            p[0] = q->size.width; p[1] = q->size.height; break;
        }
        case XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR: {
            const XrCompositionLayerCylinderKHR *c = (const XrCompositionLayerCylinderKHR *)l;
            type = L_CYLINDER; si[0] = si[1] = &c->subImage; lp = c->pose; vis = c->eyeVisibility;
            p[0] = c->radius; p[1] = c->centralAngle; p[2] = c->aspectRatio; break;
        }
        case XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR: {
            const XrCompositionLayerEquirect2KHR *q = (const XrCompositionLayerEquirect2KHR *)l;
            type = L_EQUIRECT; si[0] = si[1] = &q->subImage; lp = q->pose; vis = q->eyeVisibility;
            p[0] = isinf(q->radius) ? 0 : q->radius; p[1] = q->centralHorizontalAngle; p[2] = q->upperVerticalAngle; p[3] = q->lowerVerticalAngle; break;
        }
        default: {
            static int warned; if (!warned++) logmsg("layer type %d is not composited", l->type);
            continue;
        }
        }
        if (vis == XR_EYE_VISIBILITY_LEFT) eyes = 1; else if (vis == XR_EYE_VISIBILITY_RIGHT) eyes = 2;
        if (type == L_CYLINDER && (p[0] <= 0 || p[1] <= 0 || p[2] <= 0)) continue;
        if (type == L_QUAD && (p[0] <= 0 || p[1] <= 0)) continue;
        int blend = l->layerFlags & XR_COMPOSITION_LAYER_BLEND_TEXTURE_SOURCE_ALPHA_BIT
                    ? (l->layerFlags & XR_COMPOSITION_LAYER_UNPREMULTIPLIED_ALPHA_BIT ? 2 : 1) : 0;
        const XrCompositionLayerColorScaleBiasKHR *cb = scale_bias(l);
        XrPosef layer = pmul(space_in_stage((Space *)l->space, &valid), lp);
        Draw *d = &draws[nd]; memset(d, 0, sizeof *d);
        int ok = 1;
        for (int e = 0; e < 2; e++) {
            Swapchain *sc = sub_image(si[e]);
            if (!sc || !(eyes >> e & 1)) { if (!sc) ok = 0; continue; }
            LayerU *u = &d->u[e];
            memcpy(u->tan, tn[e], sizeof u->tan);
            u->vp[0] = (float)(e * w); u->vp[2] = (float)w; u->vp[3] = (float)h;
            XrPosef rel = pmul(pinv(layer), eye[e]);   // eye in layer space
            Q q = rel.orientation;
            float r[3][3] = {{1 - 2 * (q.y * q.y + q.z * q.z), 2 * (q.x * q.y - q.z * q.w), 2 * (q.x * q.z + q.y * q.w)},
                             {2 * (q.x * q.y + q.z * q.w), 1 - 2 * (q.x * q.x + q.z * q.z), 2 * (q.y * q.z - q.x * q.w)},
                             {2 * (q.x * q.z - q.y * q.w), 2 * (q.y * q.z + q.x * q.w), 1 - 2 * (q.x * q.x + q.y * q.y)}};
            memcpy(u->r0, r[0], 12); memcpy(u->r1, r[1], 12); memcpy(u->r2, r[2], 12);
            u->t[0] = rel.position.x; u->t[1] = rel.position.y; u->t[2] = rel.position.z;
            memcpy(u->p, p, sizeof p);
            u->uv[0] = (float)si[e]->imageRect.offset.x / sc->w; u->uv[1] = (float)si[e]->imageRect.offset.y / sc->h;
            u->uv[2] = (float)si[e]->imageRect.extent.width / sc->w; u->uv[3] = (float)si[e]->imageRect.extent.height / sc->h;
            for (int k = 0; k < 4; k++) {
                u->scale[k] = cb ? (&cb->colorScale.r)[k] : 1; u->bias[k] = cb ? (&cb->colorBias.r)[k] : 0;
            }
            u->type = type; u->slice = (int32_t)si[e]->imageArrayIndex;
            d->tex[e] = sc->img[sc->released];
            if (!(d->ps[e] = pipeline(dev, fmt, blend, sc->array > 1))) ok = 0;
        }
        if (ok) nd++;
    }
    if (!nd) return XR_SUCCESS;

    if (!s->comp || s->comp.width != 2 * w || s->comp.height != h || s->comp.pixelFormat != fmt) {
        [s->comp release];   // frames still in flight keep their own reference
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:fmt width:2 * w height:h mipmapped:NO];
        td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead; td.storageMode = MTLStorageModePrivate;
        if (!(s->comp = [dev newTextureWithDescriptor:td])) return XR_SUCCESS;
    }
    id<MTLCommandBuffer> cmd = [s->queue commandBuffer];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = s->comp; rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore; rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 1);
    id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:rp];
    for (int i = 0; i < nd; i++)
        for (int e = 0; e < 2; e++) {
            if (!draws[i].tex[e]) continue;
            [enc setRenderPipelineState:draws[i].ps[e]];
            [enc setViewport:(MTLViewport){(double)e * w, 0, w, h, 0, 1}];
            [enc setFragmentBytes:&draws[i].u[e] length:sizeof(LayerU) atIndex:0];
            [enc setFragmentTexture:draws[i].tex[e] atIndex:0];
            [enc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
        }
    [enc endEncoding];
    id<MTLTexture> tex[1] = {s->comp}; uint32_t slice[1] = {0}; MTLOrigin org[1] = {MTLOriginMake(0, 0, 0)};
    int rgba = is_rgba(fmt);
    uint64_t t = (uint64_t)fi->displayTime;
    uint32_t fw = 2 * w, fh = h;
    readback(s, cmd, 1, tex, slice, org, fw, fh, (size_t)fw * 4, ^(const uint8_t *src) {
        uint32_t buf = (shm->frame_seq + 1) % 2;
        memcpy(vr4_frame(shm, buf), src, (size_t)fw * fh * 4);
        publish(buf, fw, fh, rgba, t, pose);
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
    // Anything but one plain projection (extra layers, color scale/bias, no projection) goes through the compositor.
    if (!proj || proj->viewCount < 2 || fi->layerCount != 1 || scale_bias((const XrCompositionLayerBaseHeader *)proj)) return composite(s, fi, proj);

    Swapchain *sc[2] = { find_sc(proj->views[0].subImage.swapchain), find_sc(proj->views[1].subImage.swapchain) };
    if (!sc[0] || !sc[1]) {
        static int warned_sc; if (!warned_sc++) logmsg("xrEndFrame: invalid swapchain handle");
        return XR_ERROR_HANDLE_INVALID;
    }
    if (sc[0]->released < 0 || sc[1]->released < 0) return XR_SUCCESS;

    uint64_t w = (uint64_t)proj->views[0].subImage.imageRect.extent.width, hgt = (uint64_t)proj->views[0].subImage.imageRect.extent.height;
    if (!w || !hgt || 2ULL * w * hgt * 4ULL > (uint64_t)VR4_FRAME_MAX) {
        static int warned; if (!warned++) logmsg("unsupported eye rect %llux%llu", (unsigned long long)w, (unsigned long long)hgt);
        return XR_SUCCESS;
    }
    MTLPixelFormat fmt = sc[0]->fmt;
    if ((uint64_t)proj->views[1].subImage.imageRect.extent.width != w || (uint64_t)proj->views[1].subImage.imageRect.extent.height != hgt ||
        sc[1]->fmt != fmt) return composite(s, fi, proj);   // eyes differ in size or format: resample them

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
    readback(s, nil, 2, tex, slice, org, (uint32_t)w, (uint32_t)hgt, (size_t)fw * 4, ^(const uint8_t *src) {
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
    int n = 0;   // a new suggestion for a profile replaces the previous one
    for (int k = 0; k < nsugg; k++) if (sugg[k].profile != sb->interactionProfile) sugg[n++] = sugg[k];
    nsugg = n;
    for (uint32_t k = 0; k < sb->countSuggestedBindings && nsugg < 1024; k++)
        sugg[nsugg++] = (Suggestion){sb->interactionProfile, (Action *)sb->suggestedBindings[k].action, sb->suggestedBindings[k].binding};
    return XR_SUCCESS;
}
static void push_profile_changed(Session *s) {
    ((XrEventDataInteractionProfileChanged *)push_event(XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED))->session = (XrSession)s;
}
/// Binds the app's best-ranked controller profile (all of them are fed from the Touch controllers) and, if suggested,
/// XR_EXT_hand_interaction, which takes over a hand while it is hand-tracked.
static XrResult XRAPI_CALL xrAttachSessionActionSets_(XrSession h, const XrSessionActionSetsAttachInfo *ai) {
    Session *s = (Session *)h; (void)ai;
    XrPath hand = intern(HAND_PROFILE), best = XR_NULL_PATH;
    for (int k = 0; k < nsugg; k++)
        if (sugg[k].profile != hand && (!best || profile_index(sugg[k].profile) < profile_index(best))) best = sugg[k].profile;
    s->profile = best ? best : intern(profileTable[0].path);
    s->kind = profile_kind(s->profile);
    s->handProfile = XR_NULL_PATH;
    for (int k = 0; k < nsugg; k++) sugg[k].action->nb = 0;
    for (int k = 0; k < nsugg; k++) {
        Action *a = sugg[k].action;
        int hp = sugg[k].profile == hand;
        if ((sugg[k].profile != s->profile && !hp) || a->nb >= 32) continue;
        if (parse_binding(pstr(sugg[k].binding), &a->b[a->nb])) { a->b[a->nb++].hp = hp; if (hp) s->handProfile = hand; }
    }
    s->handMode[0] = s->handMode[1] = 0;
    logmsg("attached %d bindings, profile %s%s", nsugg, pstr(s->profile), s->handProfile ? " + hand interaction" : "");
    push_profile_changed(s);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetCurrentInteractionProfile_(XrSession h, XrPath user, XrInteractionProfileState *st) {
    Session *s = (Session *)h;
    const char *u = pstr(user);
    int hand = !strcmp(u, "/user/hand/left") ? 0 : !strcmp(u, "/user/hand/right") ? 1 : -1;
    st->interactionProfile = hand < 0 ? XR_NULL_PATH : s->handMode[hand] ? s->handProfile : s->profile;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrSyncActions_(XrSession h, const XrActionsSyncInfo *si) {
    (void)si; Session *s = (Session *)h;
    read_tracking();
    syncGen++;
    int changedMode = 0;   // a hand put its controller down (or picked it up): switch its profile
    for (int k = 0; k < 2; k++) {
        int m = s->handProfile && joints[k].tracked;
        if (m != s->handMode[k]) { s->handMode[k] = m; changedMode = 1; }
    }
    if (changedMode) push_profile_changed(s);
    return s->focused ? XR_SUCCESS : XR_SESSION_NOT_FOCUSED;
}
static int active(Session *s) { return s->focused && !shm->input_blocked; }
static float action_value(Session *s, Action *a, XrPath sub, XrVector2f *v2, int *bound) {
    float best = 0; *bound = 0;
    if (v2) *v2 = (XrVector2f){0, 0};
    for (int i = 0; i < a->nb; i++) {
        Binding *b = &a->b[i];
        if (!sub_matches(sub, b->hand) || !live(s, b)) continue;
        *bound = 1;
        if (!active(s)) continue;
        const VR4Hand *hh = &track.hand[b->hand];
        if (v2 && (!strcmp(b->comp, "thumbstick") || !strcmp(b->comp, "trackpad") || !strcmp(b->comp, "joystick") ||
                   !strcmp(b->comp, "thumbstick/2d") || !strcmp(b->comp, "trackpad/2d") || !strcmp(b->comp, "joystick/2d"))) {
            if (!strncmp(b->comp, "trackpad", 8) && s->kind != K_VIVE) continue;
            if (fabsf(hh->stick_x) + fabsf(hh->stick_y) > fabsf(v2->x) + fabsf(v2->y)) *v2 = (XrVector2f){hh->stick_x, hh->stick_y};
            continue;
        }
        float v = b->hp ? hand_value(b->hand, b->comp) : comp_value(hh, b->hand, b->comp, s->kind);
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
    Session *s = (Session *)h; Action *a = (Action *)gi->action;
    st->isActive = XR_FALSE;
    for (int i = 0; i < a->nb; i++) {
        Binding *b = &a->b[i];
        if (!is_pose_comp(b->comp) || !live(s, b) || !sub_matches(gi->subactionPath, b->hand)) continue;
        if (b->hp ? joints[b->hand].tracked != 0 : (track.hand[b->hand].flags & VR4_HAND_ACTIVE) != 0) st->isActive = XR_TRUE;
    }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrEnumerateBoundSourcesForAction_(XrSession h, const XrBoundSourcesForActionEnumerateInfo *ei, uint32_t cap, uint32_t *n, XrPath *out) {
    Session *s = (Session *)h; Action *a = (Action *)ei->action;
    Binding *b[32]; uint32_t nb = 0;
    for (int i = 0; i < a->nb; i++) if (live(s, &a->b[i])) b[nb++] = &a->b[i];
    char buf[160];
    FILL_ARRAY(cap, n, out, nb, {
        snprintf(buf, sizeof buf, "/user/hand/%s/input/%s", b[i_]->hand ? "right" : "left", b[i_]->comp);
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
/// Hands (bit 0 left, bit 1 right) an output action drives for this subaction path.
static int haptic_hands(const XrHapticActionInfo *hi) {
    Action *a = (Action *)hi->action; int m = 0;
    for (int i = 0; i < a->nb; i++) if (sub_matches(hi->subactionPath, a->b[i].hand)) m |= 1 << a->b[i].hand;
    return m;
}
/// Plain vibrations, plus XR_FB_haptic_pcm (a sample buffer: its peak for its length) and XR_FB_haptic_amplitude_envelope
/// (its peak for its duration): MacVR plays one amplitude/duration/frequency pulse per hand.
static XrResult XRAPI_CALL xrApplyHapticFeedback_(XrSession h, const XrHapticActionInfo *hi, const XrHapticBaseHeader *hb) {
    (void)h;
    float amp = 0, dur = 0, freq = 0;
    if (hb->type == XR_TYPE_HAPTIC_VIBRATION) {
        const XrHapticVibration *v = (const XrHapticVibration *)hb;
        amp = v->amplitude; dur = v->duration <= 0 ? 0.02f : (float)v->duration / 1e9f; freq = v->frequency;
    } else if (hb->type == XR_TYPE_HAPTIC_PCM_VIBRATION_FB) {
        const XrHapticPcmVibrationFB *p = (const XrHapticPcmVibrationFB *)hb;
        for (uint32_t i = 0; i < p->bufferSize; i++) amp = fmaxf(amp, fabsf(p->buffer[i]));
        dur = p->sampleRate > 0 ? p->bufferSize / p->sampleRate : 0;
        if (p->samplesConsumed) *p->samplesConsumed = p->bufferSize;
    } else if (hb->type == XR_TYPE_HAPTIC_AMPLITUDE_ENVELOPE_VIBRATION_FB) {
        const XrHapticAmplitudeEnvelopeVibrationFB *e = (const XrHapticAmplitudeEnvelopeVibrationFB *)hb;
        for (uint32_t i = 0; i < e->amplitudeCount; i++) amp = fmaxf(amp, e->amplitudes[i]);
        dur = (float)e->duration / 1e9f;
    } else return XR_SUCCESS;
    int m = haptic_hands(hi);
    for (int k = 0; k < 2; k++) if (m >> k & 1) sxr_haptic(shm, k, amp, dur, freq, 0);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrStopHapticFeedback_(XrSession h, const XrHapticActionInfo *hi) {
    (void)h;
    int m = haptic_hands(hi);
    for (int k = 0; k < 2; k++) if (m >> k & 1) sxr_haptic(shm, k, 0, 0, 0, 0);
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetDeviceSampleRateFB_(XrSession h, const XrHapticActionInfo *hi, XrDevicePcmSampleRateGetInfoFB *r) {
    (void)h; (void)hi; r->sampleRate = 2000; return XR_SUCCESS;   // Touch PCM haptics rate
}

// ---------------------------------------------------------------- hand tracking (XR_EXT_hand_tracking, XR_FB_hand_tracking_aim)
typedef struct { int hand; } HandTracker;
static XrResult XRAPI_CALL xrCreateHandTrackerEXT_(XrSession h, const XrHandTrackerCreateInfoEXT *ci, XrHandTrackerEXT *out) {
    (void)h;
    if ((ci->hand != XR_HAND_LEFT_EXT && ci->hand != XR_HAND_RIGHT_EXT) || ci->handJointSet != XR_HAND_JOINT_SET_DEFAULT_EXT)
        return XR_ERROR_VALIDATION_FAILURE;
    HandTracker *t = calloc(1, sizeof *t); t->hand = ci->hand == XR_HAND_RIGHT_EXT;
    *out = (XrHandTrackerEXT)t; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroyHandTrackerEXT_(XrHandTrackerEXT t) { free(t); return XR_SUCCESS; }
/// Joints from MacVR while the hand is tracked (controller put down), relative to the base space. Radii are typical
/// adult values; velocities are not reported (velocityFlags 0).
static XrResult XRAPI_CALL xrLocateHandJointsEXT_(XrHandTrackerEXT ht, const XrHandJointsLocateInfoEXT *li, XrHandJointLocationsEXT *loc) {
    static const float radius[XR_HAND_JOINT_COUNT_EXT] = {0.025f, 0.02f, 0.015f, 0.012f, 0.01f, 0.009f,   // palm, wrist, thumb
                                                          0.012f, 0.01f, 0.009f, 0.008f, 0.007f, 0.012f, 0.01f, 0.009f, 0.008f, 0.007f,
                                                          0.012f, 0.009f, 0.008f, 0.0075f, 0.0065f, 0.011f, 0.008f, 0.007f, 0.0065f, 0.006f};
    if (loc->jointCount != XR_HAND_JOINT_COUNT_EXT) return XR_ERROR_VALIDATION_FAILURE;
    int h = ((HandTracker *)ht)->hand, valid;
    XrPosef base = space_in_stage((Space *)li->baseSpace, &valid), inv = pinv(base);
    const VR4HandJoints *j = &joints[h];
    int on = j->tracked && valid && !shm->input_blocked;   // no hand input while MacVR's menu is open
    float ws = current_world_scale();
    loc->isActive = on;
    for (uint32_t i = 0; i < XR_HAND_JOINT_COUNT_EXT; i++) {
        XrHandJointLocationEXT *l = &loc->jointLocations[i];
        l->locationFlags = on ? XR_SPACE_LOCATION_ORIENTATION_VALID_BIT | XR_SPACE_LOCATION_POSITION_VALID_BIT |
                                XR_SPACE_LOCATION_ORIENTATION_TRACKED_BIT | XR_SPACE_LOCATION_POSITION_TRACKED_BIT : 0;
        l->pose = on ? pmul(inv, xp(j->joint[i])) : IDENT;
        l->radius = radius[i] / ws;
    }
    for (XrBaseOutStructure *n = (XrBaseOutStructure *)loc->next; n; n = n->next) {
        if (n->type == XR_TYPE_HAND_JOINT_VELOCITIES_EXT) {
            XrHandJointVelocitiesEXT *v = (XrHandJointVelocitiesEXT *)n;
            for (uint32_t i = 0; i < v->jointCount; i++) v->jointVelocities[i] = (XrHandJointVelocityEXT){0};
        } else if (n->type == XR_TYPE_HAND_TRACKING_AIM_STATE_FB) {
            XrHandTrackingAimStateFB *a = (XrHandTrackingAimStateFB *)n;
            const VR4Hand *vh = &track.hand[h];
            a->status = 0; a->aimPose = IDENT;
            a->pinchStrengthIndex = a->pinchStrengthMiddle = a->pinchStrengthRing = a->pinchStrengthLittle = 0;
            if (!on) continue;
            a->aimPose = pmul(inv, xp(vh->aim));
            a->pinchStrengthIndex = fmaxf(vh->trigger, sxr_pinch(j, 1));
            a->pinchStrengthMiddle = sxr_pinch(j, 2); a->pinchStrengthRing = sxr_pinch(j, 3); a->pinchStrengthLittle = sxr_pinch(j, 4);
            a->status = XR_HAND_TRACKING_AIM_COMPUTED_BIT_FB | XR_HAND_TRACKING_AIM_VALID_BIT_FB;
            if (a->pinchStrengthIndex > 0.9f) a->status |= XR_HAND_TRACKING_AIM_INDEX_PINCHING_BIT_FB;
            if (a->pinchStrengthMiddle > 0.9f) a->status |= XR_HAND_TRACKING_AIM_MIDDLE_PINCHING_BIT_FB;
            if (a->pinchStrengthRing > 0.9f) a->status |= XR_HAND_TRACKING_AIM_RING_PINCHING_BIT_FB;
            if (a->pinchStrengthLittle > 0.9f) a->status |= XR_HAND_TRACKING_AIM_LITTLE_PINCHING_BIT_FB;
            if (h) a->status |= XR_HAND_TRACKING_AIM_DOMINANT_HAND_BIT_FB;
            if (!h && (vh->buttons & VR4_BTN_MENU)) a->status |= XR_HAND_TRACKING_AIM_MENU_PRESSED_BIT_FB;
            // system gesture: the palm (its -Y) faces the headset
            VR4Pose pp = j->joint[XR_HAND_JOINT_PALM_EXT];
            V palmN = qrot((Q){pp.qx, pp.qy, pp.qz, pp.qw}, (V){0, -1, 0});
            V toHead = {track.head.px - pp.px, track.head.py - pp.py, track.head.pz - pp.pz};
            float d = sqrtf(toHead.x * toHead.x + toHead.y * toHead.y + toHead.z * toHead.z);
            if (d > 1e-4f && (palmN.x * toHead.x + palmN.y * toHead.y + palmN.z * toHead.z) / d > 0.7f) a->status |= XR_HAND_TRACKING_AIM_SYSTEM_GESTURE_BIT_FB;
        }
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
/// The lens mask (siliconxr_shared.h) in the view's tangent space (the z = -1 plane) for the eye's current field of view.
static XrResult XRAPI_CALL xrGetVisibilityMaskKHR_(XrSession h, XrViewConfigurationType t, uint32_t view, XrVisibilityMaskTypeKHR mt, XrVisibilityMaskKHR *m) {
    (void)h;
    if (t != XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO) return XR_ERROR_VIEW_CONFIGURATION_TYPE_UNSUPPORTED;
    if (view > 1) return XR_ERROR_VALIDATION_FAILURE;
    float tri[2 * SXR_MASK_N][3][2], ring[SXR_MASK_N][2], rect[SXR_MASK_N][2];
    const float *pts; uint32_t n;
    if (mt == XR_VISIBILITY_MASK_TYPE_LINE_LOOP_KHR) { sxr_mask_ring(ring, rect); pts = &ring[0][0]; n = SXR_MASK_N; }
    else if (mt == XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR || mt == XR_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH_KHR) {
        n = 3 * (uint32_t)sxr_mask_triangles(mt == XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR, tri); pts = &tri[0][0][0];
    } else return XR_ERROR_VALIDATION_FAILURE;
    m->vertexCountOutput = m->indexCountOutput = n;
    if (!m->vertexCapacityInput || !m->indexCapacityInput) return XR_SUCCESS;
    if (m->vertexCapacityInput < n || m->indexCapacityInput < n) return XR_ERROR_SIZE_INSUFFICIENT;
    VR4Fov f = frame_eye(&frameTrack, (int)view).fov;
    float L = tanf(f.left), R = tanf(f.right), U = tanf(f.up), D = tanf(f.down);
    for (uint32_t i = 0; i < n; i++) {
        m->vertices[i] = (XrVector2f){L + (pts[2 * i] + 1) / 2 * (R - L), D + (pts[2 * i + 1] + 1) / 2 * (U - D)};
        m->indices[i] = i;
    }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrLocateSpacesKHR_(XrSession h, const XrSpacesLocateInfoKHR *li, XrSpaceLocationsKHR *out) {
    (void)h;
    if (out->locationCount != li->spaceCount) return XR_ERROR_VALIDATION_FAILURE;
    XrSpaceVelocitiesKHR *vel = NULL;
    for (XrBaseOutStructure *n = (XrBaseOutStructure *)out->next; n; n = n->next)
        if (n->type == XR_TYPE_SPACE_VELOCITIES_KHR) vel = (XrSpaceVelocitiesKHR *)n;
    for (uint32_t i = 0; i < li->spaceCount; i++) {
        XrSpaceVelocity v = {XR_TYPE_SPACE_VELOCITY};
        XrSpaceLocation l = {XR_TYPE_SPACE_LOCATION, vel ? &v : NULL};
        xrLocateSpace_(li->spaces[i], li->baseSpace, li->time, &l);
        out->locations[i] = (XrSpaceLocationDataKHR){l.locationFlags, l.pose};
        if (vel && i < vel->velocityCount) vel->velocities[i] = (XrSpaceVelocityDataKHR){v.velocityFlags, v.linearVelocity, v.angularVelocity};
    }
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrPerfSettingsSetPerformanceLevelEXT_(XrSession h, XrPerfSettingsDomainEXT d, XrPerfSettingsLevelEXT l) {
    (void)h; (void)d; (void)l; return XR_SUCCESS;   // a hint: the Mac runs the game as fast as it can
}

// XR_META_performance_metrics: the game's frame time and frames dropped before streaming.
static const char *perfPaths[] = {"/perfmetrics_meta/app/cpu_frametime", "/perfmetrics_meta/compositor/dropped_frame_count"};
static XrResult XRAPI_CALL xrEnumeratePerformanceMetricsCounterPathsMETA_(XrInstance i, uint32_t cap, uint32_t *n, XrPath *p) {
    (void)i; FILL_ARRAY(cap, n, p, 2, p[i_] = intern(perfPaths[i_])); return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrSetPerformanceMetricsStateMETA_(XrSession h, const XrPerformanceMetricsStateMETA *st) {
    ((Session *)h)->perfMetrics = st->enabled != 0; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetPerformanceMetricsStateMETA_(XrSession h, XrPerformanceMetricsStateMETA *st) {
    st->enabled = ((Session *)h)->perfMetrics; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrQueryPerformanceMetricsCounterMETA_(XrSession h, XrPath path, XrPerformanceMetricsCounterMETA *c) {
    const char *p = pstr(path);
    c->counterFlags = 0; c->uintValue = 0; c->floatValue = 0;
    if (!strcmp(p, perfPaths[0])) {
        c->counterUnit = XR_PERFORMANCE_METRICS_COUNTER_UNIT_MILLISECONDS_META; c->floatValue = gameMs;
        if (((Session *)h)->perfMetrics) c->counterFlags = XR_PERFORMANCE_METRICS_COUNTER_ANY_VALUE_VALID_BIT_META | XR_PERFORMANCE_METRICS_COUNTER_FLOAT_VALUE_VALID_BIT_META;
    } else if (!strcmp(p, perfPaths[1])) {
        c->counterUnit = XR_PERFORMANCE_METRICS_COUNTER_UNIT_GENERIC_META; c->uintValue = droppedFrames;
        if (((Session *)h)->perfMetrics) c->counterFlags = XR_PERFORMANCE_METRICS_COUNTER_ANY_VALUE_VALID_BIT_META | XR_PERFORMANCE_METRICS_COUNTER_UINT_VALUE_VALID_BIT_META;
    } else return XR_ERROR_PATH_UNSUPPORTED;
    return XR_SUCCESS;
}

// XR_FB_foveation (+ configuration, swapchain update state): foveation is a rendering hint we cannot apply to the
// stream; profiles are kept so apps read back what they set.
typedef struct { XrFoveationLevelFB level; float verticalOffset; XrFoveationDynamicFB dynamic; } FoveationProfile;
static XrResult XRAPI_CALL xrCreateFoveationProfileFB_(XrSession h, const XrFoveationProfileCreateInfoFB *ci, XrFoveationProfileFB *out) {
    (void)h;
    FoveationProfile *f = calloc(1, sizeof *f);
    for (const XrBaseInStructure *n = ci->next; n; n = n->next)
        if (n->type == XR_TYPE_FOVEATION_LEVEL_PROFILE_CREATE_INFO_FB) {
            const XrFoveationLevelProfileCreateInfoFB *l = (const XrFoveationLevelProfileCreateInfoFB *)n;
            f->level = l->level; f->verticalOffset = l->verticalOffset; f->dynamic = l->dynamic;
        }
    *out = (XrFoveationProfileFB)f; return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrDestroyFoveationProfileFB_(XrFoveationProfileFB p) { free(p); return XR_SUCCESS; }
static XrResult XRAPI_CALL xrUpdateSwapchainFB_(XrSwapchain h, const XrSwapchainStateBaseHeaderFB *st) {
    Swapchain *sc = find_sc(h);
    if (st->type != XR_TYPE_SWAPCHAIN_STATE_FOVEATION_FB) return XR_ERROR_VALIDATION_FAILURE;
    const XrSwapchainStateFoveationFB *f = (const XrSwapchainStateFoveationFB *)st;
    sc->fovFlags = f->flags; sc->fovProfile = f->profile;
    return XR_SUCCESS;
}
static XrResult XRAPI_CALL xrGetSwapchainStateFB_(XrSwapchain h, XrSwapchainStateBaseHeaderFB *st) {
    Swapchain *sc = find_sc(h);
    if (st->type != XR_TYPE_SWAPCHAIN_STATE_FOVEATION_FB) return XR_ERROR_VALIDATION_FAILURE;
    XrSwapchainStateFoveationFB *f = (XrSwapchainStateFoveationFB *)st;
    f->flags = sc->fovFlags; f->profile = sc->fovProfile;
    return XR_SUCCESS;
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
    F(xrCreateHandTrackerEXT) F(xrDestroyHandTrackerEXT) F(xrLocateHandJointsEXT) F(xrLocateSpacesKHR) F(xrGetDeviceSampleRateFB)
    F(xrPerfSettingsSetPerformanceLevelEXT) F(xrEnumeratePerformanceMetricsCounterPathsMETA) F(xrSetPerformanceMetricsStateMETA)
    F(xrGetPerformanceMetricsStateMETA) F(xrQueryPerformanceMetricsCounterMETA) F(xrCreateFoveationProfileFB) F(xrDestroyFoveationProfileFB)
    F(xrUpdateSwapchainFB) F(xrGetSwapchainStateFB)
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
