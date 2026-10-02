// libopenvr_api.dylib against a temp shm. Submit: two 8x4 eye textures must land side by side, top row first, BGRA,
// through the asynchronous PBO path (GL 3.2 core) and the synchronous one (legacy GL 2.1); FadeToColor darkens them.
// Also: skeletal input (bones from tracked joints and from controller curls, SteamVR's left/right mirror rules,
// parent/model/compressed spaces, summary curls), events, device properties, the hidden area mesh, overlays,
// chaperone and haptics reaching both hands.
#define GL_SILENCE_DEPRECATION
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl3.h>
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "../openvr_capi.h"
#include "../../common/vr4mac.h"

static VR4Shm *s;
static void *(*get)(const char *, EVRInitError *);
static GLuint tex(uint32_t top, uint32_t bottom) {   // RGBA words; GL row 0 is the image bottom
    uint32_t px[8 * 4];
    for (int i = 0; i < 32; i++) px[i] = i < 8 ? bottom : top;
    GLuint t; glGenTextures(1, &t); glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 8, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    return t;
}
static CGLContextObj context(CGLOpenGLProfile profile) {
    CGLPixelFormatAttribute attrs[] = {kCGLPFAOpenGLProfile, (CGLPixelFormatAttribute)profile, 0};
    CGLPixelFormatObj pf; GLint n; CGLContextObj ctx;
    CGLChoosePixelFormat(attrs, &pf, &n); CGLCreateContext(pf, NULL, &ctx); CGLSetCurrentContext(ctx);
    return ctx;
}
/// Submits a red/blue left eye and green/white right eye and checks the published frame (black when faded).
static void submit_frame(struct VR_IVRCompositor_FnTable *c, int faded) {
    const uint32_t RED = 0xff0000ffu, BLUE = 0xffff0000u, GREEN = 0xff00ff00u, WHITE = 0xffffffffu;   // RGBA bytes, little endian
    Texture_t l = {(void *)(uintptr_t)tex(RED, BLUE), ETextureType_TextureType_OpenGL, EColorSpace_ColorSpace_Auto};
    Texture_t r = {(void *)(uintptr_t)tex(GREEN, WHITE), ETextureType_TextureType_OpenGL, EColorSpace_ColorSpace_Auto};
    TrackedDevicePose_t poses[3];
    c->WaitGetPoses(poses, 3, NULL, 0);
    uint32_t seq = s->frame_seq;
    memset(vr4_frame(s, 0), 0x7f, 16 * 4 * 4); memset(vr4_frame(s, 1), 0x7f, 16 * 4 * 4);   // stale pixels must not pass
    assert(c->Submit(EVREye_Eye_Left, &l, NULL, 0) == 0 && s->frame_seq == seq);
    assert(c->Submit(EVREye_Eye_Right, &r, NULL, 0) == 0);
    for (int i = 0; i < 1000 && s->frame_seq == seq; i++) usleep(1000);   // published by the readback worker
    assert(s->frame_seq == seq + 1);
    uint32_t b = s->frame_seq % 2;
    assert(s->frame_w[b] == 16 && s->frame_h[b] == 4 && s->frame_rgba[b] == 0);
    uint8_t *f = vr4_frame(s, b);
    #define PX(x, y) (f + ((y) * 16 + (x)) * 4)
    if (faded) { for (int i = 0; i < 64; i++) assert(f[4 * i] == 0 && f[4 * i + 1] == 0 && f[4 * i + 2] == 0); return; }
    assert(PX(0, 0)[2] == 255 && PX(0, 0)[0] == 0);     // left eye top row red (B,G,R,A)
    assert(PX(7, 3)[0] == 255 && PX(7, 3)[2] == 0);     // left eye bottom row blue
    assert(PX(8, 0)[1] == 255 && PX(8, 0)[0] == 0);     // right eye top row green
    assert(PX(15, 3)[0] == 255 && PX(15, 3)[2] == 255); // right eye bottom row white
    assert(glGetError() == GL_NO_ERROR);
}

// --- small rigid-transform helpers for the bone checks (quaternions as VRBoneTransform_t: w, x, y, z)
typedef struct { float w, x, y, z; } Q;
static Q qm(Q a, Q b) { return (Q){a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z, a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
                                   a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x, a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w}; }
static void qrot(Q q, const float v[3], float out[3]) {
    Q r = qm(qm(q, (Q){0, v[0], v[1], v[2]}), (Q){q.w, -q.x, -q.y, -q.z}); out[0] = r.x; out[1] = r.y; out[2] = r.z;
}
static Q bq(VRBoneTransform_t b) { return (Q){b.orientation.w, b.orientation.x, b.orientation.y, b.orientation.z}; }
static int near(float a, float b) { return fabsf(a - b) < 2e-4f; }
static int same_rot(Q a, Q b) { float d = fabsf(a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z); return d > 1 - 1e-4f; }

int main(void) {
    char path[] = "/tmp/vr4mac_test_shm_XXXXXX";
    int fd = mkstemp(path); ftruncate(fd, VR4_SHM_SIZE);
    s = mmap(NULL, VR4_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    s->magic = VR4_SHM_MAGIC; s->version = VR4_SHM_VERSION; s->eye_w = 8; s->eye_h = 4; s->fps = 500;
    s->track.head = (VR4Pose){0, 1.6f, 0, 0, 0, 0, 1};
    for (int h = 0; h < 2; h++) s->track.hand[h] = (VR4Hand){VR4_HAND_ACTIVE | VR4_HAND_POSE_VALID, 0, {0, 1, -0.3f, 0, 0, 0, 1}, {0, 1, -0.3f, 0, 0, 0, 1}, 0, 0, 0, 0};
    s->track_seq = 2;
    setenv("VR4MAC_SHM", path, 1);
    CGLContextObj core = context(kCGLOGLPVersion_3_2_Core);

    void *lib = dlopen("../build/libopenvr_api.dylib", RTLD_NOW);
    assert(lib);
    intptr_t (*init)(EVRInitError *, EVRApplicationType) = dlsym(lib, "VR_InitInternal");
    get = dlsym(lib, "VR_GetGenericInterface");
    EVRInitError err; init(&err, EVRApplicationType_VRApplication_Scene); assert(err == 0);
    struct VR_IVRCompositor_FnTable *c = get("FnTable:IVRCompositor_027", &err);
    struct VR_IVRSystem_FnTable *sys = get("FnTable:IVRSystem_022", &err);
    struct VR_IVRInput_FnTable *in = get("FnTable:IVRInput_010", &err);
    struct VR_IVROverlay_FnTable *ov = get("FnTable:IVROverlay_026", &err);
    struct VR_IVRChaperone_FnTable *chap = get("FnTable:IVRChaperone_004", &err);
    struct VR_IVRChaperoneSetup_FnTable *chapSetup = get("FnTable:IVRChaperoneSetup_006", &err);
    assert(c && sys && in && ov && chap && chapSetup);

    submit_frame(c, 0);
    submit_frame(c, 0);   // the next buffer of the ring
    puts("ok  asynchronous readback (GL 3.2 core)");
    c->FadeToColor(0, 0, 0, 0, 1, false);
    submit_frame(c, 1);
    HmdColor_t fc = c->GetCurrentFadeColor(false); assert(fc.a == 1);
    c->FadeToColor(0, 0, 0, 0, 0, false);
    puts("ok  FadeToColor");
    CGLContextObj legacy = context(kCGLOGLPVersion_Legacy);
    submit_frame(c, 0);
    puts("ok  synchronous readback (legacy GL)");
    CGLSetCurrentContext(core); CGLDestroyContext(legacy);

    // events: devices at start, a hand switching to hand tracking, MacVR's menu
    struct VREvent_t ev; int n = 0;
    while (sys->PollNextEvent(&ev, sizeof ev)) { assert(ev.eventType == EVREventType_VREvent_TrackedDeviceActivated); n++; }
    assert(n == 3);
    s->hand_joints[0].tracked = 1; s->input_blocked = 1;
    assert(sys->PollNextEvent(&ev, sizeof ev) && ev.eventType == EVREventType_VREvent_TrackedDeviceUpdated && ev.trackedDeviceIndex == 1);
    assert(sys->PollNextEvent(&ev, sizeof ev) && ev.eventType == EVREventType_VREvent_DashboardActivated && !sys->PollNextEvent(&ev, sizeof ev));
    s->hand_joints[0].tracked = 0; s->input_blocked = 0;
    while (sys->PollNextEvent(&ev, sizeof ev)) {}
    puts("ok  events");

    // device properties
    ETrackedPropertyError pe; char buf[128];
    assert(sys->GetFloatTrackedDeviceProperty(1, ETrackedDeviceProperty_Prop_DeviceBatteryPercentage_Float, &pe) == 1 && pe == 0);
    assert(!sys->GetBoolTrackedDeviceProperty(1, ETrackedDeviceProperty_Prop_DeviceProvidesBatteryStatus_Bool, &pe) && pe == 0);
    assert(sys->GetStringTrackedDeviceProperty(2, ETrackedDeviceProperty_Prop_InputProfilePath_String, buf, sizeof buf, &pe) && !strcmp(buf, "{oculus}/input/touch_profile.json"));
    assert(sys->GetInt32TrackedDeviceProperty(1, ETrackedDeviceProperty_Prop_Axis1Type_Int32, &pe) == EVRControllerAxisType_k_eControllerAxis_Trigger);
    sys->GetFloatTrackedDeviceProperty(5, ETrackedDeviceProperty_Prop_DeviceBatteryPercentage_Float, &pe); assert(pe == ETrackedPropertyError_TrackedProp_InvalidDevice);
    puts("ok  device properties");

    // hidden area mesh: corners in 0..1 texture space; hidden + visible tile the eye
    float area[2] = {0, 0};
    for (int t = 0; t < 2; t++) {
        HiddenAreaMesh_t m = sys->GetHiddenAreaMesh(EVREye_Eye_Left, (EHiddenAreaMeshType)t);
        assert(m.unTriangleCount > 0 && m.pVertexData);
        for (uint32_t i = 0; i < m.unTriangleCount; i++) {
            HmdVector2_t *v = m.pVertexData + 3 * i;
            for (int k = 0; k < 3; k++) assert(v[k].v[0] >= -1e-6f && v[k].v[0] <= 1 + 1e-6f && v[k].v[1] >= -1e-6f && v[k].v[1] <= 1 + 1e-6f);
            area[t] += fabsf((v[1].v[0] - v[0].v[0]) * (v[2].v[1] - v[0].v[1]) - (v[1].v[1] - v[0].v[1]) * (v[2].v[0] - v[0].v[0])) / 2;
        }
    }
    assert(area[0] > 0.01f && area[0] < 0.2f && fabsf(area[0] + area[1] - 1) < 1e-3f);
    assert(sys->GetHiddenAreaMesh(EVREye_Eye_Right, EHiddenAreaMeshType_k_eHiddenAreaMesh_LineLoop).unTriangleCount == 32);
    printf("ok  hidden area mesh (%.1f%% hidden)\n", 100 * area[0]);

    // chaperone
    float px, pz; assert(chap->GetPlayAreaSize(&px, &pz) && px == 2 && pz == 2);
    HmdQuad_t walls[4]; uint32_t nw = 0;
    assert(!chapSetup->GetLiveCollisionBoundsInfo(NULL, &nw) && nw == 4 && chapSetup->GetLiveCollisionBoundsInfo(walls, &nw) && walls[0].vCorners[2].v[1] > 2);
    puts("ok  chaperone");

    // overlays
    VROverlayHandle_t o, o2;
    assert(ov->CreateOverlay("test.key", "Test", &o) == 0 && o && ov->FindOverlay("test.key", &o2) == 0 && o2 == o);
    assert(ov->CreateOverlay("test.key", "Again", &o2) == EVROverlayError_VROverlayError_KeyInUse);
    HmdMatrix34_t at = {{{1, 0, 0, 0}, {0, 1, 0, 1}, {0, 0, 1, -2}}};
    assert(ov->SetOverlayWidthInMeters(o, 2) == 0 && ov->SetOverlayTransformAbsolute(o, ETrackingUniverseOrigin_TrackingUniverseStanding, &at) == 0);
    VROverlayIntersectionParams_t ip = {{{0.5f, 1, 0}}, {{0, 0, -1}}, ETrackingUniverseOrigin_TrackingUniverseStanding};
    VROverlayIntersectionResults_t ir;
    assert(!ov->ComputeOverlayIntersection(o, &ip, &ir));   // hidden
    assert(ov->ShowOverlay(o) == 0 && ov->IsOverlayVisible(o) && ov->ComputeOverlayIntersection(o, &ip, &ir));
    assert(near(ir.vPoint.v[0], 0.5f) && near(ir.vPoint.v[2], -2) && near(ir.vUVs.v[0], 0.75f) && near(ir.vUVs.v[1], 0.5f) && near(ir.fDistance, 2));
    assert(ov->ShowKeyboard(0, 0, 0, "", 10, "", 0) == EVROverlayError_VROverlayError_RequestFailed);
    assert(ov->DestroyOverlay(o) == 0 && ov->FindOverlay("test.key", &o2) == EVROverlayError_VROverlayError_UnknownOverlay);
    puts("ok  overlays");

    // input manifest with skeletons and haptics
    char dir[] = "/tmp/siliconxr_ovr_XXXXXX"; assert(mkdtemp(dir));
    char mf[256], bf[256]; snprintf(mf, sizeof mf, "%s/m.json", dir); snprintf(bf, sizeof bf, "%s/b.json", dir);
    FILE *fp = fopen(bf, "w");
    fputs("{\"bindings\":{\"/actions/default\":{\"skeleton\":[{\"output\":\"/actions/default/in/skel_right\",\"path\":\"/user/hand/right/input/skeleton/right\"}],"
          "\"haptics\":[{\"output\":\"/actions/default/out/buzz_left\",\"path\":\"/user/hand/left/output/haptic\"},"
          "{\"output\":\"/actions/default/out/buzz_right\",\"path\":\"/user/hand/right/output/haptic\"}]}}}", fp); fclose(fp);
    fp = fopen(mf, "w");
    fputs("{\"actions\":[{\"name\":\"/actions/default/in/skel_left\",\"type\":\"skeleton\",\"skeleton\":\"/skeleton/hand/left\"},"
          "{\"name\":\"/actions/default/in/skel_right\",\"type\":\"skeleton\",\"skeleton\":\"/skeleton/hand/right\"},"
          "{\"name\":\"/actions/default/out/buzz_left\",\"type\":\"vibration\"},{\"name\":\"/actions/default/out/buzz_right\",\"type\":\"vibration\"}],"
          "\"action_sets\":[{\"name\":\"/actions/default\"}],\"default_bindings\":[{\"controller_type\":\"oculus_touch\",\"binding_url\":\"b.json\"}]}", fp); fclose(fp);
    assert(in->SetActionManifestPath(mf) == 0);
    VRActionHandle_t sl, sr, bl, br;
    in->GetActionHandle("/actions/default/in/skel_left", &sl); in->GetActionHandle("/actions/default/in/skel_right", &sr);
    in->GetActionHandle("/actions/default/out/buzz_left", &bl); in->GetActionHandle("/actions/default/out/buzz_right", &br);
    VRActiveActionSet_t aset = {0}; in->GetActionSetHandle("/actions/default", &aset.ulActionSet);

    // haptics on both hands back to back: MacVR (1 ms poll) must see both
    {
        uint32_t h0 = s->haptic_seq, seen = 0;
        assert(in->TriggerHapticVibrationAction(bl, 0, 0.05f, 160, 0.5f, 0) == 0 && in->TriggerHapticVibrationAction(br, 0, 0.05f, 160, 0.5f, 0) == 0);
        for (int i = 0; i < 2000 && s->haptic_seq - h0 < 2; i++) {
            uint32_t q0 = s->haptic_seq; usleep(50);
            if (s->haptic_seq != q0) seen |= 1u << s->haptic.hand;
        }
        seen |= 1u << s->haptic.hand;
        assert(s->haptic_seq - h0 == 2 && seen == 3);
        puts("ok  haptics reach both hands");
    }

    // skeleton basics
    uint32_t nb; BoneIndex_t parents[31]; char name[64];
    assert(in->GetBoneCount(sl, &nb) == 0 && nb == 31 && in->GetBoneHierarchy(sl, parents, 31) == 0 && parents[0] == -1 && parents[7] == 6 && parents[26] == 0);
    assert(in->GetBoneName(sl, 1, name, sizeof name) == 0 && !strcmp(name, "wrist_l") && in->GetBoneName(sr, 30, name, sizeof name) == 0 && !strcmp(name, "finger_pinky_r_aux"));
    assert(in->GetBoneCount(bl, &nb) == EVRInputError_VRInputError_WrongType);
    InputSkeletalActionData_t sd; assert(in->GetSkeletalActionData(sr, &sd, sizeof sd) == 0 && sd.bActive && sd.activeOrigin == 2);

    // reference poses follow SteamVR's mirror rules: finger bones (parent space) right = -left position, same rotation;
    // the wrist (model space) is mirrored across x
    VRBoneTransform_t L[31], R[31];
    for (int pose = 0; pose < 4; pose++) {
        assert(in->GetSkeletalReferenceTransforms(sl, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Parent, pose, L, 31) == 0);
        assert(in->GetSkeletalReferenceTransforms(sr, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Parent, pose, R, 31) == 0);
        for (int b = 2; b < 26; b++) {
            if (parents[b] == 1) continue;   // metacarpals: parented to the wrist
            for (int k = 0; k < 3; k++) assert(near(R[b].position.v[k], -L[b].position.v[k]));
            assert(same_rot(bq(L[b]), bq(R[b])));
        }
        assert(near(R[1].position.v[0], -L[1].position.v[0]) && near(R[1].position.v[1], L[1].position.v[1]) && near(R[1].position.v[2], L[1].position.v[2]));
        Q lw = bq(L[1]), rw = bq(R[1]); assert(same_rot((Q){lw.w, lw.x, -lw.y, -lw.z}, rw));
    }
    // fist vs open hand: the index tip moves toward the palm
    VRBoneTransform_t open[31], fist[31];
    in->GetSkeletalReferenceTransforms(sl, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model, EVRSkeletalReferencePose_VRSkeletalReferencePose_OpenHand, open, 31);
    in->GetSkeletalReferenceTransforms(sl, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model, EVRSkeletalReferencePose_VRSkeletalReferencePose_Fist, fist, 31);
    float dOpen = 0, dFist = 0;
    for (int k = 0; k < 3; k++) {
        dOpen += powf(open[10].position.v[k] - open[1].position.v[k], 2); dFist += powf(fist[10].position.v[k] - fist[1].position.v[k], 2);
    }
    assert(dFist < 0.6f * dOpen);
    puts("ok  skeleton reference poses (SteamVR left/right mirror rules)");

    // controller hand: model space == parent space chained == decompressed; summary curls from the sensors
    s->track.hand[0].buttons = VR4_BTN_TRIGGER_TOUCH | VR4_BTN_THUMB_TOUCH; s->track.hand[0].trigger = 1; s->track.hand[0].squeeze = 1;
    s->track_seq += 2; in->UpdateActionState(&aset, sizeof aset, 1);   // the app reads input after each update
    VRBoneTransform_t model[31], parent[31], dec[31];
    assert(in->GetSkeletalBoneData(sl, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model, EVRSkeletalMotionRange_VRSkeletalMotionRange_WithoutController, model, 31) == 0);
    assert(in->GetSkeletalBoneData(sl, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Parent, EVRSkeletalMotionRange_VRSkeletalMotionRange_WithoutController, parent, 31) == 0);
    assert(in->GetSkeletalBoneData(sl, 0, 0, model, 30) == EVRInputError_VRInputError_InvalidBoneCount);
    assert(model[0].orientation.w == 1 && model[0].position.v[0] == 0);
    for (int b = 1; b < 31; b++) {   // chain the parent-space bones back up to model space
        VRBoneTransform_t p = parent[parents[b]]; float t[3];
        if (parents[b] > 0) p = model[parents[b]];
        qrot(bq(p), parent[b].position.v, t);
        for (int k = 0; k < 3; k++) assert(near(p.position.v[k] + t[k], model[b].position.v[k]));
        assert(same_rot(qm(bq(p), bq(parent[b])), bq(model[b])));
    }
    char cbuf[1100]; uint32_t need;
    assert(in->GetSkeletalBoneDataCompressed(sl, 1, cbuf, sizeof cbuf, &need) == 0 && need <= sizeof cbuf);
    assert(in->DecompressSkeletalBoneData(cbuf, need, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model, dec, 31) == 0);
    for (int b = 0; b < 31; b++) for (int k = 0; k < 3; k++) assert(near(dec[b].position.v[k], model[b].position.v[k]));
    cbuf[0] ^= 1; assert(in->DecompressSkeletalBoneData(cbuf, need, 0, dec, 31) == EVRInputError_VRInputError_InvalidCompressedData);
    VRSkeletalSummaryData_t sum;
    assert(in->GetSkeletalSummaryData(sl, EVRSummaryType_VRSummaryType_FromDevice, &sum) == 0 && near(sum.flFingerCurl[1], 1) && near(sum.flFingerCurl[3], 1) && sum.flFingerCurl[0] > 0.5f);
    EVRSkeletalTrackingLevel lvl; assert(in->GetSkeletalTrackingLevel(sl, &lvl) == 0 && lvl == EVRSkeletalTrackingLevel_VRSkeletalTracking_Partial);
    puts("ok  controller skeleton (curls from trigger, grip and touch)");

    // tracked hand: the index finger straight along -z from the palm, then bent 90 degrees at the proximal joint
    s->hand_joints[0].tracked = 1;
    VR4Pose *j = s->hand_joints[0].joint;
    for (int k = 0; k < 26; k++) j[k] = (VR4Pose){0, 1, -0.3f, 0, 0, 0, 1};
    for (int k = 6; k <= 10; k++) j[k] = (VR4Pose){0.02f, 1, -0.3f - 0.03f * (k - 6), 0, 0, 0, 1};
    s->track_seq += 2; in->UpdateActionState(&aset, sizeof aset, 1);   // the app reads input after each update
    assert(in->GetSkeletalSummaryData(sl, EVRSummaryType_VRSummaryType_FromDevice, &sum) == 0 && sum.flFingerCurl[1] < 0.01f);
    assert(in->GetSkeletalBoneData(sl, EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model, EVRSkeletalMotionRange_VRSkeletalMotionRange_WithController, model, 31) == 0);
    assert(near(model[8].position.v[0], 0.02f) && near(model[8].position.v[2], -0.06f));   // grip-relative joint positions
    float x[3], ax[3] = {1, 0, 0}; qrot(bq(model[7]), ax, x);
    assert(near(x[2], -1));   // left-hand bones point +X along the finger (toward -z here)
    assert(near(model[27].position.v[2], model[9].position.v[2]) && same_rot(bq(model[27]), bq(model[9])));   // aux = distal
    for (int k = 8; k <= 10; k++) j[k] = (VR4Pose){0.02f, 1 - 0.03f * (k - 7), -0.33f, 0, 0, 0, 1};   // bent down
    s->track_seq += 2; in->UpdateActionState(&aset, sizeof aset, 1);   // the app reads input after each update
    assert(in->GetSkeletalSummaryData(sl, EVRSummaryType_VRSummaryType_FromDevice, &sum) == 0 && near(sum.flFingerCurl[1], 0.5f));
    assert(in->GetSkeletalTrackingLevel(sl, &lvl) == 0 && lvl == EVRSkeletalTrackingLevel_VRSkeletalTracking_Full);
    puts("ok  tracked-hand skeleton (joints, curls)");

    unlink(path); unlink(mf); unlink(bf); rmdir(dir);
    puts("PASS openvr");
    return 0;
}
