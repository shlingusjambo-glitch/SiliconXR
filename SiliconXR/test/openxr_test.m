// End-to-end check of libsiliconxr_openxr.dylib against a temp shm: Metal session, two 8x4 eye swapchains cleared
// red/green, xrEndFrame, then the published frame must be side-by-side BGRA. Also: one action, quad and cylinder layer
// compositing, the visibility mask, hand tracking (joints, FB aim), the hand-interaction profile switching with the
// hand, Index -> Touch remapping, haptics on both hands, user presence and xrLocateSpacesKHR.
#import <Metal/Metal.h>
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#define XR_USE_GRAPHICS_API_METAL
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/openxr_loader_negotiation.h>
#include "../../common/vr4mac.h"

static PFN_xrGetInstanceProcAddr gipa;
static XrInstance inst;
#define FN(name) PFN_##name name; gipa(inst, #name, (PFN_xrVoidFunction *)&name)
#define OK(x) do { XrResult r_ = (x); if (r_ < 0) { fprintf(stderr, "%s -> %d\n", #x, r_); abort(); } } while (0)

static VR4Shm *s;
static id<MTLCommandQueue> q;
static void bump(void) { s->track_seq += 2; }   // the "Mac" published a new sample

/// One swapchain whose image is cleared to a color (BGRA8, w x h), acquired, waited and released.
static XrSwapchain solid(XrSession sess, int w, int h, MTLClearColor c) {
    FN(xrCreateSwapchain); FN(xrEnumerateSwapchainImages); FN(xrAcquireSwapchainImage); FN(xrWaitSwapchainImage); FN(xrReleaseSwapchainImage);
    XrSwapchainCreateInfo ci = {XR_TYPE_SWAPCHAIN_CREATE_INFO, NULL, 0, XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT, MTLPixelFormatBGRA8Unorm, 1, (uint32_t)w, (uint32_t)h, 1, 1, 1};
    XrSwapchain sc; OK(xrCreateSwapchain(sess, &ci, &sc));
    XrSwapchainImageMetalKHR im[3]; uint32_t n;
    for (int i = 0; i < 3; i++) im[i] = (XrSwapchainImageMetalKHR){XR_TYPE_SWAPCHAIN_IMAGE_METAL_KHR};
    OK(xrEnumerateSwapchainImages(sc, 3, &n, (XrSwapchainImageBaseHeader *)im));
    uint32_t idx; OK(xrAcquireSwapchainImage(sc, NULL, &idx));
    XrSwapchainImageWaitInfo wi = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, NULL, XR_INFINITE_DURATION};
    OK(xrWaitSwapchainImage(sc, &wi));
    id<MTLCommandBuffer> cb = [q commandBuffer];
    MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
    rp.colorAttachments[0].texture = (id<MTLTexture>)im[idx].texture; rp.colorAttachments[0].loadAction = MTLLoadActionClear;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore; rp.colorAttachments[0].clearColor = c;
    [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    [cb commit];
    OK(xrReleaseSwapchainImage(sc, NULL));
    return sc;
}
/// Waits for the next published frame; returns its buffer.
static uint8_t *next_frame(uint32_t *seq) {
    for (int i = 0; i < 1000 && s->frame_seq == *seq; i++) usleep(1000);
    assert(s->frame_seq == *seq + 1);
    *seq = s->frame_seq;
    return vr4_frame(s, *seq);
}
static XrPath P(const char *str) { FN(xrStringToPath); XrPath p; OK(xrStringToPath(inst, str, &p)); return p; }
static int has_event(XrStructureType t, XrEventDataBuffer *out) {
    FN(xrPollEvent);
    int found = 0;
    XrEventDataBuffer ev = {XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(inst, &ev) == XR_SUCCESS) { if (ev.type == t) { found = 1; if (out) *out = ev; } ev = (XrEventDataBuffer){XR_TYPE_EVENT_DATA_BUFFER}; }
    return found;
}

int main(void) {
    @autoreleasepool {
    char path[] = "/tmp/siliconxr_test_shm_XXXXXX";
    int fd = mkstemp(path); ftruncate(fd, VR4_SHM_SIZE);
    s = mmap(NULL, VR4_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    s->magic = VR4_SHM_MAGIC; s->version = VR4_SHM_VERSION; s->eye_w = 8; s->eye_h = 4; s->fps = 500;
    s->track.head = (VR4Pose){0, 1.6f, 0, 0, 0, 0, 1};
    s->track.hand[1].trigger = 0.8f; s->track.hand[1].flags = VR4_HAND_ACTIVE | VR4_HAND_POSE_VALID; s->track_seq = 2;
    setenv("VR4MAC_SHM", path, 1);

    void *lib = dlopen("../build/libsiliconxr_openxr.dylib", RTLD_NOW);
    assert(lib);
    PFN_xrNegotiateLoaderRuntimeInterface neg = (PFN_xrNegotiateLoaderRuntimeInterface)dlsym(lib, "xrNegotiateLoaderRuntimeInterface");
    XrNegotiateLoaderInfo li = {XR_LOADER_INTERFACE_STRUCT_LOADER_INFO, XR_LOADER_INFO_STRUCT_VERSION, sizeof li, 1, 1, XR_MAKE_VERSION(1, 0, 0), XR_MAKE_VERSION(1, 1, 0)};
    XrNegotiateRuntimeRequest rr = {XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST, XR_RUNTIME_INFO_STRUCT_VERSION, sizeof rr};
    OK(neg(&li, &rr)); gipa = rr.getInstanceProcAddr;

    FN(xrCreateInstance);
    const char *ext[] = {XR_KHR_METAL_ENABLE_EXTENSION_NAME};
    XrInstanceCreateInfo ici = {XR_TYPE_INSTANCE_CREATE_INFO, NULL, 0, {"siliconxr-test", 1, "", 0, XR_MAKE_VERSION(1, 0, 0)}, 0, NULL, 1, ext};
    OK(xrCreateInstance(&ici, &inst));
    FN(xrGetSystem); FN(xrGetMetalGraphicsRequirementsKHR); FN(xrCreateSession); FN(xrBeginSession);
    FN(xrCreateReferenceSpace); FN(xrWaitFrame); FN(xrBeginFrame); FN(xrEndFrame); FN(xrLocateViews);
    FN(xrCreateActionSet); FN(xrCreateAction); FN(xrSuggestInteractionProfileBindings);
    FN(xrAttachSessionActionSets); FN(xrSyncActions); FN(xrGetActionStateFloat); FN(xrDestroyInstance);

    XrSystemGetInfo sgi = {XR_TYPE_SYSTEM_GET_INFO, NULL, XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
    XrSystemId sys; OK(xrGetSystem(inst, &sgi, &sys));
    XrGraphicsRequirementsMetalKHR req = {XR_TYPE_GRAPHICS_REQUIREMENTS_METAL_KHR};
    OK(xrGetMetalGraphicsRequirementsKHR(inst, sys, &req));
    id<MTLDevice> dev = (id<MTLDevice>)req.metalDevice; q = [dev newCommandQueue];
    XrGraphicsBindingMetalKHR gb = {XR_TYPE_GRAPHICS_BINDING_METAL_KHR, NULL, (void *)q};
    XrSessionCreateInfo sci = {XR_TYPE_SESSION_CREATE_INFO, &gb, 0, sys};
    XrSession sess; OK(xrCreateSession(inst, &sci, &sess));
    has_event(XR_TYPE_EVENT_DATA_BUFFER, NULL);
    XrSessionBeginInfo sbi = {XR_TYPE_SESSION_BEGIN_INFO, NULL, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
    OK(xrBeginSession(sess, &sbi));

    // input: right trigger value through the touch profile
    XrActionSetCreateInfo asci = {XR_TYPE_ACTION_SET_CREATE_INFO, NULL, "game", "Game", 0};
    XrActionSet set; OK(xrCreateActionSet(inst, &asci, &set));
    XrActionCreateInfo aci = {XR_TYPE_ACTION_CREATE_INFO, NULL, "fire", XR_ACTION_TYPE_FLOAT_INPUT, 0, NULL, "Fire"};
    XrAction fire; OK(xrCreateAction(set, &aci, &fire));
    XrActionSuggestedBinding sb = {fire, P("/user/hand/right/input/trigger/value")};
    XrInteractionProfileSuggestedBinding ipsb = {XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING, NULL, P("/interaction_profiles/oculus/touch_controller"), 1, &sb};
    OK(xrSuggestInteractionProfileBindings(inst, &ipsb));
    XrSessionActionSetsAttachInfo att = {XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO, NULL, 1, &set};
    OK(xrAttachSessionActionSets(sess, &att));
    XrActiveActionSet aas = {set, XR_NULL_PATH};
    XrActionsSyncInfo sync = {XR_TYPE_ACTIONS_SYNC_INFO, NULL, 1, &aas};
    OK(xrSyncActions(sess, &sync));
    XrActionStateGetInfo agi = {XR_TYPE_ACTION_STATE_GET_INFO, NULL, fire, XR_NULL_PATH};
    XrActionStateFloat fs = {XR_TYPE_ACTION_STATE_FLOAT};
    OK(xrGetActionStateFloat(sess, &agi, &fs));
    assert(fs.isActive && fs.currentState > 0.79f && fs.currentState < 0.81f);

    // frame 1: projection only (left red, right green)
    XrReferenceSpaceCreateInfo rsci = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO, NULL, XR_REFERENCE_SPACE_TYPE_STAGE, {{0, 0, 0, 1}, {0, 0, 0}}};
    XrSpace stage; OK(xrCreateReferenceSpace(sess, &rsci, &stage));
    rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_VIEW;
    XrSpace view; OK(xrCreateReferenceSpace(sess, &rsci, &view));
    XrSwapchain sc[2] = {solid(sess, 8, 4, MTLClearColorMake(1, 0, 0, 1)), solid(sess, 8, 4, MTLClearColorMake(0, 1, 0, 1))};
    XrFrameState fst = {XR_TYPE_FRAME_STATE};
    OK(xrWaitFrame(sess, NULL, &fst)); OK(xrBeginFrame(sess, NULL));
    XrViewLocateInfo vli = {XR_TYPE_VIEW_LOCATE_INFO, NULL, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, fst.predictedDisplayTime, stage};
    XrViewState vs = {XR_TYPE_VIEW_STATE}; XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}; uint32_t nv;
    OK(xrLocateViews(sess, &vli, &vs, 2, &nv, views));
    XrCompositionLayerProjectionView pv[2];
    for (int e = 0; e < 2; e++)
        pv[e] = (XrCompositionLayerProjectionView){XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW, NULL, views[e].pose, views[e].fov, {sc[e], {{0, 0}, {8, 4}}, 0}};
    XrCompositionLayerProjection layer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION, NULL, 0, stage, 2, pv};
    const XrCompositionLayerBaseHeader *layers[2] = {(XrCompositionLayerBaseHeader *)&layer};
    XrFrameEndInfo fei = {XR_TYPE_FRAME_END_INFO, NULL, fst.predictedDisplayTime, XR_ENVIRONMENT_BLEND_MODE_OPAQUE, 1, layers};
    OK(xrEndFrame(sess, &fei));
    uint32_t seq = 0;
    uint8_t *f = next_frame(&seq);
    assert(s->frame_w[1] == 16 && s->frame_h[1] == 4 && s->frame_rgba[1] == 0);
    assert(f[2] == 255 && f[1] == 0 && f[0] == 0);                   // left eye red (BGRA)
    uint8_t *r = f + 8 * 4;
    assert(r[1] == 255 && r[2] == 0 && r[0] == 0);                   // right eye green
    puts("ok  projection frame");

    // frame 2: projection + a blue 1x1 m quad 1 m in front of the head (VIEW space): covers the eye centers, not the edges
    XrSwapchain blue = solid(sess, 4, 4, MTLClearColorMake(0, 0, 1, 1));
    OK(xrWaitFrame(sess, NULL, &fst)); OK(xrBeginFrame(sess, NULL));
    XrCompositionLayerQuad quad = {XR_TYPE_COMPOSITION_LAYER_QUAD, NULL, 0, view, XR_EYE_VISIBILITY_BOTH, {blue, {{0, 0}, {4, 4}}, 0},
                                   {{0, 0, 0, 1}, {0, 0, -1}}, {1, 1}};
    layers[1] = (XrCompositionLayerBaseHeader *)&quad;
    fei = (XrFrameEndInfo){XR_TYPE_FRAME_END_INFO, NULL, fst.predictedDisplayTime, XR_ENVIRONMENT_BLEND_MODE_OPAQUE, 2, layers};
    OK(xrEndFrame(sess, &fei));
    f = next_frame(&seq);
    assert(s->frame_w[seq % 2] == 16 && s->frame_h[seq % 2] == 4);
    #define PX(x, y) (f + ((y) * 16 + (x)) * 4)
    for (int e = 0; e < 2; e++) {
        assert(PX(e * 8 + 3, 1)[0] == 255 && PX(e * 8 + 4, 2)[0] == 255 && PX(e * 8 + 4, 2)[2] == 0);   // quad over the center
        assert(PX(e * 8, 0)[0] == 0 && (e ? PX(8, 0)[1] : PX(0, 0)[2]) == 255);                          // projection at the edge
    }
    puts("ok  quad layer over the projection");

    // frame 3: no projection, just a blue cylinder around the head (radius 1 m, 1 rad wide, 1 m tall)
    OK(xrWaitFrame(sess, NULL, &fst)); OK(xrBeginFrame(sess, NULL));
    XrCompositionLayerCylinderKHR cyl = {XR_TYPE_COMPOSITION_LAYER_CYLINDER_KHR, NULL, 0, view, XR_EYE_VISIBILITY_BOTH, {blue, {{0, 0}, {4, 4}}, 0},
                                         {{0, 0, 0, 1}, {0, 0, 0}}, 1, 1, 1};
    layers[0] = (XrCompositionLayerBaseHeader *)&cyl;
    fei = (XrFrameEndInfo){XR_TYPE_FRAME_END_INFO, NULL, fst.predictedDisplayTime, XR_ENVIRONMENT_BLEND_MODE_OPAQUE, 1, layers};
    OK(xrEndFrame(sess, &fei));
    f = next_frame(&seq);
    uint32_t fw = s->frame_w[seq % 2], fh = s->frame_h[seq % 2];
    assert(fw == 256 && fh == 128);                                   // the recommended eye size (min 128)
    uint8_t *c = f + ((fh / 2) * fw + fw / 4) * 4, *corner = f;
    assert(c[0] == 255 && c[2] == 0 && corner[0] == 0 && corner[1] == 0 && corner[2] == 0);
    puts("ok  cylinder layer without a projection");

    // frame 4: an infinite equirect2 sphere section, 1 rad wide and +-0.3 rad tall, straight ahead
    OK(xrWaitFrame(sess, NULL, &fst)); OK(xrBeginFrame(sess, NULL));
    XrCompositionLayerEquirect2KHR eq = {XR_TYPE_COMPOSITION_LAYER_EQUIRECT2_KHR, NULL, 0, view, XR_EYE_VISIBILITY_BOTH, {blue, {{0, 0}, {4, 4}}, 0},
                                         {{0, 0, 0, 1}, {0, 0, 0}}, 0, 1, 0.3f, -0.3f};
    layers[0] = (XrCompositionLayerBaseHeader *)&eq; fei.displayTime = fst.predictedDisplayTime;
    OK(xrEndFrame(sess, &fei));
    f = next_frame(&seq);
    c = f + ((fh / 2) * fw + fw / 4) * 4; uint8_t *top = f + ((fh / 8) * fw + fw / 4) * 4;   // top: ~0.6 rad up, outside
    assert(c[0] == 255 && top[0] == 0);
    puts("ok  equirect2 layer");

    // visibility mask: hidden + visible triangles tile the eye's tangent rectangle; the line loop stays inside it
    {
        FN(xrGetVisibilityMaskKHR);
        float area[2] = {0, 0};
        for (int hidden = 0; hidden < 2; hidden++) {
            XrVisibilityMaskKHR m = {XR_TYPE_VISIBILITY_MASK_KHR};
            XrVisibilityMaskTypeKHR t = hidden ? XR_VISIBILITY_MASK_TYPE_HIDDEN_TRIANGLE_MESH_KHR : XR_VISIBILITY_MASK_TYPE_VISIBLE_TRIANGLE_MESH_KHR;
            OK(xrGetVisibilityMaskKHR(sess, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, t, &m));
            assert(m.vertexCountOutput > 0 && m.vertexCountOutput % 3 == 0 && m.indexCountOutput == m.vertexCountOutput);
            XrVector2f *v = calloc(m.vertexCountOutput, sizeof *v); uint32_t *ix = calloc(m.indexCountOutput, 4);
            m.vertexCapacityInput = m.vertexCountOutput; m.indexCapacityInput = m.indexCountOutput; m.vertices = v; m.indices = ix;
            OK(xrGetVisibilityMaskKHR(sess, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, t, &m));
            for (uint32_t i = 0; i < m.indexCountOutput; i += 3) {
                XrVector2f a = v[ix[i]], b = v[ix[i + 1]], cc = v[ix[i + 2]];
                float ar = ((b.x - a.x) * (cc.y - a.y) - (b.y - a.y) * (cc.x - a.x)) / 2;
                assert(ar > 0);   // counter-clockwise
                area[hidden] += ar;
            }
            free(v); free(ix);
        }
        float t = tanf(0.8f), rect = (2 * t) * (2 * t);   // no tracking yet: +-0.8 rad
        assert(area[1] > 0.01f * rect && area[1] < 0.2f * rect && fabsf(area[0] + area[1] - rect) < 1e-3f * rect);
        printf("ok  visibility mask hides %.1f%% (corners)\n", 100 * area[1] / rect);
    }

    // a second instance with hand tracking: hand-interaction profile while the right hand is tracked, Index otherwise
    OK(xrDestroyInstance(inst));
    const char *ext2[] = {XR_KHR_METAL_ENABLE_EXTENSION_NAME, XR_EXT_HAND_TRACKING_EXTENSION_NAME, XR_FB_HAND_TRACKING_AIM_EXTENSION_NAME,
                          XR_EXT_HAND_INTERACTION_EXTENSION_NAME, XR_EXT_USER_PRESENCE_EXTENSION_NAME, XR_KHR_LOCATE_SPACES_EXTENSION_NAME};
    ici.enabledExtensionCount = 6; ici.enabledExtensionNames = ext2;
    OK(xrCreateInstance(&ici, &inst));
    FN(xrCreateHandTrackerEXT); FN(xrLocateHandJointsEXT); FN(xrGetCurrentInteractionProfile); FN(xrGetActionStateBoolean);
    FN(xrApplyHapticFeedback); FN(xrCreateActionSpace); FN(xrLocateSpace); FN(xrLocateSpacesKHR); FN(xrGetSystemProperties);
    {
        PFN_xrCreateSession xrCreateSession2; gipa(inst, "xrCreateSession", (PFN_xrVoidFunction *)&xrCreateSession2);
        XrSystemHandTrackingPropertiesEXT htp = {XR_TYPE_SYSTEM_HAND_TRACKING_PROPERTIES_EXT};
        XrSystemProperties sp = {XR_TYPE_SYSTEM_PROPERTIES, &htp};
        OK(xrGetSystemProperties(inst, sys, &sp)); assert(htp.supportsHandTracking);
        s->client_connected = 1;
        OK(xrCreateSession2(inst, &sci, &sess));
        OK(xrBeginSession(sess, &sbi));
        XrEventDataBuffer ev;
        assert(has_event(XR_TYPE_EVENT_DATA_USER_PRESENCE_CHANGED_EXT, &ev) && ((XrEventDataUserPresenceChangedEXT *)&ev)->isUserPresent);
        puts("ok  user presence event");
    }
    // the right hand puts its controller down: 26 joints along +x from (0.2, 1.2, -0.3), pinching, palm facing down
    s->hand_joints[1].tracked = 1;
    for (int j = 0; j < 26; j++) s->hand_joints[1].joint[j] = (VR4Pose){0.2f + 0.01f * j, 1.2f, -0.3f, 0, 0, 0, 1};
    s->hand_joints[1].joint[10].px = s->hand_joints[1].joint[5].px;   // index tip on the thumb tip
    s->track.hand[1] = (VR4Hand){VR4_HAND_ACTIVE | VR4_HAND_POSE_VALID | VR4_HAND_TRACKED | VR4_HAND_PINCH_READY, 0,
                                 {0.1f, 1.3f, -0.4f, 0, 0, 0, 1}, {0.2f, 1.2f, -0.3f, 0, 0, 0, 1}, 1, 0, 0, 0};
    s->track.hand[0] = (VR4Hand){VR4_HAND_ACTIVE | VR4_HAND_POSE_VALID, VR4_BTN_X, {0}, {0, 0, 0, 0, 0, 0, 1}, 0, 0, 0, 0};
    bump();

    XrActionSet set2; OK(xrCreateActionSet(inst, &asci, &set2));
    XrAction sel, abtn, poke, buzz;
    aci = (XrActionCreateInfo){XR_TYPE_ACTION_CREATE_INFO, NULL, "select", XR_ACTION_TYPE_FLOAT_INPUT, 0, NULL, "Select"}; OK(xrCreateAction(set2, &aci, &sel));
    aci = (XrActionCreateInfo){XR_TYPE_ACTION_CREATE_INFO, NULL, "abtn", XR_ACTION_TYPE_BOOLEAN_INPUT, 0, NULL, "A"}; OK(xrCreateAction(set2, &aci, &abtn));
    aci = (XrActionCreateInfo){XR_TYPE_ACTION_CREATE_INFO, NULL, "poke", XR_ACTION_TYPE_POSE_INPUT, 0, NULL, "Poke"}; OK(xrCreateAction(set2, &aci, &poke));
    aci = (XrActionCreateInfo){XR_TYPE_ACTION_CREATE_INFO, NULL, "buzz", XR_ACTION_TYPE_VIBRATION_OUTPUT, 0, NULL, "Buzz"}; OK(xrCreateAction(set2, &aci, &buzz));
    XrPath indexP = P("/interaction_profiles/valve/index_controller"), handP = P("/interaction_profiles/ext/hand_interaction_ext");
    XrActionSuggestedBinding ib[] = {{sel, P("/user/hand/right/input/trigger/value")}, {abtn, P("/user/hand/left/input/a/click")},
                                     {buzz, P("/user/hand/left/output/haptic")}, {buzz, P("/user/hand/right/output/haptic")}};
    XrActionSuggestedBinding hb[] = {{sel, P("/user/hand/right/input/pinch_ext/value")}, {poke, P("/user/hand/right/input/poke_ext/pose")}};
    ipsb = (XrInteractionProfileSuggestedBinding){XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING, NULL, indexP, 4, ib};
    OK(xrSuggestInteractionProfileBindings(inst, &ipsb));
    ipsb = (XrInteractionProfileSuggestedBinding){XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING, NULL, handP, 2, hb};
    OK(xrSuggestInteractionProfileBindings(inst, &ipsb));
    att.actionSets = &set2; OK(xrAttachSessionActionSets(sess, &att));
    aas.actionSet = set2; OK(xrSyncActions(sess, &sync));
    XrInteractionProfileState ps = {XR_TYPE_INTERACTION_PROFILE_STATE};
    OK(xrGetCurrentInteractionProfile(sess, P("/user/hand/right"), &ps)); assert(ps.interactionProfile == handP);
    OK(xrGetCurrentInteractionProfile(sess, P("/user/hand/left"), &ps)); assert(ps.interactionProfile == indexP);
    agi = (XrActionStateGetInfo){XR_TYPE_ACTION_STATE_GET_INFO, NULL, sel, XR_NULL_PATH};
    OK(xrGetActionStateFloat(sess, &agi, &fs)); assert(fs.isActive && fs.currentState == 1);   // the pinch
    agi.action = abtn; XrActionStateBoolean bs = {XR_TYPE_ACTION_STATE_BOOLEAN};
    OK(xrGetActionStateBoolean(sess, &agi, &bs)); assert(bs.isActive && bs.currentState);           // Index left A = Touch X
    XrActionSpaceCreateInfo asc = {XR_TYPE_ACTION_SPACE_CREATE_INFO, NULL, poke, XR_NULL_PATH, {{0, 0, 0, 1}, {0, 0, 0}}};
    XrSpace pokeSpace; OK(xrCreateActionSpace(sess, &asc, &pokeSpace));
    rsci.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_STAGE; XrSpace stage2; OK(xrCreateReferenceSpace(sess, &rsci, &stage2));
    XrSpaceLocation sl = {XR_TYPE_SPACE_LOCATION};
    OK(xrLocateSpace(pokeSpace, stage2, 1, &sl));
    assert((sl.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) && fabsf(sl.pose.position.x - 0.25f) < 1e-4f);   // index tip (joint 10 = joint 5)
    XrSpaceLocationDataKHR ld; XrSpaceLocationsKHR lks = {XR_TYPE_SPACE_LOCATIONS_KHR, NULL, 1, &ld};
    XrSpacesLocateInfoKHR sli = {XR_TYPE_SPACES_LOCATE_INFO_KHR, NULL, stage2, 1, 1, &pokeSpace};
    OK(xrLocateSpacesKHR(sess, &sli, &lks)); assert(fabsf(ld.pose.position.x - 0.25f) < 1e-4f);
    puts("ok  hand-interaction profile on the tracked hand, Index profile remapped on the other");

    XrHandTrackerCreateInfoEXT hci = {XR_TYPE_HAND_TRACKER_CREATE_INFO_EXT, NULL, XR_HAND_RIGHT_EXT, XR_HAND_JOINT_SET_DEFAULT_EXT};
    XrHandTrackerEXT ht; OK(xrCreateHandTrackerEXT(sess, &hci, &ht));
    XrHandJointLocationEXT jl[26];
    XrHandTrackingAimStateFB aim = {XR_TYPE_HAND_TRACKING_AIM_STATE_FB};
    XrHandJointLocationsEXT locs = {XR_TYPE_HAND_JOINT_LOCATIONS_EXT, &aim, 0, 26, jl};
    XrHandJointsLocateInfoEXT hli = {XR_TYPE_HAND_JOINTS_LOCATE_INFO_EXT, NULL, stage2, 1};
    OK(xrLocateHandJointsEXT(ht, &hli, &locs));
    assert(locs.isActive && (jl[3].locationFlags & XR_SPACE_LOCATION_POSITION_TRACKED_BIT) && fabsf(jl[3].pose.position.x - 0.23f) < 1e-4f);
    assert(jl[0].radius > 0 && fabsf(jl[25].pose.position.y - 1.2f) < 1e-4f);
    assert((aim.status & XR_HAND_TRACKING_AIM_VALID_BIT_FB) && (aim.status & XR_HAND_TRACKING_AIM_INDEX_PINCHING_BIT_FB) &&
           (aim.status & XR_HAND_TRACKING_AIM_DOMINANT_HAND_BIT_FB) && aim.pinchStrengthIndex == 1);
    assert(fabsf(aim.aimPose.position.x - 0.1f) < 1e-4f && fabsf(aim.aimPose.position.z + 0.4f) < 1e-4f);
    puts("ok  hand joints + FB aim state");

    // both hands buzz from one call (null subaction path): MacVR must see two pulses, one per hand
    {
        uint32_t h0 = s->haptic_seq, seen = 0;
        XrHapticVibration vib = {XR_TYPE_HAPTIC_VIBRATION, NULL, 50000000, 0, 0.7f};
        XrHapticActionInfo hai = {XR_TYPE_HAPTIC_ACTION_INFO, NULL, buzz, XR_NULL_PATH};
        OK(xrApplyHapticFeedback(sess, &hai, (XrHapticBaseHeader *)&vib));
        for (int i = 0; i < 2000 && s->haptic_seq - h0 < 2; i++) {
            uint32_t q0 = s->haptic_seq; usleep(50);
            if (s->haptic_seq != q0) { vr4_fence(); seen |= 1u << s->haptic.hand; }
        }
        seen |= 1u << s->haptic.hand;
        assert(s->haptic_seq - h0 == 2 && seen == 3 && s->haptic.amplitude > 0.69f);
        puts("ok  haptics reach both hands");
    }

    // the controller is picked up again: profile changed event, the right hand is back on the Index profile
    s->hand_joints[1].tracked = 0;
    s->track.hand[1].flags = VR4_HAND_ACTIVE | VR4_HAND_POSE_VALID; s->track.hand[1].trigger = 0.3f;
    bump();
    has_event(XR_TYPE_EVENT_DATA_BUFFER, NULL);
    OK(xrSyncActions(sess, &sync));
    assert(has_event(XR_TYPE_EVENT_DATA_INTERACTION_PROFILE_CHANGED, NULL));
    OK(xrGetCurrentInteractionProfile(sess, P("/user/hand/right"), &ps)); assert(ps.interactionProfile == indexP);
    agi.action = sel; OK(xrGetActionStateFloat(sess, &agi, &fs)); assert(fabsf(fs.currentState - 0.3f) < 1e-6f);
    OK(xrLocateHandJointsEXT(ht, &hli, &locs)); assert(!locs.isActive && !(aim.status & XR_HAND_TRACKING_AIM_VALID_BIT_FB));
    puts("ok  controller picked up: profile switches back");

    unlink(path);
    puts("PASS openxr");
    }
    return 0;
}
