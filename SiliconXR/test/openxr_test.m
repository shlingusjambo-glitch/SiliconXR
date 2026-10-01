// End-to-end check of libsiliconxr_openxr.dylib against a temp shm: Metal session, two 8x4 eye swapchains cleared
// red/green, xrEndFrame, then the published frame must be side-by-side BGRA. Also binds and reads one action.
#import <Metal/Metal.h>
#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
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

int main(void) {
    @autoreleasepool {
    char path[] = "/tmp/siliconxr_test_shm_XXXXXX";
    int fd = mkstemp(path); ftruncate(fd, VR4_SHM_SIZE);
    VR4Shm *s = mmap(NULL, VR4_SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    s->magic = VR4_SHM_MAGIC; s->version = VR4_SHM_VERSION; s->eye_w = 8; s->eye_h = 4; s->fps = 500;
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
    FN(xrGetSystem); FN(xrGetMetalGraphicsRequirementsKHR); FN(xrCreateSession); FN(xrPollEvent); FN(xrBeginSession);
    FN(xrCreateReferenceSpace); FN(xrCreateSwapchain); FN(xrEnumerateSwapchainImages); FN(xrAcquireSwapchainImage);
    FN(xrWaitSwapchainImage); FN(xrReleaseSwapchainImage); FN(xrWaitFrame); FN(xrBeginFrame); FN(xrEndFrame); FN(xrLocateViews);
    FN(xrStringToPath); FN(xrCreateActionSet); FN(xrCreateAction); FN(xrSuggestInteractionProfileBindings);
    FN(xrAttachSessionActionSets); FN(xrSyncActions); FN(xrGetActionStateFloat);

    XrSystemGetInfo sgi = {XR_TYPE_SYSTEM_GET_INFO, NULL, XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY};
    XrSystemId sys; OK(xrGetSystem(inst, &sgi, &sys));
    XrGraphicsRequirementsMetalKHR req = {XR_TYPE_GRAPHICS_REQUIREMENTS_METAL_KHR};
    OK(xrGetMetalGraphicsRequirementsKHR(inst, sys, &req));
    id<MTLDevice> dev = (id<MTLDevice>)req.metalDevice; id<MTLCommandQueue> q = [dev newCommandQueue];
    XrGraphicsBindingMetalKHR gb = {XR_TYPE_GRAPHICS_BINDING_METAL_KHR, NULL, (void *)q};
    XrSessionCreateInfo sci = {XR_TYPE_SESSION_CREATE_INFO, &gb, 0, sys};
    XrSession sess; OK(xrCreateSession(inst, &sci, &sess));
    XrEventDataBuffer ev = {XR_TYPE_EVENT_DATA_BUFFER};
    while (xrPollEvent(inst, &ev) == XR_SUCCESS) ev.type = XR_TYPE_EVENT_DATA_BUFFER;
    XrSessionBeginInfo sbi = {XR_TYPE_SESSION_BEGIN_INFO, NULL, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO};
    OK(xrBeginSession(sess, &sbi));

    // input: right trigger value through the touch profile
    XrActionSetCreateInfo asci = {XR_TYPE_ACTION_SET_CREATE_INFO, NULL, "game", "Game", 0};
    XrActionSet set; OK(xrCreateActionSet(inst, &asci, &set));
    XrActionCreateInfo aci = {XR_TYPE_ACTION_CREATE_INFO, NULL, "fire", XR_ACTION_TYPE_FLOAT_INPUT, 0, NULL, "Fire"};
    XrAction fire; OK(xrCreateAction(set, &aci, &fire));
    XrPath profile, trig; OK(xrStringToPath(inst, "/interaction_profiles/oculus/touch_controller", &profile));
    OK(xrStringToPath(inst, "/user/hand/right/input/trigger/value", &trig));
    XrActionSuggestedBinding sb = {fire, trig};
    XrInteractionProfileSuggestedBinding ipsb = {XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING, NULL, profile, 1, &sb};
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

    XrReferenceSpaceCreateInfo rsci = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO, NULL, XR_REFERENCE_SPACE_TYPE_STAGE, {{0, 0, 0, 1}, {0, 0, 0}}};
    XrSpace stage; OK(xrCreateReferenceSpace(sess, &rsci, &stage));
    XrSwapchain sc[2]; id<MTLTexture> img[2];
    for (int e = 0; e < 2; e++) {
        XrSwapchainCreateInfo ci = {XR_TYPE_SWAPCHAIN_CREATE_INFO, NULL, 0, XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT, MTLPixelFormatBGRA8Unorm, 1, 8, 4, 1, 1, 1};
        OK(xrCreateSwapchain(sess, &ci, &sc[e]));
        XrSwapchainImageMetalKHR im[3]; uint32_t n;
        for (int i = 0; i < 3; i++) im[i] = (XrSwapchainImageMetalKHR){XR_TYPE_SWAPCHAIN_IMAGE_METAL_KHR};
        OK(xrEnumerateSwapchainImages(sc[e], 3, &n, (XrSwapchainImageBaseHeader *)im));
        uint32_t idx; OK(xrAcquireSwapchainImage(sc[e], NULL, &idx));
        XrSwapchainImageWaitInfo wi = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO, NULL, XR_INFINITE_DURATION};
        OK(xrWaitSwapchainImage(sc[e], &wi));
        img[e] = (id<MTLTexture>)im[idx].texture;
    }
    XrFrameState fst = {XR_TYPE_FRAME_STATE};
    OK(xrWaitFrame(sess, NULL, &fst)); OK(xrBeginFrame(sess, NULL));
    XrViewLocateInfo vli = {XR_TYPE_VIEW_LOCATE_INFO, NULL, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, fst.predictedDisplayTime, stage};
    XrViewState vs = {XR_TYPE_VIEW_STATE}; XrView views[2] = {{XR_TYPE_VIEW}, {XR_TYPE_VIEW}}; uint32_t nv;
    OK(xrLocateViews(sess, &vli, &vs, 2, &nv, views));
    id<MTLCommandBuffer> cb = [q commandBuffer];
    for (int e = 0; e < 2; e++) {   // left red, right green
        MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor renderPassDescriptor];
        rp.colorAttachments[0].texture = img[e]; rp.colorAttachments[0].loadAction = MTLLoadActionClear;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.colorAttachments[0].clearColor = e ? MTLClearColorMake(0, 1, 0, 1) : MTLClearColorMake(1, 0, 0, 1);
        [[cb renderCommandEncoderWithDescriptor:rp] endEncoding];
    }
    [cb commit];
    XrCompositionLayerProjectionView pv[2];
    for (int e = 0; e < 2; e++) {
        OK(xrReleaseSwapchainImage(sc[e], NULL));
        pv[e] = (XrCompositionLayerProjectionView){XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW, NULL, views[e].pose, views[e].fov, {sc[e], {{0, 0}, {8, 4}}, 0}};
    }
    XrCompositionLayerProjection layer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION, NULL, 0, stage, 2, pv};
    const XrCompositionLayerBaseHeader *layers[] = {(XrCompositionLayerBaseHeader *)&layer};
    XrFrameEndInfo fei = {XR_TYPE_FRAME_END_INFO, NULL, fst.predictedDisplayTime, XR_ENVIRONMENT_BLEND_MODE_OPAQUE, 1, layers};
    OK(xrEndFrame(sess, &fei));
    for (int i = 0; i < 1000 && s->frame_seq == 0; i++) usleep(1000);   // published asynchronously
    assert(s->frame_seq == 1 && s->frame_w[1] == 16 && s->frame_h[1] == 4 && s->frame_rgba[1] == 0);
    uint8_t *f = vr4_frame(s, 1);
    assert(f[2] == 255 && f[1] == 0 && f[0] == 0);                   // left eye red (BGRA)
    uint8_t *r = f + 8 * 4;
    assert(r[1] == 255 && r[2] == 0 && r[0] == 0);                   // right eye green
    unlink(path);
    puts("PASS openxr");
    }
    return 0;
}
