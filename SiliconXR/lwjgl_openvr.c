// liblwjgl_openvr.dylib for Apple Silicon: LWJGL's JNI glue for the OpenVR calls that return structs by value.
// LWJGL ships it for x64 macOS only. Each entry calls the function-table pointer LWJGL passes in and copies the
// result into LWJGL's struct buffer. Signatures: (JNIEnv*, jclass, args..., jlong fn, jlong result).
#include <stdint.h>
#include <string.h>
#include <stdbool.h>
#include "openvr_capi.h"

#define JNI(name) __attribute__((visibility("default"))) void Java_org_lwjgl_openvr_##name
#define P(x) ((void *)(intptr_t)(x))
typedef void *E; typedef void *C;

JNI(VRSystem_nVRSystem_1GetProjectionMatrix)(E e, C c, int32_t eye, float n, float f, int64_t fn, int64_t r) {
    HmdMatrix44_t m = ((HmdMatrix44_t (*)(EVREye, float, float))P(fn))((EVREye)eye, n, f); memcpy(P(r), &m, sizeof m);
}
JNI(VRSystem_nVRSystem_1GetEyeToHeadTransform)(E e, C c, int32_t eye, int64_t fn, int64_t r) {
    HmdMatrix34_t m = ((HmdMatrix34_t (*)(EVREye))P(fn))((EVREye)eye); memcpy(P(r), &m, sizeof m);
}
JNI(VRSystem_nVRSystem_1GetSeatedZeroPoseToStandingAbsoluteTrackingPose)(E e, C c, int64_t fn, int64_t r) {
    HmdMatrix34_t m = ((HmdMatrix34_t (*)(void))P(fn))(); memcpy(P(r), &m, sizeof m);
}
JNI(VRSystem_nVRSystem_1GetRawZeroPoseToStandingAbsoluteTrackingPose)(E e, C c, int64_t fn, int64_t r) {
    HmdMatrix34_t m = ((HmdMatrix34_t (*)(void))P(fn))(); memcpy(P(r), &m, sizeof m);
}
JNI(VRSystem_nVRSystem_1GetMatrix34TrackedDeviceProperty)(E e, C c, int32_t dev, int32_t prop, int64_t err, int64_t fn, int64_t r) {
    HmdMatrix34_t m = ((HmdMatrix34_t (*)(TrackedDeviceIndex_t, ETrackedDeviceProperty, ETrackedPropertyError *))P(fn))((TrackedDeviceIndex_t)dev, (ETrackedDeviceProperty)prop, P(err));
    memcpy(P(r), &m, sizeof m);
}
JNI(VRSystem_nVRSystem_1GetHiddenAreaMesh)(E e, C c, int32_t eye, int32_t type, int64_t fn, int64_t r) {
    HiddenAreaMesh_t m = ((HiddenAreaMesh_t (*)(EVREye, EHiddenAreaMeshType))P(fn))((EVREye)eye, (EHiddenAreaMeshType)type); memcpy(P(r), &m, sizeof m);
}
JNI(VRChaperone_nVRChaperone_1SetSceneColor)(E e, C c, int64_t color, int64_t fn) {
    ((void (*)(HmdColor_t))P(fn))(*(HmdColor_t *)P(color));
}
JNI(VRCompositor_nVRCompositor_1GetCurrentFadeColor)(E e, C c, uint8_t bg, int64_t fn, int64_t r) {
    HmdColor_t m = ((HmdColor_t (*)(bool))P(fn))(bg); memcpy(P(r), &m, sizeof m);
}
JNI(VROverlay_nVROverlay_1SetKeyboardPositionForOverlay)(E e, C c, int64_t overlay, int64_t rect, int64_t fn) {
    ((void (*)(VROverlayHandle_t, HmdRect2_t))P(fn))((VROverlayHandle_t)overlay, *(HmdRect2_t *)P(rect));
}
__attribute__((visibility("default"))) int32_t Java_org_lwjgl_openvr_VROverlay_nVROverlay_1GetTransformForOverlayCoordinates(E e, C c, int64_t overlay, int32_t origin, int64_t coords, int64_t out, int64_t fn) {
    return ((EVROverlayError (*)(VROverlayHandle_t, ETrackingUniverseOrigin, HmdVector2_t, HmdMatrix34_t *))P(fn))((VROverlayHandle_t)overlay, (ETrackingUniverseOrigin)origin, *(HmdVector2_t *)P(coords), P(out));
}

// Apple Silicon LWJGL core lacks the JNI.call* trampolines only OpenVR uses (callPPI(IIJIJJ) etc.); register them
// on org.lwjgl.system.JNI when this library loads (OpenVR's static init, before any OpenVR call). JNIEnv/JavaVM are
// used by vtable index: GetEnv 6, FindClass 6, ExceptionClear 17, RegisterNatives 215, ExceptionCheck 228.
#include "jni_calls.h"
__attribute__((visibility("default"))) int32_t JNI_OnLoad(void *vm, void *reserved) {
    const int32_t version = 0x00010006;   // JNI_VERSION_1_6
    void *env = NULL;
    void **vmf = *(void ***)vm;
    if (((int32_t (*)(void *, void **, int32_t))vmf[6])(vm, &env, version) != 0) return version;
    void **f = *(void ***)env;
    void *cls = ((void *(*)(void *, const char *))f[6])(env, "org/lwjgl/system/JNI");
    if (!cls) { ((void (*)(void *))f[17])(env); return version; }
    for (size_t i = 0; i < sizeof jni_calls / sizeof *jni_calls; i++) {
        ((int32_t (*)(void *, void *, const void *, int32_t))f[215])(env, cls, &jni_calls[i], 1);
        if (((uint8_t (*)(void *))f[228])(env)) ((void (*)(void *))f[17])(env);   // signature absent in this LWJGL: skip
    }
    return version;
}
