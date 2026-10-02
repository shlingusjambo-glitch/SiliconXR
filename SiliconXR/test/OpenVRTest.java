// Drives build/libopenvr_api.dylib through LWJGL's OpenVR bindings the way Vivecraft does, against the shm in
// VR4MAC_SHM (run.sh makes a temporary one). Also calls skeletal input, overlays, events and the hidden area mesh, so
// the Apple Silicon JNI trampolines (jni_calls.h) for them are exercised.
import org.lwjgl.openvr.*;
import org.lwjgl.system.MemoryStack;
import java.nio.*;
import java.nio.file.*;
import static org.lwjgl.openvr.VR.*;

public class OpenVRTest {
    static void check(boolean ok, String what) { if (!ok) throw new AssertionError(what); System.out.println("ok  " + what); }
    public static void main(String[] a) throws Exception {
        try (MemoryStack s = MemoryStack.stackPush()) {
            IntBuffer err = s.mallocInt(1);
            int token = VR_InitInternal(err, EVRApplicationType_VRApplication_Scene);
            check(err.get(0) == 0, "VR_InitInternal");
            OpenVR.create(token);
            check(OpenVR.VRSystem != null && OpenVR.VRCompositor != null && OpenVR.VRInput != null && OpenVR.VRApplications != null && OpenVR.VRRenderModels != null, "interfaces");
            IntBuffer w = s.mallocInt(1), h = s.mallocInt(1);
            VRSystem.VRSystem_GetRecommendedRenderTargetSize(w, h);
            check(w.get(0) >= 128 && h.get(0) >= 128, "render size " + w.get(0) + "x" + h.get(0));
            HmdMatrix44 p = VRSystem.VRSystem_GetProjectionMatrix(EVREye_Eye_Left, 0.05f, 100f, HmdMatrix44.calloc(s));
            check(p.m(0) > 0 && p.m(14) == -1f, "projection (JNI glue) m00=" + p.m(0));
            HmdMatrix34 e = VRSystem.VRSystem_GetEyeToHeadTransform(EVREye_Eye_Right, HmdMatrix34.calloc(s));
            check(e.m(3) > 0 && e.m(3) < 0.05f, "right eye offset " + e.m(3));
            check(VRSystem.VRSystem_GetStringTrackedDeviceProperty(1, ETrackedDeviceProperty_Prop_ControllerType_String, err).equals("oculus_touch"), "controller type");

            Path dir = Files.createTempDirectory("macvr-ovr");
            Files.writeString(dir.resolve("b.json"), "{\"bindings\":{\"/actions/ingame\":{\"sources\":[{\"path\":\"/user/hand/right/input/trigger\",\"mode\":\"button\",\"inputs\":{\"click\":{\"output\":\"/actions/ingame/in/key.attack\"}}},"
                + "{\"path\":\"/user/hand/left/input/joystick\",\"mode\":\"joystick\",\"inputs\":{\"position\":{\"output\":\"/actions/ingame/in/move\"}}}]},"
                + "\"/actions/global\":{\"poses\":[{\"output\":\"/actions/global/in/lefthand\",\"path\":\"/user/hand/left/pose/raw\"}]}}}");
            Files.writeString(dir.resolve("m.json"), "{\"actions\":[{\"name\":\"/actions/ingame/in/key.attack\",\"type\":\"boolean\"},{\"name\":\"/actions/ingame/in/move\",\"type\":\"vector2\"},{\"name\":\"/actions/global/in/lefthand\",\"type\":\"pose\"},"
                + "{\"name\":\"/actions/global/in/lefthand_anim\",\"type\":\"skeleton\",\"skeleton\":\"/skeleton/hand/left\"}],"
                + "\"action_sets\":[{\"name\":\"/actions/ingame\"},{\"name\":\"/actions/global\"}],\"default_bindings\":[{\"controller_type\":\"oculus_touch\",\"binding_url\":\"b.json\"}]}");
            check(VRInput.VRInput_SetActionManifestPath(dir.resolve("m.json").toString()) == 0, "action manifest");
            LongBuffer set = s.mallocLong(1), act = s.mallocLong(1), pose = s.mallocLong(1);
            VRInput.VRInput_GetActionSetHandle("/actions/ingame", set);
            VRInput.VRInput_GetActionHandle("/actions/ingame/in/key.attack", act);
            VRInput.VRInput_GetActionHandle("/actions/global/in/lefthand", pose);
            VRActiveActionSet.Buffer sets = VRActiveActionSet.calloc(1, s); sets.get(0).ulActionSet(set.get(0));
            check(VRInput.VRInput_UpdateActionState(sets, VRActiveActionSet.SIZEOF) == 0, "UpdateActionState");
            InputDigitalActionData d = InputDigitalActionData.calloc(s);
            check(VRInput.VRInput_GetDigitalActionData(act.get(0), d, 0) == 0 && d.bActive(), "digital action (trigger=" + d.bState() + ")");
            InputPoseActionData pd = InputPoseActionData.calloc(s);
            check(VRInput.VRInput_GetPoseActionDataForNextFrame(pose.get(0), ETrackingUniverseOrigin_TrackingUniverseStanding, pd, 0) == 0 && pd.activeOrigin() == 1, "pose action -> left hand");
            InputOriginInfo oi = InputOriginInfo.calloc(s);
            check(VRInput.VRInput_GetOriginTrackedDeviceInfo(pd.activeOrigin(), oi) == 0 && oi.trackedDeviceIndex() == 1, "origin -> device 1");

            LongBuffer skel = s.mallocLong(1); IntBuffer nb = s.mallocInt(1);
            VRInput.VRInput_GetActionHandle("/actions/global/in/lefthand_anim", skel);
            check(VRInput.VRInput_GetBoneCount(skel.get(0), nb) == 0 && nb.get(0) == 31, "skeleton: 31 bones");
            VRBoneTransform.Buffer bones = VRBoneTransform.calloc(31, s);
            check(VRInput.VRInput_GetSkeletalBoneData(skel.get(0), EVRSkeletalTransformSpace_VRSkeletalTransformSpace_Model,
                  EVRSkeletalMotionRange_VRSkeletalMotionRange_WithController, bones) == 0 && bones.get(0).orientation().w() == 1, "skeletal bone data");
            VRSkeletalSummaryData sum = VRSkeletalSummaryData.calloc(s);
            check(VRInput.VRInput_GetSkeletalSummaryData(skel.get(0), EVRSummaryType_VRSummaryType_FromDevice, sum) == 0, "skeletal summary (index curl " + sum.flFingerCurl(1) + ")");
            check(OpenVR.VROverlay != null && VROverlay.VROverlay_CreateOverlay("siliconxr.test", "Test", s.mallocLong(1)) == 0, "overlay");
            check(VRSystem.VRSystem_GetHiddenAreaMesh(EVREye_Eye_Left, EHiddenAreaMeshType_k_eHiddenAreaMesh_Standard, HiddenAreaMesh.calloc(s)).unTriangleCount() > 0, "hidden area mesh");
            VREvent ev = VREvent.calloc(s);
            check(VRSystem.VRSystem_PollNextEvent(ev) && ev.eventType() == EVREventType_VREvent_TrackedDeviceActivated, "events");

            TrackedDevicePose.Buffer poses = TrackedDevicePose.calloc(64, s);
            long t0 = System.nanoTime();
            for (int i = 0; i < 10; i++) VRCompositor.VRCompositor_WaitGetPoses(poses, null);
            double hz = 9 / ((System.nanoTime() - t0) / 1e9);
            check(poses.get(0).bPoseIsValid() && hz > 40 && hz < 200, String.format("WaitGetPoses paced at %.0f Hz, head y=%.2f", hz, poses.get(0).mDeviceToAbsoluteTracking().m(7)));
            VR_ShutdownInternal();
            System.out.println("PASS");
        }
    }
}
