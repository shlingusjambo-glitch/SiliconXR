# SiliconXR — VR runtime for native macOS games

SiliconXR is the Mac-native counterpart of [WineXR](https://github.com/shlingusjambo-glitch/WineXR). WineXR serves
Windows games running in Wine. SiliconXR serves games that run natively on macOS and have VR support or VR mods.
Both hand poses, input and rendered frames to [MacVR](https://github.com/shlingusjambo-glitch/MacVR) over shared
memory (`common/vr4mac.h`), and MacVR streams the frames to the Quest.

| Library | API | For |
|---|---|---|
| `libsiliconxr_openxr.dylib` | OpenXR 1.0 + `XR_KHR_metal_enable` | native Mac OpenXR apps (Metal) |
| `libopenvr_api.dylib` | OpenVR (`FnTable:` interfaces: IVRSystem_022, IVRCompositor_027/028, IVRInput_010, IVROverlay_026, …), OpenGL textures | Vivecraft (Minecraft Java) via LWJGL |
| `liblwjgl_openvr.dylib` | LWJGL JNI glue | LWJGL ships this for Intel Macs only. It also registers the `JNI.call*` trampolines that Apple Silicon LWJGL omits (`jni_calls.h`, generated from `jni_sigs.txt`). |

## Install (automatic)

The MacVR app bundles SiliconXR and sets it up on launch:

- **OpenXR:** writes `~/.config/openxr/1/active_runtime.json` pointing at the bundled runtime, unless another runtime is already registered there. Manual: `XR_RUNTIME_JSON=/path/to/active_runtime.json`.
- **Vivecraft:** drops the [SiliconXR mod](https://github.com/shlingusjambo-glitch/SiliconXR-Mod) (`siliconxr.jar`) next to every Vivecraft jar it finds (Prism, MultiMC, Modrinth, CurseForge, official launcher). Fabric, Quilt, Forge and NeoForge are supported.

Start MacVR before you enable VR in a game.

## Build

```sh
SiliconXR/build.sh   # -> SiliconXR/build/*.dylib (universal arm64 + x86_64)
```

## How it works

- Poses and controller state come from MacVR. OpenXR uses the app's suggested bindings: Touch is preferred, and the Touch Plus/Pro, Index, HP, Windows Mixed Reality, Vive and simple profiles are remapped from the Touch controllers. OpenVR follows the app's action manifest and its `oculus_touch` default bindings.
- Hand tracking: when you put a controller down, MacVR sends that hand's 26 joints. OpenXR serves them through `XR_EXT_hand_tracking` and `XR_FB_hand_tracking_aim`. `XR_EXT_hand_interaction` bindings take over that hand until you pick the controller up again, and an interaction-profile-changed event is sent each time. OpenVR skeletal input gives the 31 SteamVR hand bones and finger curls from the joints, or from the trigger, grip and touch sensors while you hold a controller.
- OpenXR frames are copied by a Metal blit on the app's own queue into a ring of shared buffers, then published when the GPU finishes, so the game thread never waits. Quad, cylinder and equirect2 layers, and color scale/bias, are composited on the GPU first. This also works without a projection layer.
- OpenVR `Submit` flips GL textures with a blit and reads them into pixel buffer objects. A worker thread with a shared GL context publishes them once the GPU is done, so the render thread never waits either. Legacy GL 2.1 contexts read back synchronously.
- Both runtimes report a lens mask (`xrGetVisibilityMaskKHR`, `GetHiddenAreaMesh`) so games skip the corners nobody sees.
- `xrWaitFrame` / `WaitGetPoses` pace the game to the headset's refresh rate.
- Session, space and action logic is the same as WineXR's.
- Graphics: Metal (OpenXR) and OpenGL (OpenVR) only. OpenXR has no macOS OpenGL binding, and `XR_KHR_vulkan_enable2` through MoltenVK is not implemented.

## Tests

- `test/openxr_test.m`: OpenXR end to end against a temporary shm. It covers the Metal session, projection, quad, cylinder and equirect2 frames, the visibility mask, hand joints and FB aim, hand-interaction and Index profile switching, haptics on both hands, user presence and `xrLocateSpacesKHR`. Build: `cd SiliconXR/test && clang -fno-objc-arc -I../include openxr_test.m -framework Metal -framework Foundation`.
- `test/submit_test.c`: OpenVR against a temporary shm. It covers asynchronous (GL 3.2) and synchronous (legacy GL) readback, fades, skeletal input (SteamVR mirror rules, all transform spaces, curls), events, device properties, the hidden area mesh, overlays, chaperone and haptics. Build: `cd SiliconXR/test && clang submit_test.c -framework OpenGL`.
- `test/run.sh`: drives the OpenVR libraries through LWJGL's bindings, as Vivecraft does, including the skeletal, overlay and event calls. It uses a temporary shm, so MacVR does not need to be running.
