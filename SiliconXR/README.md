# SiliconXR — VR runtime for native macOS games

SiliconXR is the Mac-native counterpart of [WineXR](https://github.com/shlingusjambo-glitch/WineXR). WineXR serves
Windows games running in Wine. SiliconXR serves games that run natively on macOS and have VR support or VR mods.
Both hand poses, input and rendered frames to [MacVR](https://github.com/shlingusjambo-glitch/MacVR) over shared
memory (`common/vr4mac.h`), and MacVR streams the frames to the Quest.

| Library | API | For |
|---|---|---|
| `libsiliconxr_openxr.dylib` | OpenXR 1.0 + `XR_KHR_metal_enable` | native Mac OpenXR apps (Metal) |
| `libopenvr_api.dylib` | OpenVR (`FnTable:` interfaces: IVRSystem_022, IVRCompositor_027/028, IVRInput_010, …), OpenGL textures | Vivecraft (Minecraft Java) via LWJGL |
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

- Poses and controller state come from MacVR. OpenXR uses the app's suggested bindings (Touch profile preferred). OpenVR follows the app's action manifest and its `oculus_touch` default bindings.
- OpenXR frames are copied by a Metal blit on the app's own queue into a ring of shared buffers, then published when the GPU finishes, so the game thread never waits. OpenVR `Submit` flips GL textures with a blit and reads them back.
- `xrWaitFrame` / `WaitGetPoses` pace the game to the headset's refresh rate.
- Session, space and action logic is the same as WineXR's.

## Tests

- `test/openxr_test.m`: OpenXR end to end against a temporary shm (Metal session, action, two eye frames). Build: `cd SiliconXR/test && clang -fno-objc-arc -I../include openxr_test.m -framework Metal -framework Foundation`.
- `test/submit_test.c`: OpenVR GL readback against a temporary shm.
- `test/run.sh`: drives the OpenVR libraries through LWJGL's bindings, as Vivecraft does. Needs MacVR running.
