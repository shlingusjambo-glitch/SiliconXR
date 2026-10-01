#!/bin/sh
# Builds SiliconXR: build/libsiliconxr_openxr.dylib (OpenXR runtime), build/libopenvr_api.dylib (OpenVR runtime)
# + build/liblwjgl_openvr.dylib (LWJGL JNI glue)
# for native macOS VR apps (OpenXR games; Vivecraft through OpenVR).
set -e
cd "$(dirname "$0")"
mkdir -p build
ARCH="-arch arm64 -arch x86_64 -mmacosx-version-min=11.0"
clang $ARCH -O2 -Wall -Wextra -Wno-unused-parameter -Wno-deprecated-declarations -fobjc-arc -fvisibility=hidden -dynamiclib \
    -install_name @rpath/libopenvr_api.dylib -o build/libopenvr_api.dylib vr4mac_openvr.m -framework Foundation -framework OpenGL
clang $ARCH -O2 -Wall -Wno-unused-parameter -fvisibility=hidden -dynamiclib \
    -install_name @rpath/liblwjgl_openvr.dylib -o build/liblwjgl_openvr.dylib lwjgl_openvr.c
clang $ARCH -O2 -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -fno-objc-arc -fvisibility=hidden -dynamiclib -Iinclude \
    -install_name @rpath/libsiliconxr_openxr.dylib -o build/libsiliconxr_openxr.dylib openxr/siliconxr_openxr.m -framework Foundation -framework Metal
codesign -s - -f build/*.dylib 2>/dev/null || true
# The Minecraft mod jar that carries the OpenVR natives is built in ../SiliconXR-Mod.
echo "built build/libopenvr_api.dylib build/liblwjgl_openvr.dylib build/libsiliconxr_openxr.dylib"
