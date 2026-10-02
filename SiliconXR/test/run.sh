#!/bin/sh
# Runs OpenVRTest against ../build with LWJGL from PrismLauncher (lwjgl 3.3.1 core) + lwjgl-openvr 3.3.2 (from a Vivecraft jar).
# Uses a temporary MacVR shm (1440x1584 per eye, 72 Hz), so it never touches a running MacVR.
set -e
cd "$(dirname "$0")"
P="$HOME/Library/Application Support/PrismLauncher"
JB="$P/java/java-runtime-gamma/bin"
VC=$(ls "$P"/instances/*/minecraft/mods/vivecraft-*.jar | head -1)
T=$(mktemp -d)
export VR4MAC_SHM="$T/shm"
python3 -c 'import struct,sys; f=open(sys.argv[1],"wb"); f.truncate(67112960); f.write(struct.pack("<IIIIf",0x3452564D,4,1440,1584,72))' "$VR4MAC_SHM"
unzip -oq "$VC" META-INF/jars/lwjgl-openvr-3.3.2.jar -d "$T"
CP="$P/libraries/org/lwjgl/lwjgl/3.3.1/lwjgl-3.3.1.jar:$P/libraries/org/lwjgl/lwjgl-natives-macos-arm64/3.3.1/lwjgl-natives-macos-arm64-3.3.1.jar:$T/META-INF/jars/lwjgl-openvr-3.3.2.jar"
"$JB/javac" -d "$T" -cp "$CP" OpenVRTest.java
"$JB/java" -Dorg.lwjgl.librarypath="$(cd ../build && pwd)" -cp "$CP:$T" OpenVRTest
rm -rf "$T"
