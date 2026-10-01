#!/bin/sh
# Runs OpenVRTest against ../build with LWJGL from PrismLauncher (lwjgl 3.3.1 core) + lwjgl-openvr 3.3.2 (from a Vivecraft jar).
set -e
cd "$(dirname "$0")"
P="$HOME/Library/Application Support/PrismLauncher"
JB="$P/java/java-runtime-gamma/bin"
VC=$(ls "$P"/instances/*/minecraft/mods/vivecraft-*.jar | head -1)
T=$(mktemp -d)
unzip -oq "$VC" META-INF/jars/lwjgl-openvr-3.3.2.jar -d "$T"
CP="$P/libraries/org/lwjgl/lwjgl/3.3.1/lwjgl-3.3.1.jar:$P/libraries/org/lwjgl/lwjgl-natives-macos-arm64/3.3.1/lwjgl-natives-macos-arm64-3.3.1.jar:$T/META-INF/jars/lwjgl-openvr-3.3.2.jar"
"$JB/javac" -d "$T" -cp "$CP" OpenVRTest.java
"$JB/java" -Dorg.lwjgl.librarypath="$(cd ../build && pwd)" -cp "$CP:$T" OpenVRTest
