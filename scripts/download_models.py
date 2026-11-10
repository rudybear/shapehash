#!/usr/bin/env python3
"""Download test models from KhronosGroup/glTF-Sample-Assets into data/models/.

Downloaded files are untrusted data: they only ever land in data/models/ (no
code lives there) and are parsed, never executed.
"""
import os
import sys
import urllib.request

BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models"
GLB = [
    # blobs / organic
    "Duck", "Avocado", "BarramundiFish",
    # hard surface props
    "DamagedHelmet", "BoomBox", "WaterBottle", "Corset", "AntiqueCamera", "Lantern",
    # characters with limbs (non star-shaped)
    "Fox", "BrainStem", "CesiumMan", "RiggedFigure",
    # vehicles / multi-part
    "CesiumMilkTruck", "ToyCar",
    # furniture (thin legs)
    "SheenChair", "GlamVelvetSofa", "ChairDamaskPurplegold",
    # thin / awkward
    "SunglassesKhronos", "IridescenceLamp", "MaterialsVariantsShoe", "DiffuseTransmissionPlant",
    # colour-path tests
    "BoxVertexColors", "BoxTextured",
]
# models only shipped as multi-file glTF
MULTI = {
    "Suzanne": ["Suzanne.gltf", "Suzanne.bin", "Suzanne_BaseColor.png", "Suzanne_MetallicRoughness.png"],
}


def fetch(url, dst):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            data = r.read()
        with open(dst + ".part", "wb") as f:
            f.write(data)
        os.replace(dst + ".part", dst)
        print(f"ok   {os.path.basename(dst)} ({len(data)/1e6:.1f} MB)")
        return True
    except Exception as e:  # noqa
        print(f"FAIL {url}: {e}")
        return False


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "models")
    os.makedirs(root, exist_ok=True)
    ok = 0
    for name in GLB:
        ok += fetch(f"{BASE}/{name}/glTF-Binary/{name}.glb", os.path.join(root, f"{name}.glb"))
    for name, files in MULTI.items():
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        ok += all([fetch(f"{BASE}/{name}/glTF/{f}", os.path.join(d, f)) for f in files])
    print(f"{ok} models available")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
