/** Minimal GLB writer for decoded placeholders. Zero dependencies. */
import type { DecodedMesh } from "./decode.js";

export interface GlbOptions {
  /** Uniform scale applied to the normalised positions (default 1). */
  scale?: number;
  /** Translation applied after scaling (default 0,0,0). */
  translate?: [number, number, number];
}

function srgbToLinear(c: number): number {
  const x = c / 255;
  return x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4);
}

/**
 * GLB with POSITION, NORMAL, COLOR_0 (linear RGB float, as glTF requires) and
 * one rough, double-sided material.
 */
export function exportGLB(mesh: DecodedMesh, options: GlbOptions = {}): Uint8Array {
  const scale = options.scale ?? 1;
  const tr = options.translate ?? [0, 0, 0];
  const nv = mesh.positions.length / 3;
  const pos = new Float32Array(nv * 3);
  const min = [Infinity, Infinity, Infinity], max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nv * 3; i++) {
    const v = Math.fround(mesh.positions[i] * scale + tr[i % 3]);
    pos[i] = v;
    if (v < min[i % 3]) min[i % 3] = v;
    if (v > max[i % 3]) max[i % 3] = v;
  }
  const col = new Float32Array(nv * 3);
  for (let i = 0; i < nv * 3; i++) col[i] = srgbToLinear(mesh.colors[i]);
  const small = nv <= 65535;
  const idx = small ? Uint16Array.from(mesh.indices) : mesh.indices;
  const chunks: Uint8Array[] = [];
  const views: object[] = [];
  const accessors: object[] = [];
  let offset = 0;
  const add = (arr: Float32Array | Uint16Array | Uint32Array, target: number, componentType: number, type: string, count: number, extra: object = {}): number => {
    const raw = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
    const padded = new Uint8Array(Math.ceil(raw.length / 4) * 4);
    padded.set(raw);
    chunks.push(padded);
    views.push({ buffer: 0, byteOffset: offset, byteLength: raw.length, target });
    accessors.push({ bufferView: views.length - 1, componentType, count, type, ...extra });
    offset += padded.length;
    return accessors.length - 1;
  };
  const attributes: Record<string, number> = {
    POSITION: add(pos, 34962, 5126, "VEC3", nv, { min, max }),
    NORMAL: add(Float32Array.from(mesh.normals), 34962, 5126, "VEC3", nv),
    COLOR_0: add(col, 34962, 5126, "VEC3", nv),
  };
  const indices = add(idx, 34963, small ? 5123 : 5125, "SCALAR", idx.length);
  const gltf = {
    asset: { version: "2.0", generator: "shapehash" },
    scene: 0,
    scenes: [{ nodes: [0] }],
    nodes: [{ mesh: 0, name: `shapehash-${mesh.method}` }],
    meshes: [{ primitives: [{ attributes, indices, material: 0, mode: 4 }] }],
    materials: [{ pbrMetallicRoughness: { baseColorFactor: [1, 1, 1, 1], metallicFactor: 0, roughnessFactor: 0.9 }, doubleSided: true }],
    buffers: [{ byteLength: offset }],
    bufferViews: views,
    accessors,
  };
  let json = JSON.stringify(gltf);
  while (json.length % 4) json += " ";
  const jsonBytes = new TextEncoder().encode(json);
  const total = 12 + 8 + jsonBytes.length + 8 + offset;
  const out = new Uint8Array(total);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, 0x46546c67, true);
  dv.setUint32(4, 2, true);
  dv.setUint32(8, total, true);
  dv.setUint32(12, jsonBytes.length, true);
  dv.setUint32(16, 0x4e4f534a, true);
  out.set(jsonBytes, 20);
  let o = 20 + jsonBytes.length;
  dv.setUint32(o, offset, true);
  dv.setUint32(o + 4, 0x004e4942, true);
  o += 8;
  for (const c of chunks) {
    out.set(c, o);
    o += c.length;
  }
  return out;
}
