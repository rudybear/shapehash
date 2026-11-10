/**
 * glTF 2.0 / GLB -> MeshInput adapter (geometry + base colour only).
 * Platform neutral: image decoding and external-file loading are injected.
 */
import type { MeshInput } from "../encode/source.js";

export interface RgbaImage {
  width: number;
  height: number;
  /** RGBA, 8 bits per channel, row-major, top row first. */
  data: Uint8Array | Uint8ClampedArray;
}
export type ImageDecoder = (bytes: Uint8Array, mimeType: string) => RgbaImage | Promise<RgbaImage>;
export type UriLoader = (uri: string) => Uint8Array | Promise<Uint8Array>;

export interface GltfOptions {
  decodeImage: ImageDecoder;
  /** Needed only for .gltf files that reference external buffers / images. */
  loadUri?: UriLoader;
  /**
   * "gltf" (default): baseColorFactor and COLOR_0 are linear, textures sRGB, as the spec says.
   * "raw": multiply everything as if it were sRGB (what the Python reference harness does).
   */
  colorSpace?: "gltf" | "raw";
}

const COMPONENTS: Record<number, [number, (dv: DataView, o: number) => number, number]> = {
  5120: [1, (dv, o) => dv.getInt8(o), 127],
  5121: [1, (dv, o) => dv.getUint8(o), 255],
  5122: [2, (dv, o) => dv.getInt16(o, true), 32767],
  5123: [2, (dv, o) => dv.getUint16(o, true), 65535],
  5125: [4, (dv, o) => dv.getUint32(o, true), 4294967295],
  5126: [4, (dv, o) => dv.getFloat32(o, true), 1],
};
const TYPE_SIZE: Record<string, number> = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };

function fail(msg: string): never {
  throw new Error("glTF: " + msg);
}

function dataUri(uri: string): Uint8Array | null {
  const m = /^data:[^;,]*(;base64)?,(.*)$/s.exec(uri);
  if (!m) return null;
  if (m[1]) {
    const bin = atob(m[2]);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }
  return new TextEncoder().encode(decodeURIComponent(m[2]));
}

function mul(a: number[], b: number[]): number[] {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}
function nodeMatrix(n: any): number[] {
  if (n.matrix) return n.matrix;
  const t = n.translation ?? [0, 0, 0], q = n.rotation ?? [0, 0, 0, 1], s = n.scale ?? [1, 1, 1];
  const [x, y, z, w] = q;
  return [
    (1 - 2 * (y * y + z * z)) * s[0], 2 * (x * y + z * w) * s[0], 2 * (x * z - y * w) * s[0], 0,
    2 * (x * y - z * w) * s[1], (1 - 2 * (x * x + z * z)) * s[1], 2 * (y * z + x * w) * s[1], 0,
    2 * (x * z + y * w) * s[2], 2 * (y * z - x * w) * s[2], (1 - 2 * (x * x + y * y)) * s[2], 0,
    t[0], t[1], t[2], 1,
  ];
}
const toSrgb = (c: number): number => (c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055);
const toLinear = (c: number): number => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

/** Split a GLB container into its JSON and binary chunk. */
export function splitGlb(bytes: Uint8Array): { json: any; bin: Uint8Array | null } {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (bytes.length < 20 || dv.getUint32(0, true) !== 0x46546c67) fail("not a GLB file");
  let o = 12, json: any = null, bin: Uint8Array | null = null;
  while (o + 8 <= bytes.length) {
    const len = dv.getUint32(o, true), type = dv.getUint32(o + 4, true);
    if (o + 8 + len > bytes.length) fail("truncated chunk");
    const chunk = bytes.subarray(o + 8, o + 8 + len);
    if (type === 0x4e4f534a) json = JSON.parse(new TextDecoder().decode(chunk));
    else if (type === 0x004e4942 && !bin) bin = chunk;
    o += 8 + len + ((4 - (len % 4)) % 4);
  }
  if (!json) fail("missing JSON chunk");
  return { json, bin };
}

/** Load a glTF document (already parsed JSON + optional GLB binary chunk) as encoder input. */
export async function gltfToMesh(json: any, bin: Uint8Array | null, options: GltfOptions): Promise<MeshInput> {
  const raw = options.colorSpace === "raw";
  const buffers: Uint8Array[] = [];
  for (const [i, b] of (json.buffers ?? []).entries()) {
    if (b.uri === undefined) buffers.push(i === 0 && bin ? bin : fail("buffer without data"));
    else buffers.push(dataUri(b.uri) ?? (options.loadUri ? await options.loadUri(b.uri) : fail("external buffer needs loadUri")));
  }
  const view = (i: number): Uint8Array => {
    const v = json.bufferViews[i];
    const b = buffers[v.buffer];
    return b.subarray(v.byteOffset ?? 0, (v.byteOffset ?? 0) + v.byteLength);
  };
  const accessor = (i: number, forceNorm = false): { data: Float32Array; size: number; count: number } => {
    const a = json.accessors[i];
    const size = TYPE_SIZE[a.type] ?? fail("accessor type " + a.type);
    const comp = COMPONENTS[a.componentType] ?? fail("component type " + a.componentType);
    const out = new Float32Array(a.count * size);
    if (a.bufferView !== undefined) {
      const bv = json.bufferViews[a.bufferView];
      const bytes = view(a.bufferView);
      const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const stride = bv.byteStride || comp[0] * size;
      const base = a.byteOffset ?? 0;
      const norm = (a.normalized || forceNorm) && a.componentType !== 5126;
      for (let k = 0; k < a.count; k++)
        for (let c = 0; c < size; c++) {
          const v = comp[1](dv, base + k * stride + c * comp[0]);
          out[k * size + c] = norm ? Math.max(v / comp[2], -1) : v;
        }
    }
    return { data: out, size, count: a.count };
  };
  const images = new Map<number, RgbaImage>();
  const image = async (texIndex: number): Promise<RgbaImage | null> => {
    const tex = json.textures?.[texIndex];
    const src = tex?.source;
    if (src === undefined) return null;
    if (images.has(src)) return images.get(src)!;
    const im = json.images[src];
    let bytes: Uint8Array;
    if (im.bufferView !== undefined) bytes = view(im.bufferView);
    else if (im.uri !== undefined) bytes = dataUri(im.uri) ?? (options.loadUri ? await options.loadUri(im.uri) : fail("external image needs loadUri"));
    else return null;
    const mime = im.mimeType ?? (bytes[0] === 0x89 ? "image/png" : "image/jpeg");
    const dec = await options.decodeImage(bytes, mime);
    images.set(src, dec);
    return dec;
  };

  const pos: number[] = [], idx: number[] = [], uv: number[] = [], vcol: number[] = [];
  const triPrim: number[] = [];
  const prims: { factor: number[]; tex: RgbaImage | null }[] = [];
  const visit = async (ni: number, parent: number[], depth: number): Promise<void> => {
    if (depth > 64) fail("node hierarchy too deep");
    const n = json.nodes[ni];
    const m = mul(parent, nodeMatrix(n));
    if (n.mesh !== undefined)
      for (const p of json.meshes[n.mesh].primitives) {
        if ((p.mode ?? 4) !== 4 || p.attributes?.POSITION === undefined) continue;
        if (p.extensions?.KHR_draco_mesh_compression) fail("Draco-compressed primitives are not supported");
        const P = accessor(p.attributes.POSITION);
        const base = pos.length / 3;
        for (let k = 0; k < P.count; k++) {
          const x = P.data[k * 3], y = P.data[k * 3 + 1], z = P.data[k * 3 + 2];
          pos.push(m[0] * x + m[4] * y + m[8] * z + m[12], m[1] * x + m[5] * y + m[9] * z + m[13], m[2] * x + m[6] * y + m[10] * z + m[14]);
        }
        const mat = p.material !== undefined ? json.materials[p.material] : {};
        const pbr = mat.pbrMetallicRoughness ?? {};
        const sg = mat.extensions?.KHR_materials_pbrSpecularGlossiness;
        const factor: number[] = (pbr.baseColorFactor ?? sg?.diffuseFactor ?? [1, 1, 1, 1]).slice(0, 3);
        const texInfo = pbr.baseColorTexture ?? sg?.diffuseTexture;
        const tex = texInfo ? await image(texInfo.index) : null;
        const T = p.attributes[`TEXCOORD_${texInfo?.texCoord ?? 0}`];
        const U = T !== undefined ? accessor(T, true) : null;
        const C = p.attributes.COLOR_0 !== undefined ? accessor(p.attributes.COLOR_0, true) : null;
        for (let k = 0; k < P.count; k++) {
          uv.push(U ? U.data[k * 2] : 0, U ? U.data[k * 2 + 1] : 0);
          for (let c = 0; c < 3; c++) vcol.push(C ? C.data[k * C.size + c] : 1);
        }
        const before = idx.length;
        if (p.indices !== undefined) {
          const I = accessor(p.indices);
          for (let k = 0; k + 2 < I.count; k += 3) idx.push(base + I.data[k], base + I.data[k + 1], base + I.data[k + 2]);
        } else for (let k = 0; k + 2 < P.count; k += 3) idx.push(base + k, base + k + 1, base + k + 2);
        for (let k = before; k < idx.length; k += 3) triPrim.push(prims.length);
        prims.push({ factor, tex });
      }
    for (const c of n.children ?? []) await visit(c, m, depth + 1);
  };
  const ident = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
  const scene = json.scenes?.[json.scene ?? 0];
  let roots: number[] = scene?.nodes;
  if (!roots) {
    const child = new Set<number>();
    for (const n of json.nodes ?? []) for (const c of n.children ?? []) child.add(c);
    roots = (json.nodes ?? []).map((_: unknown, i: number) => i).filter((i: number) => !child.has(i));
  }
  for (const r of roots) await visit(r, ident, 0);
  if (idx.length === 0) fail("no triangle geometry");

  const positions = Float32Array.from(pos);
  const indices = Uint32Array.from(idx);
  const colorAt = (t: number, w0: number, w1: number, w2: number, out: Float32Array | number[]): void => {
    const a = indices[t * 3], b = indices[t * 3 + 1], c = indices[t * 3 + 2];
    const pr = prims[triPrim[t]];
    let r = w0 * vcol[a * 3] + w1 * vcol[b * 3] + w2 * vcol[c * 3];
    let g = w0 * vcol[a * 3 + 1] + w1 * vcol[b * 3 + 1] + w2 * vcol[c * 3 + 1];
    let bl = w0 * vcol[a * 3 + 2] + w1 * vcol[b * 3 + 2] + w2 * vcol[c * 3 + 2];
    r *= pr.factor[0]; g *= pr.factor[1]; bl *= pr.factor[2];
    if (pr.tex) {
      const u = w0 * uv[a * 2] + w1 * uv[b * 2] + w2 * uv[c * 2], v = w0 * uv[a * 2 + 1] + w1 * uv[b * 2 + 1] + w2 * uv[c * 2 + 1];
      const W = pr.tex.width, H = pr.tex.height;
      const x = Math.min(W - 1, Math.floor((u - Math.floor(u)) * W)), y = Math.min(H - 1, Math.floor((v - Math.floor(v)) * H));
      const o = (y * W + x) * 4;
      const d = pr.tex.data;
      if (raw) { r *= d[o] / 255; g *= d[o + 1] / 255; bl *= d[o + 2] / 255; }
      else { r *= toLinear(d[o] / 255); g *= toLinear(d[o + 1] / 255); bl *= toLinear(d[o + 2] / 255); }
    }
    if (raw) { out[0] = r; out[1] = g; out[2] = bl; }
    else { out[0] = toSrgb(Math.max(0, r)); out[1] = toSrgb(Math.max(0, g)); out[2] = toSrgb(Math.max(0, bl)); }
  };
  return { positions, indices, colorAt };
}

/** Parse a .glb (or .gltf JSON text) file's bytes into encoder input. */
export async function loadGltf(bytes: Uint8Array, options: GltfOptions): Promise<MeshInput> {
  if (bytes.length >= 4 && bytes[0] === 0x67 && bytes[1] === 0x6c && bytes[2] === 0x54 && bytes[3] === 0x46) {
    const { json, bin } = splitGlb(bytes);
    return gltfToMesh(json, bin, options);
  }
  return gltfToMesh(JSON.parse(new TextDecoder().decode(bytes)), null, options);
}
