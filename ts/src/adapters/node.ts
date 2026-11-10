/** Node.js helpers: read .glb / .gltf from disk with PNG and JPEG textures (pngjs, jpeg-js). */
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import jpeg from "jpeg-js";
import { PNG } from "pngjs";
import type { MeshInput } from "../encode/source.js";
import { loadGltf, type GltfOptions, type RgbaImage } from "./gltf.js";

export function decodeImageNode(bytes: Uint8Array, mimeType: string): RgbaImage {
  if (mimeType === "image/png" || (bytes[0] === 0x89 && bytes[1] === 0x50)) {
    const png = PNG.sync.read(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength));
    return { width: png.width, height: png.height, data: png.data };
  }
  if (mimeType === "image/jpeg" || (bytes[0] === 0xff && bytes[1] === 0xd8)) {
    const j = jpeg.decode(bytes, { useTArray: true, maxMemoryUsageInMB: 1024, maxResolutionInMP: 100 });
    return { width: j.width, height: j.height, data: j.data };
  }
  throw new Error(`unsupported texture format ${mimeType} (only PNG and JPEG)`);
}

/** Load a .glb or .gltf file (external buffers/images are resolved next to it). */
export async function loadGltfFile(path: string, options: Partial<GltfOptions> = {}): Promise<MeshInput> {
  const bytes = new Uint8Array(await readFile(path));
  const dir = dirname(path);
  return loadGltf(bytes, {
    decodeImage: decodeImageNode,
    loadUri: async (uri) => {
      if (/^[a-z]+:/i.test(uri) || uri.includes("..")) throw new Error("refusing to load non-local URI " + uri);
      return new Uint8Array(await readFile(join(dir, decodeURIComponent(uri))));
    },
    ...options,
  });
}
