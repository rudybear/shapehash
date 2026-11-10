import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";

export function sha(a: ArrayBufferView): string {
  return createHash("sha256").update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest("hex");
}
export function b64f32(s: string): Float32Array {
  const b = Buffer.from(s, "base64");
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
}
export function b64u8(s: string): Uint8Array {
  return new Uint8Array(Buffer.from(s, "base64"));
}
export function loadVectors(): any {
  return JSON.parse(readFileSync("test/vectors/vectors.json", "utf8"));
}
/** Deterministic PRNG for fuzz tests. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
