import assert from "node:assert/strict";
import { test } from "node:test";
import { decode, ShapeHashError } from "../src/decode.js";
import { BitWriter, toBase64Url } from "../src/format.js";
import { loadVectors, mulberry32 } from "./util.js";

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const V = loadVectors();
const MAX_VERTS = 2_000_000; // maxTriangles defaults to 500 000 -> at most 4 vertices per 2 triangles
const MAX_MS = 3000;

/** decode must either throw ShapeHashError or return a sane, bounded mesh — quickly. */
function tryDecode(s: string): "ok" | "rejected" {
  for (const vox of ["smooth", "cubes"] as const) {
    const t = performance.now();
    try {
      const m = decode(s, { vox });
      assert.ok(m.positions.length / 3 <= MAX_VERTS, "vertex count bounded");
      assert.equal(m.colors.length, m.positions.length);
      assert.equal(m.normals.length, m.positions.length);
      const nv = m.positions.length / 3;
      for (let i = 0; i < m.indices.length; i++) assert.ok(m.indices[i] < nv, "index in range");
      for (let i = 0; i < m.positions.length; i++) assert.ok(Number.isFinite(m.positions[i]), "finite position");
    } catch (e) {
      if (!(e instanceof ShapeHashError)) throw e;
      return "rejected";
    }
    assert.ok(performance.now() - t < MAX_MS, "decode time bounded");
  }
  return "ok";
}

test("rejects non-strings, bad characters and bad lengths", () => {
  for (const bad of [undefined, null, 42, {}, []]) assert.throws(() => decode(bad as any), ShapeHashError);
  for (const bad of ["", "A", "AAAAA", "ab+/", "ab==", "héllo", "ab cd", "a".repeat(5000)]) assert.throws(() => decode(bad), ShapeHashError, JSON.stringify(bad.slice(0, 10)));
});

test("rejects wrong version, unknown method and reserved bits", () => {
  const mk = (version: number, method: number, param: number): string => {
    const w = new BitWriter();
    w.write(version, 3); w.write(method, 2); w.write(param, 3);
    for (let i = 0; i < 40; i++) w.write(0xa5, 8);
    return toBase64Url(w.toBytes());
  };
  assert.throws(() => decode(mk(0, 0, 0)), ShapeHashError);
  assert.throws(() => decode(mk(2, 0, 0)), ShapeHashError);
  assert.throws(() => decode(mk(1, 3, 0)), ShapeHashError);
  assert.throws(() => decode(mk(1, 0, 7)), ShapeHashError);
  assert.throws(() => decode(mk(1, 1, 1)), ShapeHashError);
  assert.throws(() => decode(mk(1, 2, 4)), ShapeHashError);
});

test("truncations of valid hashes never crash", () => {
  let ok = 0, rej = 0;
  for (const v of V.vectors.slice(0, 30)) {
    const h: string = v.hash;
    for (let n = 0; n < h.length; n += Math.max(1, Math.floor(h.length / 12))) tryDecode(h.slice(0, n)) === "ok" ? ok++ : rej++;
  }
  console.log(`  truncations: ${ok} decoded, ${rej} rejected`);
  // sh and mesh payloads are plain bit streams: any truncation must be rejected
  for (const v of V.vectors.filter((v: any) => v.method !== 0).slice(0, 10)) assert.equal(tryDecode(v.hash.slice(0, v.hash.length - 2)), "rejected");
});

test("bit flips and character substitutions of valid hashes never crash", () => {
  const rnd = mulberry32(1234);
  let ok = 0, rej = 0;
  for (const v of V.vectors.filter((v: any) => v.budget <= 128).slice(0, 24)) {
    for (let k = 0; k < 25; k++) {
      const a = [...(v.hash as string)];
      const edits = 1 + Math.floor(rnd() * 3);
      for (let e = 0; e < edits; e++) a[Math.floor(rnd() * a.length)] = ALPHABET[Math.floor(rnd() * 64)];
      tryDecode(a.join("")) === "ok" ? ok++ : rej++;
    }
  }
  console.log(`  mutations: ${ok} decoded, ${rej} rejected`);
});

test("random strings of every length class never crash, hang or over-allocate", () => {
  const rnd = mulberry32(99);
  let ok = 0, rej = 0;
  for (let k = 0; k < 600; k++) {
    const len = k < 300 ? 2 + Math.floor(rnd() * 200) : 2 + Math.floor(rnd() * 2700);
    let s = "";
    for (let i = 0; i < len; i++) s += ALPHABET[Math.floor(rnd() * 64)];
    // force a valid header on two thirds of them so the method parsers are actually reached
    if (k % 3) s = "IJKLMNOPQRSTUVWXYZabcdef"[Math.floor(rnd() * 24)] + s.slice(1);
    tryDecode(s) === "ok" ? ok++ : rej++;
  }
  console.log(`  random: ${ok} decoded, ${rej} rejected`);
});

test("worst-case vox header (64^3 grid) stays bounded", () => {
  const w = new BitWriter();
  w.write(1, 3); w.write(0, 2); w.write(0, 3);
  for (let i = 0; i < 3; i++) w.write(63, 6);
  for (let i = 0; i < 3; i++) w.write(63, 6);
  w.write(0, 4);
  const rnd = mulberry32(5);
  for (let i = 0; i < 1500; i++) w.write(Math.floor(rnd() * 256), 8);
  const t = performance.now();
  tryDecode(toBase64Url(w.toBytes()));
  console.log(`  64^3 random grid, both decoders: ${(performance.now() - t).toFixed(0)} ms`);
});
