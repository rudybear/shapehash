import assert from "node:assert/strict";
import { test } from "node:test";
import { decode, parse } from "../src/decode.js";
import { encode, encodeSource, Source } from "../src/encode/index.js";
import { encodeStrips } from "../src/encode/mesh.js";
import { choleskySolve } from "../src/encode/sh.js";
import { RangeDecoder, RangeEncoder, fromBase64Url, icosphere, toBase64Url } from "../src/format.js";
import { mulberry32 } from "./util.js";

/** Two-coloured dumbbell-ish test shape: sphere + offset smaller sphere. */
function testShape(): { positions: Float32Array; indices: Uint32Array; colors: Uint8Array } {
  const { dirs, faces } = icosphere(3);
  const n = dirs.length / 3;
  const positions = new Float32Array(n * 6), colors = new Uint8Array(n * 6), indices = new Uint32Array(faces.length * 2);
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < 3; d++) {
      positions[i * 3 + d] = dirs[i * 3 + d] * 10 + (d === 0 ? 100 : 0);
      positions[(n + i) * 3 + d] = dirs[i * 3 + d] * 5 + (d === 0 ? 113 : d === 1 ? 4 : 0);
    }
    colors.set([220, 40, 30], i * 3);
    colors.set([30, 60, 200], (n + i) * 3);
  }
  indices.set(faces);
  for (let i = 0; i < faces.length; i++) indices[faces.length + i] = faces[i] + n;
  return { positions, indices, colors };
}

test("base64url round trip", () => {
  const rnd = mulberry32(1);
  for (let n = 0; n < 70; n++) {
    const b = Uint8Array.from({ length: n }, () => Math.floor(rnd() * 256));
    assert.deepEqual(fromBase64Url(toBase64Url(b)), b);
  }
  assert.equal(toBase64Url(Uint8Array.from([0xfb, 0xff])), "-_8");
});

test("range coder round trip (adaptive + bypass, minimal flush)", () => {
  const rnd = mulberry32(7);
  for (let trial = 0; trial < 300; trial++) {
    const n = Math.floor(rnd() * 500);
    const bias = [0.03, 0.5, 0.95][trial % 3];
    const bits = Array.from({ length: n }, () => (rnd() < bias ? 1 : 0));
    const enc = new RangeEncoder();
    let pr = new Uint16Array(4).fill(1024);
    bits.forEach((b, i) => (i % 7 === 3 ? enc.bypass(b, 1) : enc.encode(pr, i % 4, b)));
    enc.bypass(0x2a5, 10);
    const data = enc.finish();
    const dec = new RangeDecoder(data, 0);
    pr = new Uint16Array(4).fill(1024);
    bits.forEach((b, i) => assert.equal(i % 7 === 3 ? dec.bypass(1) : dec.decode(pr, i % 4), b));
    assert.equal(dec.bypass(10), 0x2a5);
  }
});

test("cholesky solver", () => {
  const A = Float64Array.from([4, 0, 0, 2, 5, 0, 1, 3, 6]); // lower triangle used
  const full = [4, 2, 1, 2, 5, 3, 1, 3, 6];
  const x = choleskySolve(A, Float64Array.from([1, 2, 3]), 3);
  for (let i = 0; i < 3; i++) assert.ok(Math.abs(full[i * 3] * x[0] + full[i * 3 + 1] * x[1] + full[i * 3 + 2] * x[2] - (i + 1)) < 1e-12);
});

test("strip coder visits every triangle once", () => {
  const { faces } = icosphere(1);
  const ops = encodeStrips(faces);
  assert.equal(ops.length, faces.length / 3);
  assert.ok(ops.filter((o) => o.restart).length < faces.length / 3 / 4, "strips should be long on a closed manifold");
});

test("encode -> decode round trip for every method and budget", () => {
  const src = new Source(testShape(), 200_000);
  for (const budget of [32, 64, 128, 256]) {
    const picked = encodeSource(src, { budget });
    assert.ok(picked.bytes <= budget, "auto fits the budget");
    for (const method of ["vox", "sh", "mesh"] as const) {
      let r;
      try {
        r = encodeSource(src, { budget, method });
      } catch (e) {
        assert.ok(e instanceof RangeError && budget <= 32, `${method}@${budget}: ${e}`);
        continue;
      }
      assert.ok(r.bytes <= budget, `${method}@${budget} fits`);
      assert.ok(r.hash.length <= Math.ceil((budget * 4) / 3));
      assert.equal(parse(r.hash).method, ["vox", "sh", "mesh"].indexOf(method));
      const m = decode(r.hash);
      assert.equal(m.method, method);
      const s = src.score(m, true);
      // a 64-byte mesh has ~12 vertices: only require it to be in the right ballpark
      const floor = budget >= 128 ? 0.6 : method === "mesh" || budget < 64 ? 0.3 : 0.6;
      assert.ok(s > floor, `${method}@${budget}: selection score ${s.toFixed(3)} too low`);
      // the bounding box is preserved
      let max = 0;
      for (let i = 0; i < m.positions.length; i++) max = Math.max(max, Math.abs(m.positions[i]));
      assert.ok(max > 0.6 && max < 1.6, `${method}@${budget}: extent ${max}`);
    }
    assert.ok(Object.keys(picked.scores).length >= 2);
    assert.equal(picked.scores[picked.method], Math.max(...(Object.values(picked.scores) as number[])));
  }
});

test("colours survive: the big sphere is red, the small one blue", () => {
  const r = encode(testShape(), { budget: 128, samples: 200_000 });
  const m = decode(r.hash);
  let red = 0, blue = 0;
  for (let i = 0; i < m.positions.length; i += 3) {
    const x = m.positions[i];
    if (x < -0.3 && m.colors[i] > m.colors[i + 2] + 60) red++;
    if (x > 0.7 && m.colors[i + 2] > m.colors[i] + 60) blue++;
  }
  assert.ok(red > 5 && blue > 2, `red ${red}, blue ${blue} (${r.method})`);
  // normalisation info maps the input frame to the decoded frame
  assert.ok(Math.abs((100 - 10 - r.center[0]) * r.scale + 1) < 1e-6);
});

test("encoder rejects bad input cleanly", () => {
  assert.throws(() => encode({ positions: new Float32Array(0) }));
  assert.throws(() => encode({ positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: [0, 1, 5] }));
  assert.throws(() => encode({ positions: new Float32Array([0, 0, 0, 1, 0, 0, NaN, 1, 0]) }));
  assert.throws(() => encode(testShape(), { budget: 4 }), RangeError);
  assert.throws(() => encode(testShape(), { budget: 5000 }), RangeError);
});
