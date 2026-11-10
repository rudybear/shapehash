import assert from "node:assert/strict";
import { test } from "node:test";
// @ts-ignore - gltf-validator ships no type declarations
import validator from "gltf-validator";
import { loadGltf, splitGlb } from "../src/adapters/gltf.js";
import { decodeImageNode } from "../src/adapters/node.js";
import { decode } from "../src/decode.js";
import { encode } from "../src/encode/index.js";
import { exportGLB } from "../src/glb.js";
import { loadVectors } from "./util.js";

const V = loadVectors();

test("exported GLBs pass the Khronos glTF validator with no errors or warnings", async () => {
  const picks: any[] = [];
  for (const method of [0, 1, 2]) picks.push(...V.vectors.filter((v: any) => v.method === method).slice(0, 4));
  let checked = 0;
  for (const v of picks)
    for (const vox of v.method === 0 ? (["smooth", "cubes"] as const) : (["smooth"] as const)) {
      const mesh = decode(v.hash, { vox });
      const glb = exportGLB(mesh);
      const report = await validator.validateBytes(glb);
      const issues = report.issues;
      assert.equal(issues.numErrors, 0, `${v.model}@${v.budget} ${mesh.method}: ${JSON.stringify(issues.messages.slice(0, 3))}`);
      assert.equal(issues.numWarnings, 0, `${v.model}@${v.budget} ${mesh.method}: ${JSON.stringify(issues.messages.slice(0, 3))}`);
      assert.equal(report.info.totalTriangleCount, mesh.indices.length / 3);
      assert.equal(report.info.totalVertexCount, mesh.positions.length / 3);
      checked++;
    }
  console.log(`  ${checked} GLBs validated`);
});

test("GLB export -> glTF adapter -> encode round trip", async () => {
  const v = V.vectors.find((x: any) => x.method === 2 && x.budget >= 128);
  const mesh = decode(v.hash);
  const glb = exportGLB(mesh, { scale: 3, translate: [10, 0, -2] });
  const { json, bin } = splitGlb(glb);
  assert.equal(json.asset.version, "2.0");
  assert.ok(bin && bin.length > 0);
  const input = await loadGltf(glb, { decodeImage: decodeImageNode });
  assert.equal(input.positions.length, mesh.positions.length);
  // COLOR_0 is stored linear; the adapter must hand back sRGB close to the decoded colours
  const out = [0, 0, 0];
  input.colorAt!(0, 1, 0, 0, out);
  const i0 = mesh.indices[0] * 3;
  for (let d = 0; d < 3; d++) assert.ok(Math.abs(out[d] * 255 - mesh.colors[i0 + d]) < 1.5, `colour channel ${d}`);
  const again = encode(input, { budget: 128, samples: 150_000 });
  assert.ok(again.bytes <= 128);
  assert.ok(Math.abs(again.scale - 1 / 3) < 0.05, "normalisation undoes the export scale");
});
