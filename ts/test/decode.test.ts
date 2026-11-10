import assert from "node:assert/strict";
import { test } from "node:test";
import { decode, parse, type DecodedMesh } from "../src/decode.js";
import { b64f32, b64u8, loadVectors, sha } from "./util.js";

const V = loadVectors();
const TOL = V.float_tolerance as number;

function checkMesh(m: DecodedMesh, exp: any, exactFloats: boolean, label: string): { maxPos: number; maxCol: number } {
  assert.equal(m.positions.length / 3, exp.nverts, `${label}: vertex count`);
  assert.equal(m.indices.length / 3, exp.ntris, `${label}: triangle count`);
  assert.equal(sha(m.indices), exp.indices_sha, `${label}: indices`);
  assert.equal(m.flat, exp.flat, `${label}: flat flag`);
  let maxPos = 0, maxCol = 0;
  if (exactFloats) {
    assert.equal(sha(m.positions), exp.positions_sha, `${label}: positions must be bit-exact`);
    assert.equal(sha(m.colors), exp.colors_sha, `${label}: colours must be exact`);
    assert.equal(sha(m.normals), exp.normals_sha, `${label}: normals must be bit-exact`);
  } else {
    const pos = b64f32(exp.positions);
    for (let i = 0; i < pos.length; i++) maxPos = Math.max(maxPos, Math.abs(pos[i] - m.positions[i]));
    const col = b64u8(exp.colors);
    for (let i = 0; i < col.length; i++) maxCol = Math.max(maxCol, Math.abs(col[i] - m.colors[i]));
    assert.ok(maxPos <= TOL, `${label}: positions within ${TOL} (got ${maxPos})`);
    assert.ok(maxCol <= 1, `${label}: colours within 1/255 (got ${maxCol})`);
  }
  assert.equal(m.normals.length, m.positions.length);
  return { maxPos, maxCol };
}

test("vectors cover all three methods", () => {
  const methods = new Set(V.vectors.map((v: any) => v.method));
  assert.deepEqual([...methods].sort(), [0, 1, 2]);
});

test("integer content matches the Python reference exactly", () => {
  for (const v of V.vectors) {
    const p = parse(v.hash);
    assert.equal(p.method, v.method);
    assert.deepEqual(p.extentsQ, v.extents_q);
    if (p.vox) {
      assert.deepEqual(p.vox.dims, v.vox.dims);
      assert.equal(p.vox.f, v.vox.f);
      assert.equal(p.vox.colorMode, v.vox.color_mode);
      assert.equal(sha(p.vox.occ), v.vox.occ_sha, "occupancy grid");
      assert.equal(sha(p.vox.active), v.vox.active_sha, "active cells");
      assert.equal(sha(p.vox.cellColors), v.vox.cell_colors_sha, "cell colours");
    } else if (p.sh) {
      assert.deepEqual([p.sh.L, p.sh.Ly, p.sh.Lc], [v.sh.L, v.sh.Ly, v.sh.Lc]);
      assert.deepEqual(p.sh.centerQ, v.sh.center_q);
      p.sh.groups.forEach((g, i) => {
        assert.equal(g.dc, v.sh.groups[i].dc);
        assert.equal(g.scaleQ, v.sh.groups[i].scale_q);
        assert.deepEqual([...g.ac], v.sh.groups[i].ac);
      });
    } else if (p.mesh) {
      assert.equal(p.mesh.posBits, v.meshc.pos_bits);
      assert.equal(sha(p.mesh.posQ), v.meshc.pos_q_sha);
      assert.equal(sha(p.mesh.colQ), v.meshc.col_q_sha);
      assert.equal(sha(p.mesh.tris), v.meshc.tris_sha);
    }
  }
});

test("decoded meshes match the Python reference", () => {
  let worstSh = 0, worstShCol = 0, n = 0;
  for (const v of V.vectors) {
    const label = `${v.model}@${v.budget}`;
    if (v.method === 0) {
      checkMesh(decode(v.hash, { vox: "cubes" }), v.cubes, true, label + " vox cubes");
      checkMesh(decode(v.hash), v.smooth, true, label + " vox smooth");
    } else if (v.method === 2) {
      checkMesh(decode(v.hash), v.mesh, true, label + " mesh");
    } else {
      const r = checkMesh(decode(v.hash), v.mesh, false, label + " sh");
      worstSh = Math.max(worstSh, r.maxPos);
      worstShCol = Math.max(worstShCol, r.maxCol);
    }
    n++;
  }
  console.log(`  ${n} vectors; sh worst position error ${worstSh.toExponential(2)}, worst colour error ${worstShCol}/255`);
});
