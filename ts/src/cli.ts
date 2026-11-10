/**
 * shapehash CLI
 *   shapehash encode <model.glb|.gltf> [--budget 128] [--method auto|vox|sh|mesh] [--raw-colors] [--json]
 *   shapehash decode <hash> <out.glb> [--cubes]
 *   shapehash info <hash>
 *   shapehash batch <out.json> --budgets 64,128,256 [--raw-colors] <model>...
 */
import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { basename } from "node:path";
import { loadGltfFile } from "./adapters/node.js";
import { decode, parse } from "./decode.js";
import { Source, encodeSource } from "./encode/index.js";
import { METHOD_NAMES } from "./format.js";
import { exportGLB } from "./glb.js";

const sha = (a: ArrayBufferView): string => createHash("sha256").update(new Uint8Array(a.buffer, a.byteOffset, a.byteLength)).digest("hex");
/** Digests of the TypeScript decode of a hash, for cross-checking against the Python reference. */
function digests(hash: string): object {
  const out: Record<string, object> = {};
  for (const vox of ["smooth", "cubes"] as const) {
    const m = decode(hash, { vox });
    out[vox] = { nverts: m.positions.length / 3, ntris: m.indices.length / 3, positions_sha: sha(m.positions), indices_sha: sha(m.indices), colors_sha: sha(m.colors), normals_sha: sha(m.normals) };
  }
  return out;
}

function usage(): never {
  console.error("usage:\n  shapehash encode <model.glb|.gltf> [--budget 128] [--method auto|vox|sh|mesh] [--raw-colors] [--json]\n  shapehash decode <hash> <out.glb> [--cubes]\n  shapehash info <hash>\n  shapehash batch <out.json> --budgets 64,128,256 [--raw-colors] <model>...");
  process.exit(2);
}

async function main(): Promise<void> {
  const argv = process.argv.slice(2);
  const flags = new Map<string, string>();
  const pos: string[] = [];
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--budget" || a === "--method" || a === "--budgets") flags.set(a, argv[++i] ?? usage());
    else if (a.startsWith("--")) flags.set(a, "1");
    else pos.push(a);
  }
  const cmd = pos.shift();
  const colorSpace = flags.has("--raw-colors") ? "raw" : "gltf";
  if (cmd === "encode" && pos.length === 1) {
    const t0 = performance.now();
    const src = new Source(await loadGltfFile(pos[0], { colorSpace }));
    const r = encodeSource(src, { budget: Number(flags.get("--budget") ?? 128), method: (flags.get("--method") ?? "auto") as "auto" });
    if (flags.has("--json")) console.log(JSON.stringify({ hash: r.hash, method: r.method, bytes: r.bytes, chars: r.hash.length, scores: r.scores, center: r.center, scale: r.scale, ms: Math.round(performance.now() - t0) }));
    else {
      console.log(r.hash);
      console.error(`# method=${r.method} payload=${r.bytes} bytes, ${r.hash.length} chars, ${Math.round(performance.now() - t0)} ms`);
    }
  } else if (cmd === "decode" && pos.length === 2) {
    const m = decode(pos[0], { vox: flags.has("--cubes") ? "cubes" : "smooth" });
    await writeFile(pos[1], exportGLB(m));
    console.error(`wrote ${pos[1]}: method=${m.method}, ${m.positions.length / 3} vertices, ${m.indices.length / 3} triangles`);
  } else if (cmd === "info" && pos.length === 1) {
    const p = parse(pos[0]);
    const m = decode(pos[0]);
    console.log(JSON.stringify({ method: METHOD_NAMES[p.method], bytes: Math.floor((pos[0].length * 3) / 4), extents: m.extents, vertices: m.positions.length / 3, triangles: m.indices.length / 3,
      vox: p.vox && { dims: p.vox.dims, colorCellSize: p.vox.f, colorMode: p.vox.colorMode }, sh: p.sh && { L: p.sh.L, Ly: p.sh.Ly, Lc: p.sh.Lc }, mesh: p.mesh && { posBits: p.mesh.posBits } }));
  } else if (cmd === "batch" && pos.length >= 2) {
    const out = pos.shift()!;
    const budgets = (flags.get("--budgets") ?? "64,128,256").split(",").map(Number);
    const rows: object[] = [];
    for (const file of pos) {
      const model = basename(file).replace(/\.(glb|gltf)$/i, "");
      try {
        const t0 = performance.now();
        const src = new Source(await loadGltfFile(file, { colorSpace }));
        const prep = performance.now() - t0;
        for (const budget of budgets) {
          const t1 = performance.now();
          const r = encodeSource(src, { budget });
          const dig: Record<string, object> = {};
          for (const [k, h] of Object.entries(r.candidates)) dig[k] = digests(h);
          rows.push({ model, budget, hash: r.hash, method: r.method, bytes: r.bytes, scores: r.scores, candidates: r.candidates, digests: dig, enc_ms: Math.round(performance.now() - t1), prep_ms: Math.round(prep) });
          console.error(`${model} ${budget}B -> ${r.method} ${r.bytes}B in ${Math.round(performance.now() - t1)} ms`);
        }
      } catch (e) {
        rows.push({ model, error: String(e) });
        console.error(`${model}: ${e}`);
      }
    }
    await writeFile(out, JSON.stringify(rows, null, 1));
  } else usage();
}

main().catch((e) => {
  console.error(String(e?.message ?? e));
  process.exit(1);
});
