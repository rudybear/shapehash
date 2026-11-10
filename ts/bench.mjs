// Decode-time benchmark over the cross-validation vectors:  node bench.mjs   (after `npm run build`)
import { readFileSync, writeFileSync } from "node:fs";
import { decode } from "./dist/decode.js";

const V = JSON.parse(readFileSync("test/vectors/vectors.json", "utf8")).vectors;
const names = ["vox", "sh", "mesh"];
const median = (a) => a.slice().sort((x, y) => x - y)[a.length >> 1];
const rows = [];
for (const budget of [64, 128, 256]) {
  for (const [method, mode] of [[0, "smooth"], [0, "cubes"], [1, "smooth"], [2, "smooth"]]) {
    const hs = V.filter((v) => v.method === method && v.budget === budget).map((v) => v.hash);
    if (!hs.length) continue;
    const ms = [], tris = [];
    for (const h of hs) {
      for (let i = 0; i < 15; i++) decode(h, { vox: mode }); // warm up
      const t = [];
      for (let i = 0; i < 40; i++) {
        const t0 = performance.now();
        const m = decode(h, { vox: mode });
        t.push(performance.now() - t0);
        if (i === 0) tris.push(m.indices.length / 3);
      }
      ms.push(median(t));
    }
    rows.push({ budget, method: names[method] + (method === 0 ? ` (${mode})` : ""), hashes: hs.length, median_ms: +median(ms).toFixed(3), max_ms: +Math.max(...ms).toFixed(3), median_triangles: median(tris) });
  }
}
console.table(rows);
const sizes = JSON.parse(readFileSync("dist/sizes.json", "utf8"));
console.log(sizes);
writeFileSync("../results/v1/ts_decode_bench.json", JSON.stringify({ node: process.version, rows, sizes }, null, 1));
