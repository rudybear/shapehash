// Build script: type declarations (tsc) + bundles (esbuild).
import { build } from "esbuild";
import { execFileSync } from "node:child_process";
import { existsSync, readdirSync, writeFileSync } from "node:fs";
import { gzipSync } from "node:zlib";
import { readFileSync } from "node:fs";

const common = { bundle: true, format: "esm", target: "es2020", logLevel: "warning" };
const nodeExt = ["pngjs", "jpeg-js", "node:*"];

execFileSync("node_modules/.bin/tsc", ["-p", "."], { stdio: "inherit" });

const libs = [
  ["src/decode.ts", "dist/decode.js", {}],
  ["src/decode.ts", "dist/decode.min.js", { minify: true }],
  ["src/index.ts", "dist/index.js", {}],
  ["src/index.ts", "dist/index.min.js", { minify: true }],
  ["src/adapters/gltf.ts", "dist/gltf.js", {}],
  ["src/adapters/node.ts", "dist/node.js", { platform: "node", external: nodeExt }],
  ["src/cli.ts", "dist/cli.js", { platform: "node", external: nodeExt, banner: { js: "#!/usr/bin/env node" } }],
];
for (const [entry, outfile, extra] of libs) {
  if (existsSync(entry)) await build({ ...common, entryPoints: [entry], outfile, ...extra });
}
for (const f of readdirSync("test").filter((f) => f.endsWith(".test.ts"))) {
  await build({ ...common, entryPoints: [`test/${f}`], outfile: `dist/test/${f.replace(".ts", ".js")}`, platform: "node", external: [...nodeExt, "gltf-validator"] });
}
if (existsSync("demo/main.ts")) {
  await build({ ...common, entryPoints: ["demo/main.ts"], outfile: "demo/dist/main.js", minify: true });
}
const sizes = {};
for (const f of ["dist/decode.min.js", "dist/index.min.js"]) {
  if (!existsSync(f)) continue;
  const raw = readFileSync(f);
  sizes[f] = { bytes: raw.length, gzip: gzipSync(raw, { level: 9 }).length };
}
writeFileSync("dist/sizes.json", JSON.stringify(sizes, null, 1));
console.log("built;", JSON.stringify(sizes));
