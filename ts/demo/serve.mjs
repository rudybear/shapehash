// Tiny static server for the demo:  node demo/serve.mjs [port]
// Serves ts/demo/ and exposes the GLB files under ../data as /models/<name>.glb.
import { createReadStream, existsSync, readdirSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { dirname, extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const data = resolve(here, "..", "..", "data");
const port = Number(process.argv[2] ?? 8377);
const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".json": "application/json", ".glb": "model/gltf-binary", ".map": "application/json" };

const models = [];
for (const dir of ["models", "procedural"]) {
  const d = join(data, dir);
  if (!existsSync(d)) continue;
  for (const f of readdirSync(d).sort()) if (f.endsWith(".glb")) models.push({ name: f.slice(0, -4), url: `models/${dir}/${f}` });
}
models.sort((a, b) => a.name.localeCompare(b.name));

function inside(root, rel) {
  const p = normalize(join(root, rel));
  return p === root || p.startsWith(root + sep) ? p : null;
}

createServer((req, res) => {
  const url = decodeURIComponent(new URL(req.url, "http://x").pathname);
  if (url === "/models.json") {
    res.writeHead(200, { "content-type": types[".json"] });
    return res.end(JSON.stringify(models));
  }
  const file = url.startsWith("/models/") ? inside(data, url.slice(8)) : inside(here, url === "/" ? "index.html" : url.slice(1));
  if (!file || !existsSync(file) || !statSync(file).isFile()) {
    res.writeHead(404);
    return res.end("not found");
  }
  res.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
  createReadStream(file).pipe(res);
}).listen(port, "127.0.0.1", () => console.log(`ShapeHash demo: http://127.0.0.1:${port}/  (${models.length} models)`));
