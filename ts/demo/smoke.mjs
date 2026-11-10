// Headless smoke test of the demo page (needs Chrome/Chromium on PATH; Node >= 22 for the built-in WebSocket).
//   node demo/smoke.mjs [outDir]
// Starts the demo server, drives the page through the DevTools protocol, checks that models load,
// encode and decode in the browser, that paste-a-hash works, and saves screenshots.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const out = resolve(process.argv[2] ?? join(here, "..", "..", "results", "v1", "demo"));
mkdirSync(out, { recursive: true });
const PORT = 8391, DBG = 9391;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const server = spawn(process.execPath, [join(here, "serve.mjs"), String(PORT)], { stdio: "ignore" });
const chromeBin = process.env.CHROME ?? "google-chrome";
const chrome = spawn(chromeBin, ["--headless=new", "--no-sandbox", `--user-data-dir=${mkdtempSync(join(tmpdir(), "shapehash-demo-"))}`, "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader", "--window-size=1300,900", `--remote-debugging-port=${DBG}`, "about:blank"], { stdio: "ignore" });
let failed = 0;
try {
  let target;
  for (let i = 0; i < 50 && !target; i++) {
    await sleep(200);
    try { target = (await (await fetch(`http://127.0.0.1:${DBG}/json`)).json()).find((t) => t.type === "page"); } catch { /* not up yet */ }
  }
  if (!target) throw new Error("could not reach Chrome");
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((r, j) => { ws.onopen = r; ws.onerror = j; });
  let id = 0;
  const pending = new Map();
  ws.onmessage = (ev) => { const m = JSON.parse(ev.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); } };
  const send = (method, params = {}) => new Promise((r) => { pending.set(++id, r); ws.send(JSON.stringify({ id, method, params })); });
  const evaluate = async (expression) => (await send("Runtime.evaluate", { expression, returnByValue: true })).result?.value;
  await send("Page.enable");
  const visit = async (query, name) => {
    await send("Page.navigate", { url: `http://127.0.0.1:${PORT}/?${query}` });
    let state;
    for (let i = 0; i < 300; i++) {
      await sleep(200);
      state = await evaluate("document.body.dataset.state");
      if (state === "ready" || state === "error") break;
    }
    await sleep(400); // let a frame render
    const stats = await evaluate("document.getElementById('stats').textContent");
    const hash = await evaluate("document.getElementById('hash').value");
    const shot = await send("Page.captureScreenshot", { format: "png" });
    writeFileSync(join(out, name + ".png"), Buffer.from(shot.data, "base64"));
    const ok = state === "ready" && hash.length > 10;
    if (!ok) failed++;
    console.log(`${ok ? "ok  " : "FAIL"} ${name}: ${stats}`);
    return hash;
  };
  const h = await visit("model=Duck&budget=64", "duck_64");
  await visit("model=SheenChair&budget=128", "sheenchair_128");
  await visit("model=Fox&budget=128&vox=cubes&method=vox", "fox_128_vox_cubes");
  await visit("model=DamagedHelmet&budget=256", "damagedhelmet_256");
  await visit(`hash=${h}`, "paste_hash");
  await send("Page.navigate", { url: `http://127.0.0.1:${PORT}/?hash=not*a*hash` });
  await sleep(1500);
  const err = await evaluate("document.getElementById('msg').textContent");
  console.log(`${err.startsWith("Cannot decode") ? "ok  " : "FAIL"} bad hash is reported: ${err}`);
  if (!err.startsWith("Cannot decode")) failed++;
  ws.close();
} catch (e) {
  console.error("smoke test error:", e.message);
  failed++;
} finally {
  chrome.kill();
  server.kill();
}
console.log(failed ? `${failed} check(s) failed` : `demo smoke test passed; screenshots in ${out}`);
process.exit(failed ? 1 : 0);
