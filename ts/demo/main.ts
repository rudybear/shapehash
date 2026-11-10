// ShapeHash demo: original GLB next to the decoded placeholder.
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { loadGltf, type RgbaImage } from "../src/adapters/gltf.js";
import { decode, exportGLB, encodeSource, Source, type DecodedMesh } from "../src/index.js";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;
const modelSel = $<HTMLSelectElement>("model"), budgetSel = $<HTMLSelectElement>("budget"), methodSel = $<HTMLSelectElement>("method"), voxSel = $<HTMLSelectElement>("vox");
const hashBox = $<HTMLTextAreaElement>("hash"), stats = $("stats"), msg = $("msg");

class View {
  renderer: THREE.WebGLRenderer;
  scene = new THREE.Scene();
  content = new THREE.Group();
  constructor(canvas: HTMLCanvasElement, public camera: THREE.PerspectiveCamera) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene.background = new THREE.Color(matchMedia("(prefers-color-scheme: dark)").matches ? 0x2b2b29 : 0xffffff);
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x555555, 2.2));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(2, 3, 4);
    this.scene.add(key, this.content);
  }
  set(obj: THREE.Object3D | null): void {
    this.content.clear();
    if (obj) this.content.add(obj);
  }
  render(): void {
    const c = this.renderer.domElement;
    const w = c.clientWidth, h = c.clientHeight;
    if (c.width !== Math.floor(w * this.renderer.getPixelRatio()) || c.height !== Math.floor(h * this.renderer.getPixelRatio())) this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.render(this.scene, this.camera);
  }
}

const camera = new THREE.PerspectiveCamera(35, 4 / 3, 0.05, 50);
camera.position.set(2.2, 1.4, 3.0);
const orig = new View($<HTMLCanvasElement>("c-orig"), camera);
const dec = new View($<HTMLCanvasElement>("c-dec"), camera);
// one camera, two canvases: orbiting either view moves both
const controls = [new OrbitControls(camera, orig.renderer.domElement), new OrbitControls(camera, dec.renderer.domElement)];
function frame(): void {
  controls.forEach((c) => c.update());
  orig.render();
  dec.render();
  requestAnimationFrame(frame);
}

async function decodeImage(bytes: Uint8Array, mimeType: string): Promise<RgbaImage> {
  const bmp = await createImageBitmap(new Blob([bytes as BlobPart], { type: mimeType }));
  const scale = Math.min(1, 1024 / Math.max(bmp.width, bmp.height));
  const w = Math.max(1, Math.round(bmp.width * scale)), h = Math.max(1, Math.round(bmp.height * scale));
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(bmp, 0, 0, w, h);
  return { width: w, height: h, data: ctx.getImageData(0, 0, w, h).data };
}

function toThree(m: DecodedMesh): THREE.Mesh {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(m.positions, 3));
  g.setAttribute("normal", new THREE.BufferAttribute(m.normals, 3));
  const col = new Float32Array(m.colors.length);
  const c = new THREE.Color();
  for (let i = 0; i < col.length; i += 3) {
    c.setRGB(m.colors[i] / 255, m.colors[i + 1] / 255, m.colors[i + 2] / 255, THREE.SRGBColorSpace); // -> linear working space
    col[i] = c.r; col[i + 1] = c.g; col[i + 2] = c.b;
  }
  g.setAttribute("color", new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(m.indices, 1));
  return new THREE.Mesh(g, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0, side: THREE.DoubleSide, flatShading: m.flat }));
}

let source: Source | null = null;
let current: DecodedMesh | null = null;
let currentName = "placeholder";

function showHash(hash: string, extra = ""): void {
  msg.textContent = "";
  msg.className = "";
  try {
    const t0 = performance.now();
    const m = decode(hash.trim(), { vox: voxSel.value as "smooth" | "cubes" });
    const ms = performance.now() - t0;
    current = m;
    dec.set(toThree(m));
    const bytes = Math.floor((hash.trim().length * 3) / 4);
    stats.innerHTML = `${extra}method <b>${m.method}</b> · <b>${bytes}</b> bytes · <b>${hash.trim().length}</b> characters · decoded in <b>${ms.toFixed(1)} ms</b> to <b>${m.indices.length / 3}</b> triangles`;
    $("t-dec").textContent = `Decoded placeholder (${m.method}, ${bytes} bytes)`;
    document.body.dataset.state = "ready";
  } catch (e) {
    msg.textContent = "Cannot decode: " + (e as Error).message;
    msg.className = "err";
    document.body.dataset.state = "error";
  }
}

function encodeCurrent(): void {
  if (!source) return;
  stats.textContent = "Encoding…";
  document.body.dataset.state = "busy";
  setTimeout(() => {
    try {
      const t0 = performance.now();
      const r = encodeSource(source!, { budget: Number(budgetSel.value), method: methodSel.value as "auto" });
      const ms = performance.now() - t0;
      hashBox.value = r.hash;
      const sc = Object.entries(r.scores).map(([k, v]) => `${k} ${(v as number).toFixed(3)}`).join(", ");
      showHash(r.hash, `encoded in <b>${ms.toFixed(0)} ms</b>${sc ? ` (scores: ${sc})` : ""} · `);
    } catch (e) {
      stats.innerHTML = `<span class="err">Encoding failed: ${(e as Error).message}</span>`;
      document.body.dataset.state = "error";
    }
  }, 30);
}

async function loadModel(bytes: Uint8Array, name: string): Promise<void> {
  stats.textContent = "Loading " + name + "…";
  document.body.dataset.state = "busy";
  currentName = name;
  $("t-orig").textContent = `Original: ${name} (${(bytes.length / 1e6).toFixed(2)} MB)`;
  try {
    const input = await loadGltf(bytes, { decodeImage });
    source = new Source(input);
    const gltf = await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer, "");
    // put the original into the same normalised frame as the placeholder
    const holder = new THREE.Group();
    holder.add(gltf.scene);
    holder.scale.setScalar(source.scale);
    holder.position.set(-source.center[0] * source.scale, -source.center[1] * source.scale, -source.center[2] * source.scale);
    orig.set(holder);
    encodeCurrent();
  } catch (e) {
    stats.innerHTML = `<span class="err">Could not load ${name}: ${(e as Error).message}</span>`;
    document.body.dataset.state = "error";
  }
}

async function loadUrl(url: string, name: string): Promise<void> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  await loadModel(new Uint8Array(await r.arrayBuffer()), name);
}

async function init(): Promise<void> {
  frame();
  const q = new URLSearchParams(location.search);
  if (q.get("budget")) budgetSel.value = q.get("budget")!;
  if (q.get("method")) methodSel.value = q.get("method")!;
  if (q.get("vox")) voxSel.value = q.get("vox")!;
  let models: { name: string; url: string }[] = [];
  try {
    models = await (await fetch("models.json")).json();
  } catch {
    /* static hosting without the helper server: file picker and paste still work */
  }
  for (const m of models) modelSel.add(new Option(m.name, m.url));
  if (!models.length) modelSel.add(new Option("(no bundled models - pick a .glb)", ""));
  modelSel.onchange = () => modelSel.value && loadUrl(modelSel.value, modelSel.selectedOptions[0].text).catch((e) => (stats.textContent = String(e)));
  $<HTMLInputElement>("file").onchange = async (ev) => {
    const f = (ev.target as HTMLInputElement).files?.[0];
    if (f) await loadModel(new Uint8Array(await f.arrayBuffer()), f.name);
  };
  budgetSel.onchange = methodSel.onchange = encodeCurrent;
  voxSel.onchange = () => hashBox.value.trim() && showHash(hashBox.value);
  $("decode").onclick = () => showHash(hashBox.value, "pasted string · ");
  $("download").onclick = () => {
    if (!current) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([exportGLB(current) as BlobPart], { type: "model/gltf-binary" }));
    a.download = `${currentName.replace(/\.glb$/i, "")}_${current.method}_placeholder.glb`;
    a.click();
  };
  if (q.get("hash")) {
    hashBox.value = q.get("hash")!;
    modelSel.add(new Option("(hash from URL)", ""), 0);
    modelSel.selectedIndex = 0;
    showHash(hashBox.value, "from URL · ");
    orig.set(null);
    $("t-orig").textContent = "Original (none: hash given in URL)";
  } else if (models.length) {
    const want = models.find((m) => m.name === (q.get("model") ?? "Duck")) ?? models[0];
    modelSel.value = want.url;
    await loadUrl(want.url, want.name);
  } else stats.textContent = "Pick a .glb file or paste a hash.";
}
init().catch((e) => (stats.textContent = String(e)));
