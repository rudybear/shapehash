/** ShapeHash encoder: tries vox, sh and mesh and keeps the best string for the budget. */
import { decode, type Method } from "../decode.js";
import { MAX_PAYLOAD_BYTES } from "../format.js";
import { encodeMesh } from "./mesh.js";
import { encodeSh } from "./sh.js";
import { Source, type MeshInput } from "./source.js";
import { encodeVox } from "./vox.js";

export { Source } from "./source.js";
export type { MeshInput, ColorSampler } from "./source.js";

export interface EncodeOptions {
  /** Payload size limit in bytes (the string is ceil(4/3 * bytes) characters). Default 128. */
  budget?: number;
  /** "auto" (default) tries all three methods and keeps the best under the selection score. */
  method?: "auto" | Method;
  /** Number of surface samples drawn from the input (default 600 000). */
  samples?: number;
}

export interface EncodeResult {
  hash: string;
  method: Method;
  /** Payload size in bytes. */
  bytes: number;
  /** Selection score per candidate method (only when more than one was tried). */
  scores: Partial<Record<Method, number>>;
  /** Every candidate string that fitted the budget. */
  candidates: Partial<Record<Method, string>>;
  /** Maps the input frame to the normalised frame of the decoded mesh: p_norm = (p - center) * scale. */
  center: [number, number, number];
  scale: number;
}

const ENCODERS: Record<Method, (src: Source, budget: number) => string | null> = { vox: encodeVox, sh: encodeSh, mesh: encodeMesh };

/** Encode from a prepared Source (reuse it to encode the same asset at several budgets). */
export function encodeSource(src: Source, options: EncodeOptions = {}): EncodeResult {
  const budget = options.budget ?? 128;
  if (!Number.isInteger(budget) || budget < 8 || budget > MAX_PAYLOAD_BYTES) throw new RangeError("budget must be an integer between 8 and 2048 bytes");
  const names: Method[] = !options.method || options.method === "auto" ? ["vox", "sh", "mesh"] : [options.method];
  const candidates: Partial<Record<Method, string>> = {};
  for (const name of names) {
    const s = ENCODERS[name](src, budget);
    if (s !== null) candidates[name] = s;
  }
  const got = Object.keys(candidates) as Method[];
  if (got.length === 0) throw new RangeError(`nothing fits in ${budget} bytes`);
  const scores: Partial<Record<Method, number>> = {};
  let pick = got[0];
  if (got.length > 1) {
    for (const k of got) scores[k] = src.score(decode(candidates[k]!), true);
    for (const k of got) if (scores[k]! > scores[pick]!) pick = k;
  }
  const hash = candidates[pick]!;
  return { hash, method: pick, bytes: Math.floor((hash.length * 3) / 4), scores, candidates, center: src.center, scale: src.scale };
}

/** Encode a coloured triangle mesh into a ShapeHash string. */
export function encode(input: MeshInput, options: EncodeOptions = {}): EncodeResult {
  return encodeSource(new Source(input, options.samples), options);
}
