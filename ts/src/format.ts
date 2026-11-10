/** ShapeHash v1: constants, base64url, bit I/O, integer range coder, SH basis. See SPEC.md. */

export const VERSION = 1;
export const METHOD_VOX = 0;
export const METHOD_SH = 1;
export const METHOD_MESH = 2;
export const METHOD_NAMES = ["vox", "sh", "mesh"] as const;
export type Method = (typeof METHOD_NAMES)[number];
export const MAX_PAYLOAD_BYTES = 2048;

export const VOX_MAX_DIM = 64;
export const VOX_HEADER_BYTES = 6;
export const RC_PROB_INIT = 1024;
const RC_SHIFT = 4;
const RC_TOP = 0x1000000;
/** P(bit == 0) * 2048 per occupancy context (4 * faceNeighbours + edgeNeighbours). */
export const OCC_PRIOR = [1884, 1693, 1502, 1311, 1502, 1311, 1120, 928, 1120, 928, 737, 546, 737, 546, 355, 164];

export const SH_MAX_L = 24;
export const SH_MAX_LC = 15;
export const SQRT4PI = 3.5449077018110318;
export const SH_MIN_RADIUS = 0.02;
const SCALE_MANT = [1.0, 1.189207115002721, 1.4142135623730951, 1.681792830507429];

export const MESH_MAX_VERTS = 1023;
export const MESH_MAX_TRIS = 2047;

/** Thrown for every malformed or unsupported hash string. */
export class ShapeHashError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ShapeHashError";
  }
}

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
const A2I = new Int8Array(128).fill(-1);
for (let i = 0; i < 64; i++) A2I[ALPHABET.charCodeAt(i)] = i;

export function toBase64Url(data: Uint8Array): string {
  let out = "";
  for (let i = 0; i < data.length; i += 3) {
    const n = Math.min(3, data.length - i);
    const v = (data[i] << 16) | ((n > 1 ? data[i + 1] : 0) << 8) | (n > 2 ? data[i + 2] : 0);
    out += ALPHABET[(v >> 18) & 63] + ALPHABET[(v >> 12) & 63];
    if (n > 1) out += ALPHABET[(v >> 6) & 63];
    if (n > 2) out += ALPHABET[v & 63];
  }
  return out;
}

export function fromBase64Url(s: string): Uint8Array {
  if (typeof s !== "string") throw new ShapeHashError("hash must be a string");
  const len = s.length;
  if (len % 4 === 1 || len > Math.floor((MAX_PAYLOAD_BYTES * 4 + 2) / 3)) throw new ShapeHashError("bad string length");
  const out = new Uint8Array(Math.floor((len * 3) / 4));
  let o = 0;
  for (let i = 0; i < len; i += 4) {
    const n = Math.min(4, len - i);
    let v = 0;
    for (let k = 0; k < 4; k++) {
      let d = 0;
      if (k < n) {
        const c = s.charCodeAt(i + k);
        d = c < 128 ? A2I[c] : -1;
        if (d < 0) throw new ShapeHashError("character outside the base64url alphabet");
      }
      v = (v << 6) | d;
    }
    out[o++] = (v >> 16) & 255;
    if (n > 2) out[o++] = (v >> 8) & 255;
    if (n > 3) out[o++] = v & 255;
  }
  return out;
}

export class BitWriter {
  private bytes: number[] = [];
  private nbits = 0;
  write(value: number, nbits: number): void {
    if (!Number.isInteger(value) || value < 0 || value >= 2 ** nbits) throw new RangeError(`value ${value} does not fit in ${nbits} bits`);
    for (let i = nbits - 1; i >= 0; i--) {
      if ((this.nbits & 7) === 0) this.bytes.push(0);
      if ((value >>> i) & 1) this.bytes[this.bytes.length - 1] |= 0x80 >> (this.nbits & 7);
      this.nbits++;
    }
  }
  get length(): number {
    return this.nbits;
  }
  toBytes(): Uint8Array {
    return Uint8Array.from(this.bytes);
  }
}

/** Strict MSB-first reader: reading past the end throws. */
export class BitReader {
  private pos = 0;
  constructor(private data: Uint8Array) {}
  read(nbits: number): number {
    if (this.pos + nbits > this.data.length * 8) throw new ShapeHashError("truncated payload");
    let v = 0;
    for (let i = 0; i < nbits; i++) {
      v = (v << 1) | ((this.data[this.pos >> 3] >> (7 - (this.pos & 7))) & 1);
      this.pos++;
    }
    return v;
  }
}

/** LZMA-style binary range decoder. All state is unsigned 32-bit integer. */
export class RangeDecoder {
  private range = 0xffffffff;
  private code = 0;
  constructor(private data: Uint8Array, private pos: number) {
    for (let i = 0; i < 4; i++) this.code = ((this.code << 8) | this.next()) >>> 0;
  }
  private next(): number {
    return this.pos < this.data.length ? this.data[this.pos++] : (this.pos++, 0);
  }
  private norm(): void {
    while (this.range < RC_TOP) {
      this.range = (this.range << 8) >>> 0;
      this.code = ((this.code << 8) | this.next()) >>> 0;
    }
  }
  decode(probs: Uint16Array, i: number): number {
    const p = probs[i];
    const bound = (this.range >>> 11) * p;
    let bit: number;
    if (this.code < bound) {
      this.range = bound;
      probs[i] = p + ((2048 - p) >> RC_SHIFT);
      bit = 0;
    } else {
      this.code -= bound;
      this.range -= bound;
      probs[i] = p - (p >> RC_SHIFT);
      bit = 1;
    }
    this.norm();
    return bit;
  }
  bypass(nbits: number): number {
    let v = 0;
    for (let i = 0; i < nbits; i++) {
      const bound = this.range >>> 1;
      if (this.code < bound) {
        this.range = bound;
        v <<= 1;
      } else {
        this.code -= bound;
        this.range -= bound;
        v = (v << 1) | 1;
      }
      this.norm();
    }
    return v;
  }
}

const TWO32 = 4294967296;

export class RangeEncoder {
  private low = 0; // up to 33 bits, kept as an exact double
  private range = 0xffffffff;
  private cache = 0;
  private cacheSize = 1;
  out: number[] = [];
  copy(): RangeEncoder {
    const c = new RangeEncoder();
    c.low = this.low;
    c.range = this.range;
    c.cache = this.cache;
    c.cacheSize = this.cacheSize;
    c.out = this.out.slice();
    return c;
  }
  private shiftLow(): void {
    if (this.low < 0xff000000 || this.low >= TWO32) {
      const carry = this.low >= TWO32 ? 1 : 0;
      let temp = this.cache;
      do {
        this.out.push((temp + carry) & 0xff);
        temp = 0xff;
      } while (--this.cacheSize !== 0);
      this.cache = Math.floor(this.low / 16777216) & 0xff;
    }
    this.cacheSize++;
    this.low = (this.low % 16777216) * 256;
  }
  private norm(): void {
    while (this.range < RC_TOP) {
      this.range = (this.range << 8) >>> 0;
      this.shiftLow();
    }
  }
  encode(probs: Uint16Array, i: number, bit: number): void {
    const p = probs[i];
    const bound = (this.range >>> 11) * p;
    if (bit === 0) {
      this.range = bound;
      probs[i] = p + ((2048 - p) >> RC_SHIFT);
    } else {
      this.low += bound;
      this.range -= bound;
      probs[i] = p - (p >> RC_SHIFT);
    }
    this.norm();
  }
  bypass(value: number, nbits: number): void {
    for (let k = nbits - 1; k >= 0; k--) {
      const bound = this.range >>> 1;
      if ((value >>> k) & 1) {
        this.low += bound;
        this.range -= bound;
      } else {
        this.range = bound;
      }
      this.norm();
    }
  }
  /** Bytes emitted so far (lower bound on the final size). */
  get emitted(): number {
    return this.out.length;
  }
  /** Value with the most trailing zero bits inside the final interval; trailing zero bytes dropped. */
  finish(): Uint8Array {
    const c = this.copy();
    const hi = c.low + c.range - 1;
    for (const s of [32, 24, 16, 8, 0]) {
      const m = 2 ** s;
      const v = Math.ceil(c.low / m) * m;
      if (v <= hi) {
        c.low = v;
        break;
      }
    }
    for (let i = 0; i < 5; i++) c.shiftLow();
    let n = c.out.length;
    while (n > 1 && c.out[n - 1] === 0) n--;
    return Uint8Array.from(c.out.slice(1, n));
  }
}

export function coeffScale(qs: number): number {
  return SCALE_MANT[qs & 3] * 2 ** ((qs >> 2) - 13);
}

export function dequantCoeff(q: number, nbits: number, scale: number): number {
  const half = (1 << (nbits - 1)) - 1;
  const u = (q - half) / half;
  return (u < 0 ? -1 : 1) * u * u * scale;
}

export function rgb555to8(c: number): number {
  return (c << 3) | (c >> 2);
}

/** Real SH basis values for one direction, written to out[0 .. (L+1)^2). SPEC.md section 5.2. */
export function realSH(L: number, x: number, y: number, z: number, out: Float64Array, norm: Float64Array): void {
  if (z > 1) z = 1;
  else if (z < -1) z = -1;
  const s = Math.sqrt(Math.max(0, 1 - z * z));
  const phi = Math.atan2(y, x);
  let pmm = 1;
  for (let m = 0; m <= L; m++) {
    if (m > 0) pmm = -pmm * (2 * m - 1) * s;
    const cm = Math.cos(m * phi);
    const sm = Math.sin(m * phi);
    let p2 = pmm;
    let p1 = 0;
    for (let l = m; l <= L; l++) {
      let p: number;
      if (l === m) p = pmm;
      else if (l === m + 1) p = z * (2 * m + 1) * pmm;
      else p = (z * (2 * l - 1) * p1 - (l + m - 1) * p2) / (l - m);
      p2 = p1;
      p1 = p;
      const k = norm[l * (SH_MAX_L + 1) + m];
      if (m === 0) out[l * l + l] = k * p;
      else {
        out[l * l + l + m] = Math.SQRT2 * k * p * cm;
        out[l * l + l - m] = Math.SQRT2 * k * p * sm;
      }
    }
  }
}

let SH_NORM: Float64Array | null = null;
export function shNorm(): Float64Array {
  if (SH_NORM) return SH_NORM;
  const n = new Float64Array((SH_MAX_L + 1) * (SH_MAX_L + 1));
  for (let l = 0; l <= SH_MAX_L; l++)
    for (let m = 0; m <= l; m++) {
      let k = (2 * l + 1) / (4 * Math.PI);
      for (let i = l - m + 1; i <= l + m; i++) k /= i;
      n[l * (SH_MAX_L + 1) + m] = Math.sqrt(k);
    }
  return (SH_NORM = n);
}

const ICO = new Map<number, { dirs: Float64Array; faces: Uint32Array }>();
/** Unit icosphere with deterministic vertex order (SPEC.md appendix B). */
export function icosphere(subdiv: number): { dirs: Float64Array; faces: Uint32Array } {
  const hit = ICO.get(subdiv);
  if (hit) return hit;
  const t = (1 + Math.sqrt(5)) / 2;
  const base = [-1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, 0, 0, -1, t, 0, 1, t, 0, -1, -t, 0, 1, -t, t, 0, -1, t, 0, 1, -t, 0, -1, -t, 0, 1];
  const inv = Math.sqrt(1 + t * t);
  const v: number[] = base.map((c) => c / inv);
  let f = [0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8, 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1];
  for (let s = 0; s < subdiv; s++) {
    const mid = new Map<number, number>();
    const nf: number[] = [];
    const m = (a: number, b: number): number => {
      const key = a < b ? a * 65536 + b : b * 65536 + a;
      let i = mid.get(key);
      if (i === undefined) {
        const x = v[3 * a] + v[3 * b], y = v[3 * a + 1] + v[3 * b + 1], z = v[3 * a + 2] + v[3 * b + 2];
        const n = Math.sqrt(x * x + y * y + z * z);
        i = v.length / 3;
        v.push(x / n, y / n, z / n);
        mid.set(key, i);
      }
      return i;
    };
    for (let i = 0; i < f.length; i += 3) {
      const a = f[i], b = f[i + 1], c = f[i + 2];
      const ab = m(a, b), bc = m(b, c), ca = m(c, a);
      nf.push(a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca);
    }
    f = nf;
  }
  const r = { dirs: Float64Array.from(v), faces: Uint32Array.from(f) };
  ICO.set(subdiv, r);
  return r;
}
