/** ShapeHash: BlurHash-style placeholder strings for 3D assets. */
export { decode, parse, build, vertexNormals, ShapeHashError } from "./decode.js";
export type { DecodedMesh, DecodeOptions, Parsed, Method } from "./decode.js";
export { exportGLB } from "./glb.js";
export type { GlbOptions } from "./glb.js";
export { encode, encodeSource, Source } from "./encode/index.js";
export type { EncodeOptions, EncodeResult, MeshInput, ColorSampler } from "./encode/index.js";
