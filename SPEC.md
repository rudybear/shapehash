# ShapeHash format, version 1

Status: frozen for evaluation (release candidate). "ShapeHash" is a working
name.

A ShapeHash is a short ASCII string that decodes to a coloured triangle mesh
approximating a 3D asset. This document defines the string, the three payload
types and the exact arithmetic needed to decode them. Two implementations are
checked against each other: the Python reference (`hash3d/v1/`) and the
TypeScript library (`ts/`).

Conventions: all integers are unsigned; `uN` is an N-bit field written most
significant bit first; `floor`, `ceil` are the usual functions; `/` on
integers below is exact rational arithmetic evaluated in IEEE-754 binary64
unless stated otherwise.

## 1. String

- The payload is a byte string of 3 to 2048 bytes.
- The string is the payload in base64url (RFC 4648 section 5, alphabet
  `A-Z a-z 0-9 - _`) **without padding**. All characters are safe in URLs,
  JSON strings, SQL literals and HTML attributes.
- A decoder MUST reject strings containing any other character, strings whose
  length is 1 modulo 4, and strings longer than 2731 characters.
- Unused low bits of the last character are ignored.

Length: `chars = ceil(4 * bytes / 3)`. 64 bytes -> 86 characters, 128 -> 171,
256 -> 342.

## 2. Coordinate frame

The decoded mesh lives in the *normalised frame*: the asset's axis-aligned
bounding box is centred on the origin and scaled uniformly so that its longest
half-extent is 1. Absolute position and scale are not stored; the host
application applies its own transform (the encoder returns `center` and
`scale` with `p_normalised = (p - center) * scale`). Axes are the asset's own
(glTF: +Y up).

## 3. Common header (26 bits)

| field | bits | meaning |
|---|---|---|
| version | u3 | must be 1 |
| method | u2 | 0 = vox, 1 = sh, 2 = mesh, 3 = invalid |
| param | u3 | method specific |
| ex, ey, ez | 3 x u6 | bounding-box half extents: `e_k = (q_k + 1) / 64` |

Encoders write `q_k = clamp(ceil(64 * e_k) - 1, 0, 63)`, so the stored box
contains the true one and the longest axis has `e = 1`.

Colour conventions used below: a 4-bit channel `c` expands to the 8-bit sRGB
value `17 * c`; a 5-bit channel expands to `(c << 3) | (c >> 2)`.

A decoder MUST fail on version != 1, method 3, reserved values, or any read
past the end of the payload, except where section 4.2 says otherwise.

## 4. Method 0: vox (occupancy grid + colour cells)

### 4.1 Layout

Bits 26..47 complete a 6-byte header:

| field | bits | meaning |
|---|---|---|
| nx-1, ny-1, nz-1 | 3 x u6 | grid dimensions, 1..64 each |
| f-1 | u4 | colour cell size in voxels, 1..16 |

`param` is the colour mode `cm`: 0 = direct RGB444 per cell, 1..6 = palette of
`P = 2^(cm-1)` entries (1, 2, 4, 8, 16, 32), 7 = reserved (reject).

Bytes 6.. are one range-coded stream (section 4.2) containing, in order:

1. **Occupancy**: one adaptive bit per voxel in scan order x (outer), y, z
   (inner); 1 = occupied.
2. **Palette** (cm >= 1): `P` entries, each three 5-bit bypass values R, G, B.
3. **Cell colours**: for each *active* colour cell in scan order x, y, z.

Voxel (x, y, z) covers the box
`[-e + 2e * (x, y, z) / n, -e + 2e * (x+1, y+1, z+1) / n]` component-wise.
If no voxel is occupied the hash is invalid.

**Occupancy contexts.** With already-decoded neighbours (0 outside the grid)
`a = o(x-1,y,z)`, `b = o(x,y-1,z)`, `c = o(x,y,z-1)`, `d = o(x-1,y-1,z)`,
`g = o(x-1,y,z-1)`, `h = o(x,y-1,z-1)`, the context is
`4 * (a + b + c) + d + g + h` (0..15). Initial probabilities of a 0 bit,
scaled by 2048:

```
1884 1693 1502 1311  1502 1311 1120 928  1120 928 737 546  737 546 355 164
```

**Colour cells.** Cell (i, j, k) contains voxels with
`floor(x/f) = i`, etc.; the cell grid has `ceil(n/f)` cells per axis. A voxel
is a *surface voxel* if it is occupied and at least one of its six face
neighbours is empty or outside the grid. A cell is *active* if it contains a
surface voxel. Only active cells carry a colour.

- `cm = 0`: three 4-bit bypass values R, G, B per active cell.
- `cm >= 1`, `pb = cm - 1` bits per index: the index is coded most significant
  bit first as a binary tree. `node = 1`; for each bit:
  `bit = decode(prob[ctx * P + node])`, `node = 2 * node + bit`; the index is
  `node - P`. `ctx` is the index of the first of the cells (i, j, k-1),
  (i, j-1, k), (i-1, j, k) that exists and is active, else `P`. All
  `(P + 1) * P` probabilities start at 1024. For `P = 1` no bits are coded.

### 4.2 Range coder

A binary range coder with 32-bit integer state (the LZMA construction).
Probabilities are 11-bit integers `p` = P(bit = 0) * 2048.

Decoder state: `range = 0xFFFFFFFF`, `code` = the first four stream bytes, big
endian. **Reading past the end of the payload yields zero bytes** (encoders
drop trailing zero bytes), so a vox payload can never be "truncated": any
byte string with a valid 6-byte header decodes to some grid.

```
decode(p):                          # adaptive bit
    bound = (range >> 11) * p
    if code < bound: range = bound;                 p += (2048 - p) >> 4; bit = 0
    else:            code -= bound; range -= bound; p -= p >> 4;          bit = 1
    normalise(); return bit

bypass():                           # equiprobable bit, no adaptation
    bound = range >> 1
    if code < bound: range = bound; bit = 0
    else:            code -= bound; range -= bound; bit = 1
    normalise(); return bit

normalise():
    while range < 2^24: range = (range << 8) mod 2^32; code = ((code << 8) | next_byte()) mod 2^32
```

Multi-bit bypass values are read most significant bit first. All quantities
are integers below 2^32 (`bound` can equal 2^32 - 2^11 at most), so the
decoder is bit-exact on any platform with 32-bit unsigned arithmetic or
53-bit-exact doubles.

Encoder (informative): the standard LZMA encoder (`low` 33 bits, cache byte,
carry propagation). The first output byte is always 0 and is not stored. To
finish, choose the value in `[low, low + range)` with the most trailing zero
bits, flush five bytes, and strip trailing zero bytes.

### 4.3 Output

The normative content of a vox hash is the grid, the active-cell map and the
8-bit cell colours. How it is drawn is up to the decoder. Two renderings are
defined so that implementations can be compared:

- **cubes**: for axis = x, y, z and side = -, +: for every voxel in scan order
  whose neighbour on that side is empty or outside, a quad of four unshared
  vertices with the voxel's cell colour and the face normal.
- **smooth** (appendix A): surface nets on a blurred, 2x upsampled grid.

Decoders SHOULD refuse to build meshes above a triangle limit (default
500 000 in both implementations); a 64^3 checkerboard would otherwise produce
about 1.5 million triangles.

## 5. Method 1: sh (spherical harmonics, star-shaped)

`param` must be 0.

### 5.1 Layout (after the common header)

| field | bits | meaning |
|---|---|---|
| cx, cy, cz | 3 x u5 | centre in box coordinates: `c_k = q_k / 31 - 0.5` |
| L | u5 | radius degree, 0..24 |
| Ly | u5 | luma degree, 0..24 |
| Lc | u4 | chroma degree, 0..15 |
| radius group | | DC u7, then if L > 0: scale u6 and `(L+1)^2 - 1` codes of 5 bits |
| luma group | | DC u6, then if Ly > 0: scale u6 and `(Ly+1)^2 - 1` codes of 4 bits |
| Co group | | DC u5, then if Lc > 0: scale u6 and `(Lc+1)^2 - 1` codes of 3 bits |
| Cg group | | same as Co |

Dequantisation, with `S = 3.5449077018110318` (sqrt(4 pi)):

- DC: `a_0 = (lo + q / (2^b - 1) * (hi - lo)) * S` with (b, lo, hi) =
  (7, 0, 2) radius, (6, 0, 1) luma, (5, -0.5, 0.5) chroma.
- Scale: `s = M[qs mod 4] * 2^(floor(qs / 4) - 13)`,
  `M = [1, 1.189207115002721, 1.4142135623730951, 1.681792830507429]`.
- AC code `q` with `n` bits, `h = 2^(n-1) - 1`: `u = (q - h) / h`,
  `a = sign(u) * u^2 * s`. The code `2^n - 1` is reserved (reject).

Coefficient index for degree l, order m (-l..l) is `l^2 + l + m`; codes are
stored in increasing index starting at index 1.

### 5.2 Basis

For a unit direction (x, y, z): `s = sqrt(max(0, 1 - z^2))`,
`phi = atan2(y, x)`,

```
P(m, m) = (-1)^m (2m-1)!! s^m        computed as P(m,m) = -P(m-1,m-1) * (2m-1) * s, P(0,0) = 1
P(m+1, m) = z (2m+1) P(m, m)
P(l, m) = (z (2l-1) P(l-1, m) - (l+m-1) P(l-2, m)) / (l - m)
K(l, m) = sqrt((2l+1) / (4 pi) * (l-m)! / (l+m)!)
Y(l, 0)  = K(l,0) P(l,0)
Y(l, m)  = sqrt(2) K(l,m) P(l,m) cos(m phi)      m > 0
Y(l, -m) = sqrt(2) K(l,m) P(l,m) sin(m phi)      m > 0
```

### 5.3 Output

For each direction `d` of a sphere tessellation (any; both implementations use
the icosphere of appendix B, subdivision 3 for L <= 8, else 4):

```
r = max(sum_i a_i Y_i(d), 0.02)                       radius group
position = (c + r * d) * e                            component-wise
Y, Co, Cg = the three colour series at d
R, G, B = Y + Co - Cg,  Y + Cg,  Y - Co - Cg          clamp to [0, 1]
8-bit colour = min(255, max(0, floor(255 * v + 0.5)))
```

Because `cos`, `sin` and `atan2` are not correctly rounded on every platform,
sh output is defined up to a tolerance: positions within 1e-5, colours within
1/255. (On the tested platform the two implementations agree bit for bit.)

## 6. Method 2: mesh (indexed triangles)

`param`: bits 0..1 = `pb - 5` (position bits per component, 5..8); bit 2
reserved, must be 0.

### 6.1 Layout (after the common header)

| field | bits | meaning |
|---|---|---|
| nv | u10 | vertex count, 3..1023 |
| nt | u11 | triangle count, 1..2047 |
| vertices | nv x (3 x pb + 12) | position qx, qy, qz then RGB444 |
| connectivity | variable | see below |

Position: `p_k = (2 * q_k / (2^pb - 1) - 1) * e_k`.

Connectivity is a sequence of `nt` triangles. Vertices are numbered in order
of first use; `seen` counts vertices used so far.

```
ref():
    if u1 == 1:                      # next unused vertex
        require seen < nv; return seen++
    require seen > 0
    v = u(B) where B = max(1, ceil(log2(seen)));  require v < seen;  return v

triangle 0:          (a, b, c) = (ref(), ref(), ref())
triangle t > 0:      if u1 == 0:  (a, b, c) = (ref(), ref(), ref())      # restart
                     else: side = u1; d = ref()
                           side 0: (a, b, c) = (c, b, d)                 # across edge b-c
                           side 1: (a, b, c) = (a, c, d)                 # across edge a-c
```

Winding is not guaranteed to be consistent; render double-sided. All fields
are integers, so two decoders produce identical vertex and index buffers.

## 7. Limits a decoder must enforce

| item | limit |
|---|---|
| string length | <= 2731 characters (2048 bytes) |
| vox grid | <= 64 per axis (262 144 range-decoded bits), colour cells <= voxels |
| vox mesh | triangle cap, default 500 000 |
| sh degree | L, Ly <= 24, Lc <= 15 (at most 625 coefficients per series) |
| mesh | nv <= 1023, nt <= 2047, every index < number of vertices seen |

Work and memory are therefore bounded by constants independent of the input;
nothing in the payload is used as an allocation size beyond these fields.

## 8. Budgets

Any payload size from the method minimum up to 2048 bytes is legal; the
encoder takes the budget in bytes and never exceeds it. Recommended:

- **128 bytes (171 characters)**: default. Mean composite quality 0.82 on the
  28-model test set; limbs, legs and holes are present.
- **64 bytes (86 characters)**: compact. 0.78; the smallest size at which most
  test models are recognisable.
- **256 bytes (342 characters)**: 0.85; diminishing returns above this
  (+0.03 for another doubling).

Below about 32 bytes the mesh method cannot fit anything and the others give
a coloured blob. Sizes are not stored: the string length is the payload size.

## Appendix A: smooth vox rendering (reference algorithm)

1. Upsample the occupancy grid by 2 (each voxel becomes 2x2x2 samples), pad
   with 2 zero samples on every side. Sample (i, j, k) of the padded grid has
   voxel coordinate `((i, j, k) - 2 + 0.5) / 2`.
2. Blur with the integer kernel [1, 2, 1] along x, then y, then z (zeros
   outside). Values are 0..64. Let `g = 2 * value - 64`; a sample is inside
   iff `g > 0`.
3. Surface nets: for every cell (the cube spanned by samples (i..i+1,
   j..j+1, k..k+1)) in scan order whose 8 corners are not all inside or all
   outside, emit one vertex at the cell origin plus the mean of the crossing
   points on its sign-changing edges. For an edge from corner A to corner B
   the crossing is `A + t (B - A)`, `t = gA / (gA - gB)`; edges are visited in
   the order x-edges (4), y-edges (4), z-edges (4), each by increasing corner
   index with corner index `dx + 2 dy + 4 dz`.
4. For axis a = x, y, z (b, c the next two axes cyclically) and every sample
   edge p -> p + a in scan order whose ends differ: the four cells
   `p - b - c`, `p - c`, `p`, `p - b` give vertices v0..v3; emit triangles
   (v0, v1, v2), (v0, v2, v3) if p is inside, else (v0, v3, v2), (v0, v2, v1).
5. Vertex position: voxel coordinate `v` -> `v * 2e / n - e`.
6. Vertex colour: with `u = v / f - 0.5`, blend the 8 cells around `u`
   trilinearly, using active cells only and renormalising; if none is active
   use the mean of all active cell colours; round with `floor(x + 0.5)`.

If the result has no triangles (isolated voxels vanish under the blur), fall
back to cubes.

## Appendix B: icosphere

Start from the 12 vertices
`(-1,t,0) (1,t,0) (-1,-t,0) (1,-t,0) (0,-1,t) (0,1,t) (0,-1,-t) (0,1,-t) (t,0,-1) (t,0,1) (-t,0,-1) (-t,0,1)`
with `t = (1 + sqrt 5) / 2`, each divided by `sqrt(1 + t^2)`, and the faces

```
0 11 5, 0 5 1, 0 1 7, 0 7 10, 0 10 11, 1 5 9, 5 11 4, 11 10 2, 10 7 6, 7 1 8,
3 9 4, 3 4 2, 3 2 6, 3 6 8, 3 8 9, 4 9 5, 2 4 11, 6 2 10, 8 6 7, 9 8 1
```

Each subdivision replaces face (a, b, c) by (a, ab, ca), (b, bc, ab),
(c, ca, bc), (ab, bc, ca), where `ab` is the normalised midpoint of a and b,
appended to the vertex list the first time it is needed (midpoints requested
in the order ab, bc, ca).

## Appendix C: what is not normative

The encoder. Any string that parses is a valid hash; encoders are free to
choose grid sizes, colour modes, degrees and vertex counts. The reference
encoders (Python `hash3d/v1/encode.py`, TypeScript `ts/src/encode/`) share
their algorithms and are described in the README.
