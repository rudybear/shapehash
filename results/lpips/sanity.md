# LPIPS on ShapeHash renders: sanity checks

Renders: orthographic, 256x256, white background, colour = albedo x (0.35 + 0.65 |n.v|) (headlight), six fixed views
(azimuth/elevation 30/20, 120/20, 210/20, 300/20, 75/65, 255/-40). Values are means over views and over the 28 models; lower is better.
Variants: `full` = whole frame; `crop` = cropped to the union of both silhouettes and resized to 224 px; `lp` = crop, reference texture
low-passed on the surface; `blur` = crop, reference image blurred (sigma 4 px); `grey` = crop, both with uniform grey albedo.

## Baselines and methods at 128 bytes

|  | composite | full_alex | full_vgg | full_dists | crop_alex | crop_vgg | crop_dists | lp_alex | lp_vgg | lp_dists | blur_alex | blur_vgg | blur_dists | grey_alex | grey_vgg | grey_dists |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| blank white image | - | 0.468 | 0.247 | 0.589 | 0.638 | 0.389 | 0.645 | 0.629 | 0.386 | 0.639 | 0.650 | 0.402 | 0.601 | 0.572 | 0.347 | 0.602 |
| bounding box, mean colour | 0.428 | 0.453 | 0.332 | 0.371 | 0.565 | 0.450 | 0.392 | 0.552 | 0.439 | 0.385 | 0.661 | 0.460 | 0.428 | 0.474 | 0.378 | 0.342 |
| ellipsoid in the bbox, mean colour | - | 0.305 | 0.237 | 0.352 | 0.465 | 0.402 | 0.394 | 0.451 | 0.387 | 0.380 | 0.568 | 0.406 | 0.404 | 0.418 | 0.343 | 0.347 |
| box + 6 face images | 0.462 | 0.439 | 0.319 | 0.351 | 0.546 | 0.436 | 0.367 | 0.525 | 0.423 | 0.351 | 0.618 | 0.443 | 0.398 | 0.474 | 0.378 | 0.342 |
| convex hull (k-DOP) | 0.572 | 0.239 | 0.196 | 0.294 | 0.399 | 0.359 | 0.318 | 0.381 | 0.345 | 0.312 | 0.565 | 0.382 | 0.385 | 0.335 | 0.303 | 0.285 |
| ORIGINAL geometry, one flat colour | - | 0.086 | 0.079 | 0.169 | 0.146 | 0.140 | 0.186 | 0.104 | 0.105 | 0.164 | 0.469 | 0.281 | 0.328 | 0.000 | 0.000 | 0.000 |
| ORIGINAL geometry, low-passed texture | - | 0.041 | 0.045 | 0.118 | 0.074 | 0.081 | 0.125 | 0.000 | 0.000 | 0.000 | 0.399 | 0.221 | 0.281 | 0.000 | 0.000 | 0.000 |
| phase 1 sh | 0.734 | 0.200 | 0.160 | 0.280 | 0.300 | 0.271 | 0.300 | 0.264 | 0.249 | 0.274 | 0.458 | 0.296 | 0.326 | 0.269 | 0.236 | 0.266 |
| phase 1 dct | 0.698 | 0.210 | 0.170 | 0.292 | 0.305 | 0.280 | 0.315 | 0.275 | 0.259 | 0.291 | 0.470 | 0.300 | 0.341 | 0.276 | 0.243 | 0.283 |
| phase 1 ellip | 0.757 | 0.173 | 0.159 | 0.278 | 0.269 | 0.275 | 0.294 | 0.258 | 0.262 | 0.287 | 0.496 | 0.317 | 0.359 | 0.249 | 0.245 | 0.287 |
| phase 1 mesh | 0.766 | 0.179 | 0.146 | 0.275 | 0.282 | 0.260 | 0.298 | 0.246 | 0.238 | 0.278 | 0.468 | 0.297 | 0.342 | 0.244 | 0.217 | 0.263 |
| phase 1 vox | 0.714 | 0.197 | 0.175 | 0.287 | 0.306 | 0.304 | 0.306 | 0.293 | 0.294 | 0.305 | 0.528 | 0.346 | 0.396 | 0.263 | 0.256 | 0.281 |
| phase 1 voxs | 0.763 | 0.136 | 0.128 | 0.242 | 0.228 | 0.235 | 0.261 | 0.189 | 0.213 | 0.240 | 0.457 | 0.283 | 0.326 | 0.178 | 0.194 | 0.228 |
| v1 vox1c | 0.770 | 0.181 | 0.164 | 0.283 | 0.283 | 0.289 | 0.302 | 0.280 | 0.283 | 0.304 | 0.517 | 0.335 | 0.393 | 0.255 | 0.252 | 0.280 |
| v1 vox1 | 0.811 | 0.124 | 0.123 | 0.237 | 0.212 | 0.225 | 0.250 | 0.170 | 0.201 | 0.229 | 0.440 | 0.273 | 0.320 | 0.179 | 0.197 | 0.231 |
| v1 sh1 | 0.730 | 0.200 | 0.162 | 0.284 | 0.301 | 0.274 | 0.303 | 0.263 | 0.251 | 0.276 | 0.460 | 0.297 | 0.327 | 0.268 | 0.236 | 0.270 |
| v1 mesh1 | 0.781 | 0.172 | 0.139 | 0.269 | 0.268 | 0.247 | 0.292 | 0.234 | 0.227 | 0.273 | 0.463 | 0.290 | 0.339 | 0.222 | 0.200 | 0.252 |
| v1 union1 | 0.822 | 0.130 | 0.121 | 0.240 | 0.216 | 0.220 | 0.257 | 0.173 | 0.195 | 0.233 | 0.438 | 0.269 | 0.314 | 0.181 | 0.185 | 0.229 |

Mean fraction of the 256 px frame covered by the object: 0.169.

## Monotonic with budget? (crop variant)


`crop_alex`

| method | 32 B | 64 B | 128 B | 256 B | 512 B |
|---|---|---|---|---|---|
| kdop | 0.495 | 0.443 | 0.399 | 0.377 | 0.362 |
| sh | 0.384 | 0.334 | 0.300 | 0.272 | 0.255 |
| dct | 0.368 | 0.334 | 0.305 | 0.261 | 0.226 |
| ellip | 0.321 | 0.287 | 0.269 | 0.245 | 0.232 |
| mesh | 0.500 | 0.341 | 0.282 | 0.235 | 0.211 |
| vox | 0.417 | 0.349 | 0.306 | 0.268 | 0.235 |
| voxs | 0.323 | 0.260 | 0.228 | 0.195 | 0.172 |
| vox1c | 0.376 | 0.326 | 0.283 | 0.245 | 0.208 |
| vox1 | 0.286 | 0.247 | 0.212 | 0.185 | 0.157 |
| sh1 | 0.376 | 0.326 | 0.301 | 0.271 | 0.245 |
| mesh1 | 0.435 | 0.338 | 0.268 | 0.225 | 0.197 |
| union1 | 0.280 | 0.240 | 0.216 | 0.186 | 0.157 |

`crop_vgg`

| method | 32 B | 64 B | 128 B | 256 B | 512 B |
|---|---|---|---|---|---|
| kdop | 0.409 | 0.386 | 0.359 | 0.341 | 0.330 |
| sh | 0.317 | 0.291 | 0.271 | 0.256 | 0.249 |
| dct | 0.314 | 0.294 | 0.280 | 0.253 | 0.231 |
| ellip | 0.302 | 0.282 | 0.275 | 0.262 | 0.255 |
| mesh | 0.358 | 0.292 | 0.260 | 0.233 | 0.216 |
| vox | 0.373 | 0.329 | 0.304 | 0.283 | 0.262 |
| voxs | 0.301 | 0.259 | 0.235 | 0.214 | 0.194 |
| vox1c | 0.338 | 0.311 | 0.289 | 0.265 | 0.237 |
| vox1 | 0.268 | 0.245 | 0.225 | 0.205 | 0.184 |
| sh1 | 0.316 | 0.289 | 0.274 | 0.259 | 0.246 |
| mesh1 | 0.342 | 0.287 | 0.247 | 0.220 | 0.200 |
| union1 | 0.264 | 0.239 | 0.220 | 0.201 | 0.178 |

`crop_dists`

| method | 32 B | 64 B | 128 B | 256 B | 512 B |
|---|---|---|---|---|---|
| kdop | 0.348 | 0.333 | 0.318 | 0.309 | 0.304 |
| sh | 0.370 | 0.327 | 0.300 | 0.283 | 0.278 |
| dct | 0.335 | 0.328 | 0.315 | 0.292 | 0.273 |
| ellip | 0.316 | 0.300 | 0.294 | 0.295 | 0.293 |
| mesh | 0.467 | 0.335 | 0.298 | 0.270 | 0.247 |
| vox | 0.340 | 0.319 | 0.306 | 0.293 | 0.279 |
| voxs | 0.299 | 0.275 | 0.261 | 0.245 | 0.231 |
| vox1c | 0.325 | 0.311 | 0.302 | 0.285 | 0.267 |
| vox1 | 0.283 | 0.266 | 0.250 | 0.236 | 0.222 |
| sh1 | 0.364 | 0.322 | 0.303 | 0.285 | 0.278 |
| mesh1 | 0.394 | 0.334 | 0.292 | 0.264 | 0.242 |
| union1 | 0.294 | 0.269 | 0.257 | 0.240 | 0.218 |

`grey_alex`

| method | 32 B | 64 B | 128 B | 256 B | 512 B |
|---|---|---|---|---|---|
| kdop | 0.422 | 0.375 | 0.335 | 0.314 | 0.302 |
| sh | 0.346 | 0.297 | 0.269 | 0.242 | 0.224 |
| dct | 0.328 | 0.304 | 0.276 | 0.233 | 0.195 |
| ellip | 0.290 | 0.264 | 0.249 | 0.237 | 0.229 |
| mesh | 0.456 | 0.300 | 0.244 | 0.201 | 0.173 |
| vox | 0.352 | 0.296 | 0.263 | 0.232 | 0.217 |
| voxs | 0.251 | 0.206 | 0.178 | 0.145 | 0.129 |
| vox1c | 0.332 | 0.290 | 0.255 | 0.230 | 0.209 |
| vox1 | 0.249 | 0.211 | 0.179 | 0.161 | 0.145 |
| sh1 | 0.338 | 0.289 | 0.268 | 0.241 | 0.219 |
| mesh1 | 0.395 | 0.294 | 0.222 | 0.181 | 0.155 |
| union1 | 0.240 | 0.201 | 0.181 | 0.159 | 0.141 |

Per model: doubling the budget (64->128, 128->256) made `crop_alex` worse in 21 of 560 method/model cases.

## Is it texture or geometry?

`orig_lp` is the exact original geometry with the texture low-passed, `orig_flat` the exact geometry in one colour. If LPIPS were dominated by missing
texture detail (as in 2-D) these would score close to the placeholders. Per-model `crop_alex`:

| model | coverage | orig_lp | orig_flat | v1 union 128 B | sphere | blank |
|---|---|---|---|---|---|---|
| AntiqueCamera | 0.06 | 0.031 | 0.041 | 0.186 | 0.451 | 0.547 |
| Avocado | 0.13 | 0.024 | 0.067 | 0.099 | 0.300 | 0.637 |
| BarramundiFish | 0.06 | 0.086 | 0.100 | 0.176 | 0.398 | 0.577 |
| BoomBox | 0.29 | 0.211 | 0.319 | 0.394 | 0.522 | 0.772 |
| BoxTextured | 0.62 | 0.308 | 0.497 | 0.409 | 0.591 | 0.791 |
| BoxVertexColors | 0.62 | 0.001 | 0.419 | 0.073 | 0.756 | 0.826 |
| BrainStem | 0.05 | 0.040 | 0.114 | 0.217 | 0.513 | 0.525 |
| CesiumMan | 0.06 | 0.072 | 0.136 | 0.194 | 0.399 | 0.452 |
| CesiumMilkTruck | 0.21 | 0.204 | 0.322 | 0.348 | 0.481 | 0.669 |
| ChairDamaskPurplegold | 0.19 | 0.174 | 0.189 | 0.280 | 0.521 | 0.705 |
| Corset | 0.15 | 0.100 | 0.138 | 0.300 | 0.557 | 0.635 |
| DamagedHelmet | 0.25 | 0.223 | 0.287 | 0.530 | 0.575 | 0.798 |
| DiffuseTransmissionPlant | 0.16 | 0.065 | 0.125 | 0.403 | 0.573 | 0.727 |
| Duck | 0.22 | 0.025 | 0.040 | 0.135 | 0.425 | 0.795 |
| Fox | 0.06 | 0.018 | 0.059 | 0.153 | 0.454 | 0.577 |
| GlamVelvetSofa | 0.10 | 0.013 | 0.016 | 0.154 | 0.313 | 0.595 |
| IridescenceLamp | 0.20 | 0.080 | 0.131 | 0.234 | 0.429 | 0.627 |
| Lantern | 0.05 | 0.016 | 0.031 | 0.110 | 0.371 | 0.473 |
| MaterialsVariantsShoe | 0.10 | 0.119 | 0.181 | 0.246 | 0.460 | 0.610 |
| ProcDumbbell | 0.06 | 0.004 | 0.141 | 0.104 | 0.419 | 0.540 |
| ProcTable | 0.16 | 0.001 | 0.034 | 0.034 | 0.555 | 0.682 |
| ProcTorus | 0.16 | 0.051 | 0.277 | 0.158 | 0.432 | 0.728 |
| RiggedFigure | 0.08 | 0.000 | 0.000 | 0.143 | 0.402 | 0.453 |
| SheenChair | 0.19 | 0.027 | 0.099 | 0.190 | 0.527 | 0.744 |
| SunglassesKhronos | 0.07 | 0.020 | 0.048 | 0.117 | 0.395 | 0.510 |
| Suzanne | 0.13 | 0.009 | 0.019 | 0.243 | 0.419 | 0.653 |
| ToyCar | 0.17 | 0.048 | 0.107 | 0.184 | 0.460 | 0.613 |
| WaterBottle | 0.13 | 0.089 | 0.144 | 0.239 | 0.332 | 0.606 |

Spearman(coverage, full_alex) across models for v1 union at 128 B: +0.61; for the blank image: +0.91

Spearman(coverage, full_vgg) across models for v1 union at 128 B: +0.74; for the blank image: +0.93

Spearman(coverage, crop_alex) across models for v1 union at 128 B: +0.39; for the blank image: +0.91

Spearman(coverage, crop_vgg) across models for v1 union at 128 B: +0.51; for the blank image: +0.86

Spearman(coverage, crop_dists) across models for v1 union at 128 B: +0.12; for the blank image: +0.43

## The blank-image check

Number of models (of 28) on which a blank white image scores *better* than the baseline:

| metric | vs box | vs sphere | vs k-DOP 128 B | vs v1 union 64 B |
|---|---|---|---|---|
| full_alex | 16 | 0 | 0 | 0 |
| full_vgg | 25 | 13 | 7 | 0 |
| full_dists | 0 | 0 | 0 | 0 |
| crop_alex | 7 | 0 | 0 | 0 |
| crop_vgg | 23 | 20 | 12 | 0 |
| crop_dists | 0 | 0 | 0 | 0 |
| lp_alex | 7 | 0 | 0 | 0 |
| lp_vgg | 23 | 16 | 10 | 0 |
| lp_dists | 0 | 0 | 0 | 0 |
| blur_alex | 17 | 5 | 6 | 0 |
| blur_vgg | 24 | 17 | 11 | 0 |
| blur_dists | 0 | 0 | 0 | 0 |
| grey_alex | 8 | 0 | 0 | 0 |
| grey_vgg | 23 | 14 | 12 | 0 |
| grey_dists | 0 | 0 | 0 | 0 |
