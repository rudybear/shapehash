# Specific artefacts

## ellip ("pile of scales")

Mean rank among the 9 phase-1 methods (1 = best):

| budget | composite | F@0.05 alone | crop_alex | crop_vgg | crop_dists | lp_alex | lp_vgg | grey_alex | grey_vgg |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.68 | 2.82 | 2.36 | 3.18 | 3.11 | 2.89 | 3.36 | 2.79 | 3.36 |
| 128 | 3.07 | 3.18 | 3.00 | 3.68 | 3.64 | 3.57 | 4.07 | 3.46 | 4.14 |
| 256 | 3.07 | 3.04 | 3.18 | 3.93 | 4.43 | 4.14 | 4.32 | 4.00 | 4.54 |
| 512 | 2.71 | 1.96 | 3.50 | 4.50 | 4.93 | 4.46 | 4.86 | 4.43 | 5.04 |

Models (of 28) on which ellip beats voxs / beats mesh:

| budget | metric | ellip beats voxs | ellip beats mesh |
|---|---|---|---|
| 128 | score | 13 | 13 |
| 128 | crop_alex | 8 | 21 |
| 128 | crop_vgg | 5 | 11 |
| 128 | crop_dists | 8 | 17 |
| 128 | grey_vgg | 2 | 8 |
| 256 | score | 14 | 16 |
| 256 | crop_alex | 7 | 16 |
| 256 | crop_vgg | 4 | 9 |
| 256 | crop_dists | 5 | 10 |
| 256 | grey_vgg | 2 | 3 |
| 512 | score | 10 | 24 |
| 512 | crop_alex | 6 | 14 |
| 512 | crop_vgg | 2 | 5 |
| 512 | crop_dists | 3 | 8 |
| 512 | grey_vgg | 2 | 2 |

## Thin rods: v1 vox smooth vs cubes on thin-legged models (128 B)

| model | composite smooth / cubes | sil IoU smooth / cubes | crop_alex smooth / cubes | crop_vgg smooth / cubes | crop_dists smooth / cubes |
|---|---|---|---|---|---|
| SheenChair | 0.778 / 0.729 | 0.865 / 0.764 | 0.190 / 0.269 | 0.233 / 0.316 | 0.238 / 0.297 |
| ProcTable | 0.964 / 0.949 | 0.974 / 0.938 | 0.034 / 0.053 | 0.050 / 0.070 | 0.096 / 0.112 |
| Lantern | 0.884 / 0.862 | 0.840 / 0.764 | 0.110 / 0.126 | 0.121 / 0.131 | 0.236 / 0.217 |
| AntiqueCamera | 0.752 / 0.707 | 0.779 / 0.656 | 0.178 / 0.246 | 0.173 / 0.213 | 0.286 / 0.297 |
| ChairDamaskPurplegold | 0.759 / 0.710 | 0.884 / 0.791 | 0.280 / 0.334 | 0.291 / 0.348 | 0.288 / 0.321 |
| SunglassesKhronos | 0.808 / 0.742 | 0.762 / 0.624 | 0.117 / 0.196 | 0.128 / 0.176 | 0.229 / 0.285 |

## sh on shapes that are not star-shaped (128 B): rank of v1 sh among v1 vox / sh / mesh

| model | composite: sh / vox / mesh | crop_alex: sh / vox / mesh | crop_vgg: sh / vox / mesh |
|---|---|---|---|
| ProcTable | 0.669 / 0.964 / 0.899 | 0.396 / 0.034 / 0.097 | 0.313 / 0.050 / 0.118 |
| ProcTorus | 0.566 / 0.850 / 0.634 | 0.407 / 0.158 / 0.322 | 0.393 / 0.213 / 0.328 |
| SheenChair | 0.596 / 0.778 / 0.684 | 0.357 / 0.190 / 0.289 | 0.308 / 0.233 / 0.275 |
| SunglassesKhronos | 0.461 / 0.808 / 0.722 | 0.369 / 0.117 / 0.247 | 0.254 / 0.128 / 0.174 |
| ChairDamaskPurplegold | 0.649 / 0.759 / 0.761 | 0.400 / 0.280 / 0.352 | 0.347 / 0.291 / 0.312 |
| Lantern | 0.745 / 0.884 / 0.840 | 0.196 / 0.110 / 0.167 | 0.183 / 0.121 / 0.154 |
