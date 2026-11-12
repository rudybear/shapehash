# LPIPS to tune encoder choices

Tuning half = every other model alphabetically starting with the first (14), held-out half = the rest (14).

## vox: weight between geometry and colour in the grid / colour-mode search

The encoder keeps the candidate (grid size, colour cell size, colour mode) maximising `(1-w) * F-score + w * colour term`; it ships w = 0.5.
Each row re-selects with another w from the same candidate set, decodes (smooth) and renders. Mean over models, lower is better.


64 bytes

| w | tune crop_alex | tune crop_vgg | tune crop_dists | held-out crop_alex | held-out crop_vgg | held-out crop_dists | mean grid size | mean colour cell | same string as w=0.5 (of 28) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.2644 | 0.2584 | 0.2846 | 0.2931 | 0.2907 | 0.2853 | 17.9 | 1.54 | 0 |
| 0.15 | 0.2482 | 0.2413 | 0.2708 | 0.2460 | 0.2518 | 0.2657 | 17.1 | 4.00 | 17 |
| 0.30 | 0.2494 | 0.2413 | 0.2700 | 0.2418 | 0.2464 | 0.2599 | 17.1 | 3.82 | 24 |
| 0.40 | 0.2494 | 0.2413 | 0.2700 | 0.2430 | 0.2480 | 0.2604 | 17.2 | 3.86 | 25 |
| 0.50 (shipped) | 0.2503 | 0.2428 | 0.2714 | 0.2439 | 0.2482 | 0.2608 | 16.8 | 3.68 | 28 |
| 0.60 | 0.2506 | 0.2423 | 0.2711 | 0.2460 | 0.2498 | 0.2621 | 16.5 | 3.36 | 24 |
| 0.70 | 0.2506 | 0.2424 | 0.2711 | 0.2493 | 0.2518 | 0.2649 | 15.9 | 3.04 | 23 |
| 0.85 | 0.2550 | 0.2437 | 0.2720 | 0.2554 | 0.2547 | 0.2663 | 15.0 | 2.54 | 18 |
| 1.00 | 0.3057 | 0.2744 | 0.2879 | 0.2789 | 0.2692 | 0.2738 | 11.2 | 1.64 | 8 |

Per-model best w by `crop_alex` (ties go to 0.5): 0.00: 1, 0.15: 4, 0.30: 1, 0.40: 3, 0.50: 17, 0.60: 1, 0.85: 1; mean gain of a per-model oracle over w=0.5: 0.0059

Per-model best w by `crop_vgg` (ties go to 0.5): 0.00: 1, 0.15: 4, 0.30: 1, 0.40: 3, 0.50: 11, 0.60: 1, 0.70: 6, 0.85: 1; mean gain of a per-model oracle over w=0.5: 0.0042

Per-model best w by `crop_dists` (ties go to 0.5): 0.00: 2, 0.15: 1, 0.30: 1, 0.40: 3, 0.50: 14, 0.60: 1, 0.70: 1, 0.85: 2, 1.00: 3; mean gain of a per-model oracle over w=0.5: 0.0034

128 bytes

| w | tune crop_alex | tune crop_vgg | tune crop_dists | held-out crop_alex | held-out crop_vgg | held-out crop_dists | mean grid size | mean colour cell | same string as w=0.5 (of 28) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.2402 | 0.2422 | 0.2720 | 0.2681 | 0.2806 | 0.2792 | 26.1 | 1.00 | 1 |
| 0.15 | 0.2200 | 0.2243 | 0.2538 | 0.2085 | 0.2301 | 0.2503 | 25.8 | 3.54 | 19 |
| 0.30 | 0.2156 | 0.2206 | 0.2497 | 0.2075 | 0.2297 | 0.2504 | 25.0 | 2.86 | 23 |
| 0.40 | 0.2156 | 0.2206 | 0.2497 | 0.2082 | 0.2296 | 0.2515 | 24.7 | 3.36 | 24 |
| 0.50 (shipped) | 0.2148 | 0.2205 | 0.2498 | 0.2083 | 0.2290 | 0.2509 | 24.2 | 3.29 | 28 |
| 0.60 | 0.2179 | 0.2224 | 0.2526 | 0.2142 | 0.2317 | 0.2539 | 23.1 | 3.04 | 23 |
| 0.70 | 0.2221 | 0.2237 | 0.2507 | 0.2226 | 0.2355 | 0.2537 | 21.4 | 2.75 | 17 |
| 0.85 | 0.2298 | 0.2266 | 0.2552 | 0.2290 | 0.2385 | 0.2545 | 19.4 | 2.64 | 12 |
| 1.00 | 0.2522 | 0.2392 | 0.2618 | 0.2490 | 0.2523 | 0.2633 | 15.6 | 1.21 | 3 |

Per-model best w by `crop_alex` (ties go to 0.5): 0.00: 1, 0.30: 1, 0.40: 1, 0.50: 24, 1.00: 1; mean gain of a per-model oracle over w=0.5: 0.0017

Per-model best w by `crop_vgg` (ties go to 0.5): 0.40: 1, 0.50: 18, 0.70: 2, 0.85: 6, 1.00: 1; mean gain of a per-model oracle over w=0.5: 0.0015

Per-model best w by `crop_dists` (ties go to 0.5): 0.00: 2, 0.30: 1, 0.40: 1, 0.50: 15, 0.70: 4, 0.85: 2, 1.00: 3; mean gain of a per-model oracle over w=0.5: 0.0046

256 bytes

| w | tune crop_alex | tune crop_vgg | tune crop_dists | held-out crop_alex | held-out crop_vgg | held-out crop_dists | mean grid size | mean colour cell | same string as w=0.5 (of 28) |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 0.2166 | 0.2261 | 0.2587 | 0.2538 | 0.2713 | 0.2747 | 37.0 | 1.00 | 0 |
| 0.15 | 0.1838 | 0.1984 | 0.2339 | 0.1851 | 0.2169 | 0.2462 | 36.8 | 3.46 | 18 |
| 0.30 | 0.1833 | 0.1976 | 0.2335 | 0.1838 | 0.2156 | 0.2446 | 36.2 | 3.39 | 19 |
| 0.40 | 0.1833 | 0.1976 | 0.2335 | 0.1845 | 0.2154 | 0.2444 | 35.9 | 3.32 | 20 |
| 0.50 (shipped) | 0.1837 | 0.1949 | 0.2276 | 0.1858 | 0.2155 | 0.2442 | 33.9 | 2.86 | 28 |
| 0.60 | 0.1894 | 0.1984 | 0.2303 | 0.1839 | 0.2160 | 0.2436 | 31.9 | 2.00 | 19 |
| 0.70 | 0.1905 | 0.1991 | 0.2318 | 0.1858 | 0.2178 | 0.2450 | 30.8 | 1.89 | 16 |
| 0.85 | 0.1913 | 0.2000 | 0.2331 | 0.1937 | 0.2217 | 0.2454 | 29.5 | 1.82 | 11 |
| 1.00 | 0.2203 | 0.2181 | 0.2474 | 0.2160 | 0.2343 | 0.2527 | 22.2 | 1.14 | 4 |

Per-model best w by `crop_alex` (ties go to 0.5): 0.15: 1, 0.30: 1, 0.40: 4, 0.50: 18, 0.60: 3, 0.85: 1; mean gain of a per-model oracle over w=0.5: 0.0040

Per-model best w by `crop_vgg` (ties go to 0.5): 0.00: 1, 0.40: 3, 0.50: 19, 0.60: 1, 0.70: 3, 0.85: 1; mean gain of a per-model oracle over w=0.5: 0.0013

Per-model best w by `crop_dists` (ties go to 0.5): 0.00: 2, 0.40: 2, 0.50: 17, 0.60: 1, 0.85: 1, 1.00: 5; mean gain of a per-model oracle over w=0.5: 0.0024

## mesh: position precision (5-8 bits per component, more bits = fewer vertices)

| budget | metric | encoder keeps 5/6/7/8 bits | metric prefers 5/6/7/8 bits | models where they differ | oracle gain, tune | oracle gain, held-out |
|---|---|---|---|---|---|---|
| 64 | crop_alex | 21 / 3 / 4 / 0 | 22 / 4 / 2 / 0 | 9 | 0.0024 | 0.0005 |
| 64 | crop_vgg | 21 / 3 / 4 / 0 | 24 / 3 / 1 / 0 | 9 | 0.0008 | 0.0006 |
| 64 | crop_dists | 21 / 3 / 4 / 0 | 24 / 2 / 2 / 0 | 9 | 0.0012 | 0.0009 |
| 128 | crop_alex | 18 / 7 / 2 / 1 | 20 / 6 / 2 / 0 | 8 | 0.0015 | 0.0003 |
| 128 | crop_vgg | 18 / 7 / 2 / 1 | 17 / 10 / 1 / 0 | 8 | 0.0010 | 0.0013 |
| 128 | crop_dists | 18 / 7 / 2 / 1 | 19 / 9 / 0 / 0 | 14 | 0.0023 | 0.0014 |
| 256 | crop_alex | 17 / 8 / 2 / 1 | 16 / 7 / 5 / 0 | 11 | 0.0017 | 0.0030 |
| 256 | crop_vgg | 17 / 8 / 2 / 1 | 16 / 8 / 2 / 2 | 12 | 0.0032 | 0.0035 |
| 256 | crop_dists | 17 / 8 / 2 / 1 | 19 / 8 / 0 / 1 | 14 | 0.0017 | 0.0050 |

Mean over models of each fixed precision:

| budget | bits | mean vertices | crop_alex | crop_vgg | crop_dists |
|---|---|---|---|---|---|
| 64 | 5 | 12.6 | 0.3375 | 0.2865 | 0.3335 |
| 64 | 6 | 11.7 | 0.3469 | 0.2923 | 0.3388 |
| 64 | 7 | 11.0 | 0.3522 | 0.2953 | 0.3425 |
| 64 | 8 | 10.0 | 0.3639 | 0.3009 | 0.3508 |
| 128 | 5 | 25.5 | 0.2683 | 0.2473 | 0.2915 |
| 128 | 6 | 23.8 | 0.2736 | 0.2496 | 0.2935 |
| 128 | 7 | 22.1 | 0.2796 | 0.2538 | 0.2980 |
| 128 | 8 | 20.9 | 0.2841 | 0.2568 | 0.3011 |
| 256 | 5 | 50.4 | 0.2244 | 0.2195 | 0.2621 |
| 256 | 6 | 47.1 | 0.2267 | 0.2203 | 0.2637 |
| 256 | 7 | 43.9 | 0.2300 | 0.2221 | 0.2664 |
| 256 | 8 | 41.2 | 0.2328 | 0.2240 | 0.2690 |

Sheet: `tuning_vox_weight_128.png` (w = 0, 0.15, 0.5, 0.85, 1 for six models).

