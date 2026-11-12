# A network-free selection score that follows LPIPS

Features are point-based (30 000 samples per side): F-score at three thresholds, colour term, normal consistency, 3-view point-splat silhouette IoU at three
resolutions. Each selector picks among v1 vox / sh / mesh at 64, 128 and 256 B; columns report the picked strings' mean crop LPIPS / DISTS, how often the pick equals
the LPIPS-AlexNet pick, and the phase-1 composite. The form was chosen on the tuning half; the held-out half is the test.

| selector | tune alex | tune = alex pick | tune vgg | tune dists | tune composite | held alex | held = alex pick | held vgg | held dists | held composite |
|---|---|---|---|---|---|---|---|---|---|---|
| F@0.05 + colour + sil24 (before this experiment) | 0.2237 | 35/42 | 0.2208 | 0.2579 | 0.7999 | 0.2089 | 33/42 | 0.2211 | 0.2572 | 0.8345 |
| F@0.05 + colour + sil24 + normal consistency (ships now) | 0.2185 | 40/42 | 0.2192 | 0.2529 | 0.7981 | 0.2052 | 36/42 | 0.2193 | 0.2524 | 0.8335 |
| F@0.025 + colour + sil24 | 0.2391 | 22/42 | 0.2289 | 0.2732 | 0.7952 | 0.2131 | 27/42 | 0.2225 | 0.2595 | 0.8316 |
| F@0.025 + colour + sil48 + nc | 0.2324 | 27/42 | 0.2250 | 0.2653 | 0.7942 | 0.2087 | 31/42 | 0.2204 | 0.2558 | 0.8337 |
| colour + sil48 + nc (no F-score) | 0.2169 | 41/42 | 0.2184 | 0.2506 | 0.7972 | 0.2062 | 34/42 | 0.2246 | 0.2487 | 0.8249 |
| sil48 only | 0.2278 | 31/42 | 0.2252 | 0.2617 | 0.7910 | 0.2117 | 27/42 | 0.2213 | 0.2518 | 0.8191 |
| normal consistency only | 0.2415 | 29/42 | 0.2282 | 0.2650 | 0.7854 | 0.2037 | 37/42 | 0.2172 | 0.2484 | 0.8283 |
| always vox | 0.2163 | 40/42 | 0.2194 | 0.2496 | 0.7959 | 0.2127 | 31/42 | 0.2309 | 0.2520 | 0.8197 |
| LPIPS-alex itself (upper bound for agreement) | 0.2159 | 42/42 | 0.2179 | 0.2499 | 0.7973 | 0.2006 | 42/42 | 0.2173 | 0.2463 | 0.8308 |

Exhaustive search over integer weights 0-3 for the 8 features (65535 combinations), best on the tuning half (ties: fewest total weight):

| weights f0.05/f0.025/f0.0125/col/nc/sil24/sil48/sil96 | tune alex | tune = alex pick | held alex | held = alex pick | held vgg | held dists | held composite |
|---|---|---|---|---|---|---|---|
| 0/0/0/1/1/1/0/0 | 0.2159 | 42/42 | 0.2041 | 35/42 | 0.2224 | 0.2472 | 0.8254 |
| 0/0/0/1/1/1/1/0 | 0.2159 | 42/42 | 0.2034 | 35/42 | 0.2194 | 0.2465 | 0.8273 |
| 0/0/0/1/2/1/0/0 | 0.2159 | 42/42 | 0.2041 | 35/42 | 0.2224 | 0.2472 | 0.8254 |
| 0/0/0/1/2/2/0/0 | 0.2159 | 42/42 | 0.2025 | 36/42 | 0.2189 | 0.2452 | 0.8271 |
| 0/0/0/1/3/1/0/0 | 0.2159 | 42/42 | 0.2020 | 37/42 | 0.2200 | 0.2466 | 0.8280 |
| 0/0/0/2/1/2/0/0 | 0.2159 | 42/42 | 0.2119 | 33/42 | 0.2302 | 0.2534 | 0.8217 |
