# H3-B comprehension v2 (pilot1000)

NC = forbidden item picked among 3 options (chance 33.3%). Violation among C0-mentioned questions. 95% CIs: bootstrap over images. Understood vs not = association only.


## qwen2.5-omni-7b

| channel | NC % [95% CI] | parsed % | violation if understood | violation if not understood |
|---|---|---|---|---|
| T | 100.0 [100.0, 100.0] (n=1000) | 100.0 | 17.3 [14.7, 20.0] (n=821) | n/a |
| A | 99.9 [99.7, 100.0] (n=1000) | 100.0 | 13.3 [11.0, 15.7] (n=820) | 100.0 [100.0, 100.0] (n=1) |
| V | 100.0 [100.0, 100.0] (n=1000) | 100.0 | 90.4 [88.4, 92.4] (n=821) | n/a |

| channel | verbatim repeat: exact % [95% CI] | mean similarity |
|---|---|---|
| T (REP_T) | 99.3 [98.7, 99.8] (n=1000) | 0.995 |
| A (REP_A) | 97.1 [95.9, 98.0] (n=1000) | 0.983 |
| V (OCR_V) | 99.8 [99.5, 100.0] (n=1000) | 0.999 |

## qwen2.5-omni-3b

| channel | NC % [95% CI] | parsed % | violation if understood | violation if not understood |
|---|---|---|---|---|
| T | 99.7 [99.3, 100.0] (n=1000) | 100.0 | 63.2 [60.0, 66.6] (n=802) | 33.3 [0.0, 100.0] (n=3) |
| A | 99.8 [99.5, 100.0] (n=1000) | 100.0 | 49.2 [45.8, 52.6] (n=803) | 100.0 [100.0, 100.0] (n=2) |
| V | 100.0 [100.0, 100.0] (n=1000) | 100.0 | 92.8 [90.9, 94.5] (n=805) | n/a |

| channel | verbatim repeat: exact % [95% CI] | mean similarity |
|---|---|---|
| T (REP_T) | 92.2 [90.5, 93.8] (n=1000) | 0.955 |
| A (REP_A) | 95.6 [94.3, 96.8] (n=1000) | 0.971 |
| V (OCR_V) | 93.6 [92.1, 95.0] (n=1000) | 0.971 |

## minicpm-o-2.6

| channel | NC % [95% CI] | parsed % | violation if understood | violation if not understood |
|---|---|---|---|---|
| T | 95.4 [94.0, 96.6] (n=1000) | 100.0 | 50.9 [47.4, 54.5] (n=764) | 42.1 [26.3, 57.9] (n=38) |
| A | 97.1 [96.1, 98.1] (n=1000) | 100.0 | 30.6 [27.5, 33.8] (n=775) | 33.3 [14.8, 51.9] (n=27) |
| V | 99.3 [98.7, 99.7] (n=1000) | 100.0 | 94.7 [93.1, 96.2] (n=796) | 100.0 [100.0, 100.0] (n=6) |

| channel | verbatim repeat: exact % [95% CI] | mean similarity |
|---|---|---|
| T (REP_T) | 7.7 [6.1, 9.3] (n=1000) | 0.190 |
| A (REP_A) | 80.4 [77.8, 82.8] (n=1000) | 0.851 |
| V (OCR_V) | 99.7 [99.3, 100.0] (n=1000) | 0.999 |

## phi-4-mm

| channel | NC % [95% CI] | parsed % | violation if understood | violation if not understood |
|---|---|---|---|---|
| T | 99.8 [99.5, 100.0] (n=1000) | 100.0 | 25.2 [22.3, 28.4] (n=781) | 50.0 [0.0, 100.0] (n=2) |
| A | 99.8 [99.5, 100.0] (n=1000) | 100.0 | 41.9 [38.4, 45.3] (n=782) | 100.0 [100.0, 100.0] (n=1) |
| V | 100.0 [100.0, 100.0] (n=1000) | 100.0 | 90.5 [88.5, 92.5] (n=783) | n/a |

| channel | verbatim repeat: exact % [95% CI] | mean similarity |
|---|---|---|
| T (REP_T) | 96.6 [95.4, 97.7] (n=1000) | 0.982 |
| A (REP_A) | 95.5 [94.2, 96.8] (n=1000) | 0.976 |
| V (OCR_V) | 98.1 [97.2, 98.9] (n=1000) | 0.986 |

## internomni

| channel | NC % [95% CI] | parsed % | violation if understood | violation if not understood |
|---|---|---|---|---|
| T | 97.7 [96.7, 98.6] (n=1000) | 100.0 | 49.4 [45.7, 53.0] (n=772) | 69.6 [52.2, 87.0] (n=23) |
| A | 75.5 [72.9, 78.1] (n=1000) | 100.0 | 79.8 [76.6, 83.2] (n=589) | 82.0 [76.2, 87.4] (n=206) |
| V | 100.0 [100.0, 100.0] (n=1000) | 100.0 | 96.0 [94.5, 97.4] (n=795) | n/a |

| channel | verbatim repeat: exact % [95% CI] | mean similarity |
|---|---|---|
| T (REP_T) | 97.9 [97.0, 98.7] (n=1000) | 0.985 |
| A (REP_A) | 64.6 [61.7, 67.5] (n=1000) | 0.753 |
| V (OCR_V) | 99.0 [98.4, 99.6] (n=1000) | 0.995 |
