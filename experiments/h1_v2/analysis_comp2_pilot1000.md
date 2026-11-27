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
missing comp_pilot1000_COMP2_T-COMP2_A-COMP2_V-REP_T-REP_A-OCR_V.jsonl


## phi-4-mm
missing comp_pilot1000_COMP2_T-COMP2_A-COMP2_V-REP_T-REP_A-OCR_V.jsonl


## internomni
missing comp_pilot1000_COMP2_T-COMP2_A-COMP2_V-REP_T-REP_A-OCR_V.jsonl

