| Model | Protocol | Fusion block | Seeds | Test clips | Accuracy % (mean +- std) | Macro-F1 (mean +- std) |
|---|---|---|---|---|---|---|
| speech | random | - | 3 | 560 | 99.8 +- 0.2 | 0.998 +- 0.002 |
| text | random | - | 3 | 560 | 13.8 +- 0.6 | 0.037 +- 0.002 |
| fusion | random_concat | concat | 3 | 560 | 99.9 +- 0.1 | 0.999 +- 0.001 |
| speech | spk_OAF2YAF | - | 3 | 1400 | 48.6 +- 8.1 | 0.446 +- 0.082 |
| speech | spk_YAF2OAF | - | 3 | 1400 | 40.2 +- 5.7 | 0.311 +- 0.039 |
| text | spk_OAF2YAF | - | 3 | 1400 | 14.3 +- 0.0 | 0.043 +- 0.009 |
| text | spk_YAF2OAF | - | 3 | 1400 | 14.3 +- 0.0 | 0.043 +- 0.009 |
| fusion | spk_OAF2YAF_concat | concat | 3 | 1400 | 51.9 +- 5.0 | 0.477 +- 0.085 |
| fusion | spk_YAF2OAF_concat | concat | 3 | 1400 | 44.0 +- 8.0 | 0.362 +- 0.091 |
| speech | word | - | 5 | 560 | 99.8 +- 0.2 | 0.998 +- 0.002 |
| text | word | - | 5 | 560 | 14.3 +- 0.0 | 0.038 +- 0.004 |
| fusion | word_concat | concat | 5 | 560 | 99.9 +- 0.1 | 0.999 +- 0.001 |
| fusion | word_cross | cross | 5 | 560 | 99.8 +- 0.3 | 0.998 +- 0.003 |
| fusion | word_gated | gated | 5 | 560 | 99.8 +- 0.2 | 0.998 +- 0.002 |

Per-emotion recall (mean over seeds):

| Model | Protocol | angry | disgust | fear | happy | neutral | pleasant_surprise | sad |
|---|---|---|---|---|---|---|---|---|
| speech | random | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 1.00 |
| text | random | 0.33 | 0.00 | 0.01 | 0.29 | 0.00 | 0.00 | 0.33 |
| fusion (concat) | random_concat | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
| speech | spk_OAF2YAF | 0.96 | 0.57 | 0.37 | 0.01 | 0.95 | 0.14 | 0.41 |
| speech | spk_YAF2OAF | 0.01 | 0.88 | 0.70 | 0.00 | 0.90 | 0.19 | 0.13 |
| text | spk_OAF2YAF | 0.00 | 0.00 | 0.33 | 0.00 | 0.62 | 0.05 | 0.00 |
| text | spk_YAF2OAF | 0.00 | 0.00 | 0.33 | 0.00 | 0.62 | 0.05 | 0.00 |
| fusion (concat) | spk_OAF2YAF_concat | 0.99 | 0.46 | 0.42 | 0.00 | 0.95 | 0.06 | 0.76 |
| fusion (concat) | spk_YAF2OAF_concat | 0.01 | 0.90 | 0.83 | 0.00 | 0.86 | 0.17 | 0.31 |
| speech | word | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| text | word | 0.19 | 0.00 | 0.20 | 0.40 | 0.01 | 0.20 | 0.00 |
| fusion (concat) | word_concat | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| fusion (cross) | word_cross | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 | 0.99 | 1.00 |
| fusion (gated) | word_gated | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
