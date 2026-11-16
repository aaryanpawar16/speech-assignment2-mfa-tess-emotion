# Error analysis (speech, protocol spk_OAF2YAF, seeds [0, 1, 2])

Total failures over 3 seeds: 2160. Most frequent confusions (true -> predicted): happy->angry (594), pleasant_surprise->angry (268), sad->angry (227), fear->angry (172), fear->happy (165).

## Failure 1: YAF, word 'yes', true **happy**, predicted **angry** (confidence 0.986, seed 1)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, happy (true) | 1.57 | 285 | 176 | -15.5 | 10.4 |
| same word, angry (predicted) | 1.95 | 253 | 146 | -19.1 | 11.8 |

Class probabilities: angry 0.99, disgust 0.00, fear 0.00, happy 0.00, neutral 0.00, pleasant_surprise 0.00, sad 0.00

## Failure 2: YAF, word 'read', true **pleasant_surprise**, predicted **angry** (confidence 0.960, seed 0)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, pleasant_surprise (true) | 1.70 | 359 | 441 | -17.3 | 9.8 |
| same word, angry (predicted) | 2.14 | 229 | 127 | -20.0 | 10.0 |

Class probabilities: angry 0.96, disgust 0.01, fear 0.00, happy 0.01, neutral 0.01, pleasant_surprise 0.01, sad 0.01

## Failure 3: YAF, word 'peg', true **sad**, predicted **angry** (confidence 0.976, seed 1)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, sad (true) | 2.27 | 213 | 104 | -20.0 | 11.1 |
| same word, angry (predicted) | 1.92 | 232 | 174 | -17.8 | 13.2 |

Class probabilities: angry 0.98, disgust 0.00, fear 0.00, happy 0.00, neutral 0.00, pleasant_surprise 0.01, sad 0.00

## Failure 4: YAF, word 'take', true **fear**, predicted **angry** (confidence 0.977, seed 0)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, fear (true) | 1.68 | 318 | 180 | -17.8 | 10.9 |
| same word, angry (predicted) | 2.08 | 239 | 85 | -19.1 | 14.5 |

Class probabilities: angry 0.98, disgust 0.00, fear 0.00, happy 0.01, neutral 0.01, pleasant_surprise 0.00, sad 0.00

## Failure 5: YAF, word 'sell', true **fear**, predicted **happy** (confidence 0.947, seed 0)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, fear (true) | 1.60 | 331 | 207 | -15.0 | 11.2 |
| same word, happy (predicted) | 1.66 | 277 | 190 | -13.8 | 11.8 |

Class probabilities: angry 0.02, disgust 0.01, fear 0.01, happy 0.95, neutral 0.01, pleasant_surprise 0.01, sad 0.01

