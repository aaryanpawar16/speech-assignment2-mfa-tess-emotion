# Error analysis (speech, protocol spk_YAF2OAF, seeds [0, 1, 2])

Total failures over 3 seeds: 2510. Most frequent confusions (true -> predicted): sad->disgust (397), angry->pleasant_surprise (304), happy->pleasant_surprise (283), pleasant_surprise->neutral (219), happy->fear (206).

## Failure 1: OAF, word 'came', true **sad**, predicted **disgust** (confidence 0.983, seed 0)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, sad (true) | 2.49 | 186 | 88 | -16.4 | 9.5 |
| same word, disgust (predicted) | 2.29 | 193 | 102 | -17.8 | 9.5 |

Class probabilities: angry 0.00, disgust 0.98, fear 0.00, happy 0.00, neutral 0.01, pleasant_surprise 0.00, sad 0.01

## Failure 2: OAF, word 'raise', true **angry**, predicted **pleasant_surprise** (confidence 0.968, seed 1)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, angry (true) | 1.60 | 248 | 111 | -14.9 | 10.8 |
| same word, pleasant_surprise (predicted) | 2.05 | 233 | 220 | -14.6 | 8.2 |

Class probabilities: angry 0.00, disgust 0.01, fear 0.01, happy 0.00, neutral 0.00, pleasant_surprise 0.97, sad 0.01

## Failure 3: OAF, word 'pass', true **happy**, predicted **pleasant_surprise** (confidence 0.972, seed 1)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, happy (true) | 2.28 | 291 | 301 | -15.2 | 9.9 |
| same word, pleasant_surprise (predicted) | 2.30 | 219 | 230 | -14.9 | 8.4 |

Class probabilities: angry 0.00, disgust 0.01, fear 0.01, happy 0.00, neutral 0.00, pleasant_surprise 0.97, sad 0.00

## Failure 4: OAF, word 'name', true **pleasant_surprise**, predicted **neutral** (confidence 0.988, seed 0)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, pleasant_surprise (true) | 1.81 | 221 | 237 | -13.3 | 6.6 |
| same word, neutral (predicted) | 2.00 | 188 | 23 | -11.7 | 6.8 |

Class probabilities: angry 0.00, disgust 0.00, fear 0.00, happy 0.00, neutral 0.99, pleasant_surprise 0.00, sad 0.00

## Failure 5: OAF, word 'fat', true **happy**, predicted **fear** (confidence 0.970, seed 2)

| | duration (s) | mean F0 (Hz) | F0 range 5-95% (Hz) | level (dB) | level variation (dB) |
|---|---|---|---|---|---|
| same word, happy (true) | 1.82 | 263 | 235 | -16.3 | 8.9 |
| same word, fear (predicted) | 1.60 | 288 | 116 | -14.0 | 11.5 |

Class probabilities: angry 0.00, disgust 0.01, fear 0.97, happy 0.00, neutral 0.01, pleasant_surprise 0.00, sad 0.01

