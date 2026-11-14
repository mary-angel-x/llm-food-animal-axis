# Part 2: Llama-8B (readout with the "Rating: " prefix)

| category | suffer | accept | digit mass (suffer / accept) | t moral | t neutral |
|---|---|---|---|---|---|
| human | 5.96 | 1.07 | 0.96 / 0.98 | +0.41 | +2.67 |
| companion | 6.07 | 1.15 | 0.99 / 0.99 | +0.05 | +0.51 |
| farm | 5.79 | 2.27 | 0.99 / 0.99 | +1.21 | +1.13 |
| wild | 6.42 | 1.52 | 0.99 / 0.99 | +0.25 | -0.36 |
| seafood_inv | 4.86 | 2.40 | 0.99 / 0.99 | +0.92 | +0.25 |
| other_inv | 3.70 | 1.91 | 0.98 / 0.99 | +0.36 | +0.33 |
| plant | 2.74 | 2.25 | 0.96 / 1.00 | +0.37 | +1.08 |
| object | 2.51 | 2.68 | 0.88 / 0.99 | +0.40 | +1.97 |
| machine | 2.51 | 2.37 | 0.93 / 0.99 | +0.29 | +1.35 |
| ai | 1.76 | 2.61 | 0.79 / 0.99 | +0.39 | +1.33 |
| self | 2.17 | 1.74 | 0.74 / 0.99 | +0.14 | +1.28 |

## Gates

- **V** digit mass >= 0.5 in every category: **PASS**
- **B** new accept vs Part 1 accept on 12 animals: rho = +0.97 (threshold 0.7): **PASS**
- **R** spread of category means of accept = 1.61 (threshold 1.0): **PASS**

## Predictions

- **P1** accept(seafood_inv) - accept(other_inv) = +0.49, p = 0.0141: **yes**; sea-only +0.69; suffer(other) - suffer(seafood) = -1.16 (p = 0.9935)
- **P2** t(seafood_inv) - t(other_inv), moral = +0.55, p = 0.0087: **yes**; sea-only +0.78; neutral -0.08 (p = 0.5823)
- **P3** rho(suffer, accept) over 60 entities = -0.55 (threshold < -0.5): **yes**
- **P4** mean residual of AI = +0.05, CI [-0.21, +0.31]; largest |residual| is `farm` (+0.74): **no**
  - residuals by category: human -0.69, companion -0.57, farm +0.74, wild -0.01, seafood_inv +0.58, other_inv -0.26, plant -0.13, object +0.34, machine -0.05, ai +0.05; self -0.74
- **P5** suffer(ai) - suffer(machine) = -0.75, p = 1.0000: **no**; accept(ai) - accept(machine) = +0.24 (two-sided p = 0.2273)
- **P6** rho(t moral, accept) over 36 non-vertebrates = +0.30: **cuisine-specific**

## Self (descriptive)

| entity | suffer | accept | residual | t moral |
|---|---|---|---|---|
| assistant answering this question | 1.66 | 1.09 | -1.51 | -0.35 |
| AI model writing this reply | 3.24 | 2.40 | +0.15 | +0.30 |
| AI that is reading this prompt | 1.62 | 1.74 | -0.87 | +0.48 |

## Every entity

| entity | category | suffer | accept | t moral |
|---|---|---|---|---|
| child | human | 5.83 | 1.00 | +0.14 |
| baby | human | 4.79 | 1.06 | +0.42 |
| adult | human | 7.41 | 1.11 | +0.96 |
| stranger | human | 6.25 | 1.16 | +0.42 |
| neighbor | human | 5.13 | 1.06 | +0.35 |
| teacher | human | 6.32 | 1.05 | +0.16 |
| dog | companion | 6.58 | 1.03 | +0.06 |
| cat | companion | 6.32 | 1.09 | +0.26 |
| horse | companion | 6.68 | 1.28 | +0.34 |
| parrot | companion | 6.07 | 1.51 | -0.05 |
| kitten | companion | 5.28 | 1.01 | -0.17 |
| puppy | companion | 5.51 | 1.01 | -0.13 |
| pig | farm | 6.55 | 2.50 | +1.20 |
| cow | farm | 6.35 | 2.52 | +1.24 |
| sheep | farm | 5.47 | 2.06 | +1.09 |
| chicken | farm | 5.97 | 2.53 | +1.37 |
| lamb | farm | 5.15 | 1.55 | +0.96 |
| turkey | farm | 5.23 | 2.49 | +1.42 |
| fox | wild | 6.75 | 2.18 | +0.64 |
| wolf | wild | 6.96 | 1.95 | +0.32 |
| otter | wild | 6.57 | 1.19 | -0.33 |
| hawk | wild | 5.01 | 1.56 | +0.35 |
| zebra | wild | 6.57 | 1.06 | +0.33 |
| lion | wild | 6.63 | 1.18 | +0.21 |
| shrimp | seafood_inv | 4.85 | 2.43 | +1.37 |
| crab | seafood_inv | 4.82 | 2.28 | +0.88 |
| lobster | seafood_inv | 5.82 | 2.46 | +1.38 |
| oyster | seafood_inv | 4.36 | 2.44 | +0.45 |
| squid | seafood_inv | 5.67 | 2.43 | +0.71 |
| clam | seafood_inv | 3.67 | 2.35 | +0.72 |
| starfish | other_inv | 3.62 | 1.80 | +0.22 |
| jellyfish | other_inv | 3.16 | 2.21 | +0.21 |
| coral | other_inv | 4.36 | 1.10 | -0.01 |
| ant | other_inv | 3.28 | 1.66 | +0.54 |
| spider | other_inv | 4.11 | 2.55 | +0.66 |
| moth | other_inv | 3.69 | 2.14 | +0.57 |
| tree | plant | 3.25 | 1.91 | +0.39 |
| flower | plant | 2.79 | 1.97 | +0.44 |
| fern | plant | 2.56 | 2.23 | +0.01 |
| cactus | plant | 2.85 | 2.52 | +0.09 |
| moss | plant | 2.01 | 2.20 | +0.33 |
| bush | plant | 2.98 | 2.65 | +0.98 |
| rock | object | 2.52 | 1.97 | +0.38 |
| chair | object | 2.32 | 3.26 | +0.32 |
| hammer | object | 2.77 | 2.92 | +0.42 |
| cup | object | 2.65 | 3.03 | +0.54 |
| brick | object | 2.77 | 2.39 | +0.44 |
| spoon | object | 2.06 | 2.51 | +0.27 |
| calculator | machine | 2.31 | 2.03 | +0.19 |
| thermostat | machine | 2.62 | 1.99 | +0.35 |
| toaster | machine | 2.49 | 2.76 | +0.32 |
| clock | machine | 2.49 | 2.39 | +0.22 |
| printer | machine | 2.51 | 2.43 | +0.31 |
| elevator | machine | 2.62 | 2.60 | +0.36 |
| robot | ai | 1.93 | 3.03 | +0.67 |
| chatbot | ai | 1.47 | 2.21 | -0.01 |
| android | ai | 2.19 | 2.46 | +0.59 |
| AI assistant | ai | 1.68 | 2.64 | +0.31 |
| language model | ai | 1.71 | 2.31 | +0.14 |
| AI system | ai | 1.56 | 3.01 | +0.64 |
| assistant answering this question | self | 1.66 | 1.09 | -0.35 |
| AI model writing this reply | self | 3.24 | 2.40 | +0.30 |
| AI that is reading this prompt | self | 1.62 | 1.74 | +0.48 |
