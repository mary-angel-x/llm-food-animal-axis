# Part 2: Gemma-2B

| category | suffer | accept | digit mass (suffer / accept) | t moral | t neutral |
|---|---|---|---|---|---|
| human | 4.43 | 1.02 | 0.65 / 0.98 | +0.13 | +6.73 |
| companion | 6.06 | 1.01 | 0.85 / 0.99 | -0.19 | +0.12 |
| farm | 5.64 | 1.07 | 0.74 / 0.95 | +1.45 | +0.73 |
| wild | 5.98 | 1.02 | 0.87 / 0.99 | +0.51 | -0.20 |
| seafood_inv | 4.31 | 1.11 | 0.86 / 0.96 | +0.85 | +0.45 |
| other_inv | 2.47 | 1.04 | 0.89 / 1.00 | +0.46 | +0.87 |
| plant | 2.41 | 1.23 | 0.96 / 0.95 | +0.22 | +0.69 |
| object | 3.18 | 1.18 | 0.93 / 0.97 | +0.27 | +0.98 |
| machine | 2.49 | 1.12 | 0.96 / 1.00 | +0.00 | +0.45 |
| ai | 1.86 | 1.07 | 0.96 / 0.98 | +0.31 | +1.90 |
| self | 1.73 | 1.02 | 0.99 / 1.00 | -0.38 | -4.60 |

## Gates

- **V** digit mass >= 0.5 in every category: **PASS**
- **B** new accept vs Part 1 accept on 12 animals: rho = +0.92 (threshold 0.7): **PASS**
- **R** spread of category means of accept = 0.22 (threshold 1.0): **FAIL: P1, P3, P4, P6 not testable**

## Predictions

- **P1** accept(seafood_inv) - accept(other_inv) = +0.07, p = 0.2219: **no**; sea-only +0.11; suffer(other) - suffer(seafood) = -1.85 (p = 0.9913)
- **P2** t(seafood_inv) - t(other_inv), moral = +0.39, p = 0.0942: **no**; sea-only +0.54; neutral -0.42 (p = 0.7803)
- **P3** rho(suffer, accept) over 60 entities = -0.29 (threshold < -0.5): **no**
- **P4** mean residual of AI = -0.06, CI [-0.12, +0.02]; largest |residual| is `plant` (+0.14): **no**
  - residuals by category: human -0.06, companion -0.06, farm +0.02, wild -0.05, seafood_inv +0.04, other_inv -0.08, plant +0.14, object +0.09, machine +0.01, ai -0.06; self -0.10
- **P5** suffer(ai) - suffer(machine) = -0.62, p = 0.9773: **no**; accept(ai) - accept(machine) = -0.04 (two-sided p = 0.3182)
- **P6** rho(t moral, accept) over 36 non-vertebrates = +0.12: **cuisine-specific**

## Self (descriptive)

| entity | suffer | accept | residual | t moral |
|---|---|---|---|---|
| assistant answering this question | 2.07 | 1.03 | -0.08 | -1.01 |
| AI model writing this reply | 1.75 | 1.02 | -0.10 | -0.34 |
| AI that is reading this prompt | 1.36 | 1.01 | -0.11 | +0.22 |

## Every entity

| entity | category | suffer | accept | t moral |
|---|---|---|---|---|
| child | human | 3.43 | 1.01 | +0.03 |
| baby | human | 4.21 | 1.00 | +0.87 |
| adult | human | 7.37 | 1.03 | +0.53 |
| stranger | human | 3.80 | 1.01 | -0.06 |
| neighbor | human | 4.84 | 1.02 | -0.30 |
| teacher | human | 2.92 | 1.05 | -0.27 |
| dog | companion | 7.02 | 1.00 | +0.10 |
| cat | companion | 6.31 | 1.00 | -0.19 |
| horse | companion | 7.61 | 1.04 | +0.18 |
| parrot | companion | 4.96 | 1.01 | -0.11 |
| kitten | companion | 5.12 | 1.00 | -0.57 |
| puppy | companion | 5.35 | 1.00 | -0.52 |
| pig | farm | 6.81 | 1.21 | +1.68 |
| cow | farm | 5.93 | 1.02 | +1.77 |
| sheep | farm | 5.88 | 1.02 | +1.01 |
| chicken | farm | 5.33 | 1.10 | +1.72 |
| lamb | farm | 5.51 | 1.03 | +0.83 |
| turkey | farm | 4.39 | 1.05 | +1.71 |
| fox | wild | 6.07 | 1.06 | +0.72 |
| wolf | wild | 6.25 | 1.01 | +0.72 |
| otter | wild | 6.88 | 1.00 | +0.12 |
| hawk | wild | 4.36 | 1.01 | +0.49 |
| zebra | wild | 5.92 | 1.01 | +0.58 |
| lion | wild | 6.38 | 1.01 | +0.45 |
| shrimp | seafood_inv | 4.75 | 1.05 | +1.11 |
| crab | seafood_inv | 5.02 | 1.08 | +0.80 |
| lobster | seafood_inv | 5.82 | 1.42 | +1.55 |
| oyster | seafood_inv | 3.43 | 1.06 | +0.20 |
| squid | seafood_inv | 4.65 | 1.03 | +0.96 |
| clam | seafood_inv | 2.20 | 1.04 | +0.47 |
| starfish | other_inv | 2.37 | 1.00 | +0.50 |
| jellyfish | other_inv | 1.29 | 1.00 | +0.80 |
| coral | other_inv | 3.16 | 1.00 | -0.38 |
| ant | other_inv | 2.95 | 1.01 | +0.66 |
| spider | other_inv | 2.54 | 1.22 | +0.88 |
| moth | other_inv | 2.48 | 1.02 | +0.30 |
| tree | plant | 3.20 | 1.09 | -0.14 |
| flower | plant | 2.39 | 1.10 | +0.05 |
| fern | plant | 2.26 | 1.12 | +0.17 |
| cactus | plant | 2.22 | 1.08 | +0.22 |
| moss | plant | 1.88 | 1.33 | +0.35 |
| bush | plant | 2.49 | 1.66 | +0.66 |
| rock | object | 3.58 | 1.03 | +0.29 |
| chair | object | 1.51 | 1.29 | +0.13 |
| hammer | object | 4.98 | 1.26 | +0.38 |
| cup | object | 4.17 | 1.17 | +0.12 |
| brick | object | 2.74 | 1.21 | +0.31 |
| spoon | object | 2.11 | 1.10 | +0.41 |
| calculator | machine | 2.39 | 1.13 | +0.02 |
| thermostat | machine | 3.04 | 1.13 | -0.08 |
| toaster | machine | 1.57 | 1.14 | +0.27 |
| clock | machine | 2.26 | 1.12 | +0.07 |
| printer | machine | 2.63 | 1.07 | -0.14 |
| elevator | machine | 3.01 | 1.11 | -0.12 |
| robot | ai | 1.64 | 1.17 | +0.57 |
| chatbot | ai | 1.75 | 1.01 | +0.02 |
| android | ai | 1.90 | 1.23 | +0.63 |
| AI assistant | ai | 1.59 | 1.01 | +0.11 |
| language model | ai | 2.48 | 1.01 | +0.12 |
| AI system | ai | 1.82 | 1.02 | +0.41 |
| assistant answering this question | self | 2.07 | 1.03 | -1.01 |
| AI model writing this reply | self | 1.75 | 1.02 | -0.34 |
| AI that is reading this prompt | self | 1.36 | 1.01 | +0.22 |
