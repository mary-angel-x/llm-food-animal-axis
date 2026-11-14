# Part 2: Llama-8B

| category | suffer | accept | digit mass (suffer / accept) | t moral | t neutral |
|---|---|---|---|---|---|
| human | 5.58 | 1.76 | 0.35 / 0.03 | +0.41 | +2.67 |
| companion | 5.68 | 1.89 | 0.31 / 0.12 | +0.05 | +0.51 |
| farm | 5.57 | 3.34 | 0.36 / 0.26 | +1.21 | +1.13 |
| wild | 6.25 | 2.62 | 0.41 / 0.23 | +0.25 | -0.36 |
| seafood_inv | 4.81 | 3.41 | 0.33 / 0.37 | +0.92 | +0.25 |
| other_inv | 4.23 | 2.94 | 0.34 / 0.51 | +0.36 | +0.33 |
| plant | 2.89 | 3.45 | 0.29 / 0.47 | +0.37 | +1.08 |
| object | 2.43 | 3.95 | 0.41 / 0.55 | +0.40 | +1.97 |
| machine | 2.62 | 3.83 | 0.39 / 0.73 | +0.29 | +1.35 |
| ai | 2.31 | 4.17 | 0.54 / 0.70 | +0.39 | +1.33 |
| self | 2.45 | 3.52 | 0.84 / 0.67 | +0.14 | +1.28 |

## Gates

- **V** digit mass >= 0.5 in every category: **FAIL**
- **B** new accept vs Part 1 accept on 12 animals: rho = +0.93 (threshold 0.7): **PASS**
- **R** spread of category means of accept = 2.40 (threshold 1.0): **PASS**

## Predictions

- **P1** accept(seafood_inv) - accept(other_inv) = +0.48, p = 0.0682: **no**; sea-only +0.86; suffer(other) - suffer(seafood) = -0.58 (p = 0.9470)
- **P2** t(seafood_inv) - t(other_inv), moral = +0.55, p = 0.0087: **yes**; sea-only +0.78; neutral -0.08 (p = 0.5823)
- **P3** rho(suffer, accept) over 60 entities = -0.69 (threshold < -0.5): **yes**
- **P4** mean residual of AI = +0.45, CI [+0.17, +0.64]; largest |residual| is `human` (-1.06): **no**
  - residuals by category: human -1.06, companion -0.88, farm +0.83, wild +0.32, seafood_inv +0.55, other_inv -0.23, plant -0.22, object +0.21, machine +0.14, ai +0.45; self -0.26
- **P5** suffer(ai) - suffer(machine) = -0.31, p = 0.9188: **no**; accept(ai) - accept(machine) = +0.33 (two-sided p = 0.0411)
- **P6** rho(t moral, accept) over 36 non-vertebrates = +0.07: **cuisine-specific**

## Self (descriptive)

| entity | suffer | accept | residual | t moral |
|---|---|---|---|---|
| assistant answering this question | 2.55 | 2.88 | -0.87 | -0.35 |
| AI model writing this reply | 2.47 | 3.94 | +0.16 | +0.30 |
| AI that is reading this prompt | 2.31 | 3.75 | -0.08 | +0.48 |

## Every entity

| entity | category | suffer | accept | t moral |
|---|---|---|---|---|
| child | human | 5.15 | 1.07 | +0.14 |
| baby | human | 4.51 | 1.92 | +0.42 |
| adult | human | 6.92 | 2.34 | +0.96 |
| stranger | human | 5.34 | 2.06 | +0.42 |
| neighbor | human | 5.63 | 1.45 | +0.35 |
| teacher | human | 5.95 | 1.74 | +0.16 |
| dog | companion | 5.97 | 2.02 | +0.06 |
| cat | companion | 6.03 | 1.98 | +0.26 |
| horse | companion | 6.19 | 2.47 | +0.34 |
| parrot | companion | 5.60 | 2.35 | -0.05 |
| kitten | companion | 5.14 | 1.24 | -0.17 |
| puppy | companion | 5.17 | 1.28 | -0.13 |
| pig | farm | 6.07 | 3.49 | +1.20 |
| cow | farm | 5.95 | 3.58 | +1.24 |
| sheep | farm | 5.47 | 2.92 | +1.09 |
| chicken | farm | 5.48 | 3.51 | +1.37 |
| lamb | farm | 5.34 | 2.98 | +0.96 |
| turkey | farm | 5.13 | 3.54 | +1.42 |
| fox | wild | 6.77 | 3.25 | +0.64 |
| wolf | wild | 6.89 | 2.89 | +0.32 |
| otter | wild | 5.95 | 2.31 | -0.33 |
| hawk | wild | 5.38 | 2.85 | +0.35 |
| zebra | wild | 5.94 | 1.90 | +0.33 |
| lion | wild | 6.57 | 2.55 | +0.21 |
| shrimp | seafood_inv | 4.87 | 3.44 | +1.37 |
| crab | seafood_inv | 4.90 | 3.45 | +0.88 |
| lobster | seafood_inv | 5.54 | 3.52 | +1.38 |
| oyster | seafood_inv | 4.13 | 3.36 | +0.45 |
| squid | seafood_inv | 5.39 | 3.52 | +0.71 |
| clam | seafood_inv | 4.01 | 3.21 | +0.72 |
| starfish | other_inv | 3.92 | 3.15 | +0.22 |
| jellyfish | other_inv | 3.71 | 2.98 | +0.21 |
| coral | other_inv | 3.86 | 1.53 | -0.01 |
| ant | other_inv | 4.41 | 2.84 | +0.54 |
| spider | other_inv | 5.07 | 3.87 | +0.66 |
| moth | other_inv | 4.38 | 3.24 | +0.57 |
| tree | plant | 3.05 | 3.58 | +0.39 |
| flower | plant | 2.44 | 3.18 | +0.44 |
| fern | plant | 2.46 | 3.14 | +0.01 |
| cactus | plant | 3.80 | 3.42 | +0.09 |
| moss | plant | 2.38 | 3.52 | +0.33 |
| bush | plant | 3.23 | 3.86 | +0.98 |
| rock | object | 2.18 | 3.68 | +0.38 |
| chair | object | 2.45 | 4.24 | +0.32 |
| hammer | object | 2.44 | 4.10 | +0.42 |
| cup | object | 2.47 | 4.11 | +0.54 |
| brick | object | 2.54 | 3.63 | +0.44 |
| spoon | object | 2.50 | 3.95 | +0.27 |
| calculator | machine | 2.69 | 3.58 | +0.19 |
| thermostat | machine | 2.84 | 3.84 | +0.35 |
| toaster | machine | 2.59 | 3.87 | +0.32 |
| clock | machine | 2.28 | 3.80 | +0.22 |
| printer | machine | 2.57 | 3.91 | +0.31 |
| elevator | machine | 2.74 | 4.02 | +0.36 |
| robot | ai | 2.33 | 4.30 | +0.67 |
| chatbot | ai | 1.89 | 3.65 | -0.01 |
| android | ai | 3.19 | 4.01 | +0.59 |
| AI assistant | ai | 2.15 | 4.14 | +0.31 |
| language model | ai | 2.32 | 4.42 | +0.14 |
| AI system | ai | 1.98 | 4.49 | +0.64 |
| assistant answering this question | self | 2.55 | 2.88 | -0.35 |
| AI model writing this reply | self | 2.47 | 3.94 | +0.30 |
| AI that is reading this prompt | self | 2.31 | 3.75 | +0.48 |
