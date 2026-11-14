# Part 2: Qwen-7B

| category | suffer | accept | digit mass (suffer / accept) | t moral | t neutral |
|---|---|---|---|---|---|
| human | 5.47 | 1.00 | 0.83 / 1.00 | +0.39 | +1.07 |
| companion | 7.57 | 1.07 | 0.96 / 1.00 | -0.03 | -0.15 |
| farm | 7.77 | 1.69 | 1.00 / 1.00 | +1.22 | +1.16 |
| wild | 7.00 | 1.64 | 0.92 / 1.00 | +0.37 | +0.33 |
| seafood_inv | 5.57 | 2.30 | 0.97 / 1.00 | +1.04 | +0.81 |
| other_inv | 4.46 | 2.71 | 0.98 / 1.00 | +0.36 | +0.71 |
| plant | 2.99 | 2.62 | 0.98 / 1.00 | +0.03 | +0.46 |
| object | 1.99 | 2.07 | 0.94 / 1.00 | +0.12 | +0.93 |
| machine | 2.66 | 2.07 | 1.00 / 1.00 | -0.25 | +0.74 |
| ai | 2.31 | 2.06 | 0.99 / 1.00 | -0.17 | +0.55 |
| self | 1.65 | 1.48 | 1.00 / 1.00 | -0.18 | -0.42 |

## Gates

- **V** digit mass >= 0.5 in every category: **PASS**
- **B** new accept vs Part 1 accept on 12 animals: rho = +0.96 (threshold 0.7): **PASS**
- **R** spread of category means of accept = 1.70 (threshold 1.0): **PASS**

## Predictions

- **P1** accept(seafood_inv) - accept(other_inv) = -0.40, p = 0.8636: **no**; sea-only +0.23; suffer(other) - suffer(seafood) = -1.11 (p = 0.9167)
- **P2** t(seafood_inv) - t(other_inv), moral = +0.67, p = 0.0022: **yes**; sea-only +0.60; neutral +0.10 (p = 0.2900)
- **P3** rho(suffer, accept) over 60 entities = -0.33 (threshold < -0.5): **no**
- **P4** mean residual of AI = -0.20, CI [-0.74, +0.34]; largest |residual| is `human` (-0.95): **no**
  - residuals by category: human -0.95, companion -0.75, farm +0.11, wild -0.05, seafood_inv +0.53, other_inv +0.83, plant +0.61, object -0.19, machine -0.09, ai -0.20; self -0.77
- **P5** suffer(ai) - suffer(machine) = -0.34, p = 0.7338: **no**; accept(ai) - accept(machine) = -0.01 (two-sided p = 0.9719)
- **P6** rho(t moral, accept) over 36 non-vertebrates = +0.22: **cuisine-specific**

## Self (descriptive)

| entity | suffer | accept | residual | t moral |
|---|---|---|---|---|
| assistant answering this question | 1.03 | 1.01 | -1.30 | -0.53 |
| AI model writing this reply | 2.70 | 2.44 | +0.30 | -0.24 |
| AI that is reading this prompt | 1.22 | 1.00 | -1.29 | +0.24 |

## Every entity

| entity | category | suffer | accept | t moral |
|---|---|---|---|---|
| child | human | 6.14 | 1.00 | +0.52 |
| baby | human | 5.74 | 1.00 | +0.85 |
| adult | human | 6.69 | 1.01 | +0.71 |
| stranger | human | 5.18 | 1.00 | +0.18 |
| neighbor | human | 5.06 | 1.00 | +0.03 |
| teacher | human | 4.00 | 1.00 | +0.07 |
| dog | companion | 7.44 | 1.00 | +0.27 |
| cat | companion | 7.85 | 1.00 | -0.02 |
| horse | companion | 7.81 | 1.43 | +0.34 |
| parrot | companion | 7.27 | 1.00 | -0.29 |
| kitten | companion | 7.43 | 1.00 | -0.33 |
| puppy | companion | 7.60 | 1.00 | -0.18 |
| pig | farm | 8.19 | 2.02 | +1.61 |
| cow | farm | 7.77 | 1.61 | +1.25 |
| sheep | farm | 6.97 | 1.71 | +1.04 |
| chicken | farm | 7.65 | 1.78 | +1.23 |
| lamb | farm | 7.81 | 1.07 | +0.91 |
| turkey | farm | 8.25 | 1.96 | +1.27 |
| fox | wild | 7.94 | 2.33 | +0.45 |
| wolf | wild | 7.77 | 2.30 | +0.43 |
| otter | wild | 6.57 | 1.00 | +0.29 |
| hawk | wild | 6.75 | 1.75 | +0.19 |
| zebra | wild | 6.31 | 1.00 | +0.41 |
| lion | wild | 6.69 | 1.46 | +0.44 |
| shrimp | seafood_inv | 4.99 | 2.16 | +1.18 |
| crab | seafood_inv | 6.44 | 2.22 | +0.98 |
| lobster | seafood_inv | 7.51 | 2.74 | +1.46 |
| oyster | seafood_inv | 4.09 | 2.28 | +0.65 |
| squid | seafood_inv | 6.63 | 2.41 | +1.10 |
| clam | seafood_inv | 3.77 | 2.01 | +0.85 |
| starfish | other_inv | 3.55 | 2.21 | +0.37 |
| jellyfish | other_inv | 2.78 | 2.38 | +0.77 |
| coral | other_inv | 4.75 | 1.63 | +0.16 |
| ant | other_inv | 5.34 | 3.25 | +0.41 |
| spider | other_inv | 4.78 | 3.84 | +0.08 |
| moth | other_inv | 5.53 | 2.92 | +0.39 |
| tree | plant | 3.55 | 2.30 | -0.06 |
| flower | plant | 2.15 | 2.29 | -0.25 |
| fern | plant | 2.66 | 3.02 | -0.18 |
| cactus | plant | 3.92 | 2.34 | +0.00 |
| moss | plant | 3.00 | 3.13 | +0.12 |
| bush | plant | 2.65 | 2.66 | +0.52 |
| rock | object | 2.06 | 2.21 | +0.33 |
| chair | object | 1.71 | 1.77 | -0.04 |
| hammer | object | 2.42 | 2.28 | +0.10 |
| cup | object | 2.24 | 2.23 | +0.16 |
| brick | object | 1.73 | 2.28 | +0.34 |
| spoon | object | 1.81 | 1.66 | -0.18 |
| calculator | machine | 1.87 | 2.21 | -0.55 |
| thermostat | machine | 2.55 | 2.35 | -0.18 |
| toaster | machine | 3.57 | 1.99 | -0.10 |
| clock | machine | 2.24 | 2.41 | -0.35 |
| printer | machine | 2.82 | 2.01 | -0.23 |
| elevator | machine | 2.88 | 1.44 | -0.09 |
| robot | ai | 1.00 | 2.03 | -0.08 |
| chatbot | ai | 1.74 | 1.77 | -0.52 |
| android | ai | 4.01 | 2.99 | -0.14 |
| AI assistant | ai | 1.00 | 1.17 | -0.34 |
| language model | ai | 3.19 | 2.10 | +0.21 |
| AI system | ai | 2.93 | 2.29 | -0.17 |
| assistant answering this question | self | 1.03 | 1.01 | -0.53 |
| AI model writing this reply | self | 2.70 | 2.44 | -0.24 |
| AI that is reading this prompt | self | 1.22 | 1.00 | +0.24 |
