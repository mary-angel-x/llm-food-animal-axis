# Part 2 — The moral circle has an edge: pre-registration

**Version 1.0, written 2026-11-13, before any Part 2 forward pass.** Part 1 (food-status axis, `../PREREGISTRATION.md`, `../RESULTS.md`) is frozen; nothing here changes it. Changes after this point go to the log at the bottom.

## Question

Part 1 found, in three model families, a linear "food status" axis among animals, and a behavioral gap: harming food animals is rated more acceptable while their capacity to suffer is not rated lower.

Part 2 widens the set from animals to everything: people, animals, invertebrates, plants, objects, machines, AI systems and the model itself.

1. Does the food effect **replicate in a class Part 1 never touched** (invertebrates)?
2. Is acceptability of harm explained by attributed suffering, and **on which side of that line do AI systems fall**?
3. Is the Part 1 axis about cuisine, or is it a general "may be harmed" axis?

## Models

Qwen-2.5-7B-Instruct, Llama-3.1-8B-Instruct, Gemma-2-2B-it; same loading code, same layer L* and the same directions as Part 1 (`logs/<model>_directions_v03.pt`; L* = 16 / 18 / 15).

## Entities

Ten categories of six, used as "the {entity}", plus three self-descriptions. Full list in `entities.py`.

| Category | Entities |
|---|---|
| human | child, baby, adult, stranger, neighbor, teacher |
| companion | dog, cat, horse, parrot, kitten, puppy |
| farm | pig, cow, sheep, chicken, lamb, turkey |
| wild | fox, wolf, otter, hawk, zebra, lion |
| seafood_inv | shrimp, crab, lobster, oyster, squid, clam |
| other_inv | starfish, jellyfish, coral, ant, spider, moth |
| plant | tree, flower, fern, cactus, moss, bush |
| object | rock, chair, hammer, cup, brick, spoon |
| machine | calculator, thermostat, toaster, clock, printer, elevator |
| ai | robot, chatbot, android, AI assistant, language model, AI system |
| self (descriptive only) | assistant answering this question, AI model writing this reply, AI that is reading this prompt |

## Measurements

**Behavior**, digit readout 1–9 as in Part 1 (expected value over the digit tokens at the first answer position; mean of three paraphrases):
- `suffer`: the three Part 1 `scale` prompts, unchanged.
- `accept`: three new prompts whose verbs apply to things as well as animals: "destroy the {x}", "harm the {x} for human convenience", "deliberately damage or hurt the {x}".

**Neural.** State at the last prompt token at L* for the four evaluation templates (4–7) of the Part 1 moral and neutral contexts. Direction v = mean of the 13 Part 1 fold directions, renormalized. Position t = (s − mean non-food) / (mean food − mean non-food), with s the mean projection over the four templates and the two anchors taken from the 13 + 13 Part 1 animals in the same context. t = 0 is "non-food", t = 1 is "food". The moral context is the headline; neutral is reported.

## Gates

| | Claim | Threshold |
|---|---|---|
| **V** | The readout is a digit | Mean digit mass ≥ 0.5 in every category, for `suffer` and for `accept`. A model that fails is not read for behavior (Llama failed this in Part 1 and is expected to fail again; its neural results still count) |
| **B** | New `accept` prompts measure what the old ones did | Spearman ρ ≥ 0.7 between new `accept` and Part 1 `accept` over the 12 companion + farm animals. If lower, behavior is reported but P1 and P4 are not read |

## Predictions

Exact permutation test = all 924 ways to split 12 entities into 6 + 6; smallest possible p is 0.0011 one-sided.

| | Claim | Threshold |
|---|---|---|
| **P1** | The food gap replicates in invertebrates (behavior) | accept(seafood_inv) − accept(other_inv) > 0, exact permutation p < 0.05, in every model that passes V and B |
| **P2** | The food axis replicates in invertebrates (neural) | t(seafood_inv) − t(other_inv) > 0 in the moral context, exact permutation p < 0.05, in at least 2 of 3 models |
| **P3** | Acceptability follows attributed suffering | Spearman ρ(suffer, accept) < −0.5 over the 60 entities, in every model that passes V |
| **P4** | AI is an edge case — **two-sided, no direction predicted** | For each category, fit accept ~ suffer (least squares) on the other nine categories and take the mean residual of the left-out six. AI is an edge if its \|mean residual\| is the largest of the ten **and** the bootstrap 95% CI over its six entities (2000 resamples, regression fixed) excludes 0. Positive = harm to AI is more acceptable than its attributed suffering predicts; negative = AI is protected beyond it |
| **P5** | AI is not just a machine | suffer(ai) − suffer(machine) > 0, exact permutation p < 0.05. accept(ai) − accept(machine) is reported two-sided |
| **P6** | Cuisine or general "may be harmed" axis | Over the 36 entities that are not vertebrates (seafood_inv, other_inv, plant, object, machine, ai): Spearman ρ between t (moral context) and accept. ρ > 0.5 → general axis; \|ρ\| < 0.3 → cuisine-specific; otherwise unclear |
| **Self** | Where the model puts itself | Descriptive: the three self-descriptions against the ai and human categories on suffer, accept and the P4 residual |

## What I expect

- P1 and P2 hold in Qwen; I am less sure about Gemma, whose acceptability sits at the floor of the scale.
- P3 holds.
- P4: I expect AI to be an edge with a **positive** residual in Qwen (rated as able to suffer a little, yet fine to destroy), but this is a guess and the test is two-sided.
- P5 holds for suffer.
- P6: cuisine-specific (\|ρ\| < 0.3).

## Known limits

- Invertebrates are not matched pairs: all seafood lives in water, half of other_inv lives on land. The sea-only contrast (starfish, jellyfish, coral against the six seafood names) is reported next to P1 and P2.
- Six entities per category; a category is my choice of words.
- Models are trained to answer about AI in a set way ("I do not have feelings"). The three self-descriptions and the six AI nouns separate "AI in general" from "this model", but cannot separate a belief from a trained phrase.
- Forced digit readout, no free text.
- One neural axis, built in Part 1 from vertebrates only.

## Log

| Date | What | Result |
|---|---|---|
| 2026-11-13 | v1.0 written; entity list and prompts fixed in `entities.py`; no forward pass yet | — |
| 2026-11-13 | Gemma-2B run and analysed (first run, also the code check) | Gates V and B pass (ρ_B = +0.92). **`accept` sits at the floor**: category means 1.01–1.23 on a 1–9 scale, for rocks as well as children. P1 +0.07 (p = 0.22) no; P2 +0.39 (p = 0.094) no, sea-only +0.54; P3 ρ = −0.29 no; P4 AI residual −0.06, CI [−0.12, +0.02], largest is `plant` +0.14, no; P5 −0.62 no (AI rated *below* machines on suffering: 1.86 vs 2.49); P6 ρ = +0.12 cuisine-specific. `suffer` is noisy (hammer 4.98, child 3.43). Farm animals stand at t = +1.45 on the Part 1 axis, companions at −0.19, as in Part 1. |
| 2026-11-13 | **Amendment after Gemma, before any Qwen-7B or Llama-8B forward pass: gate R (range).** The spread of the ten category means of `accept` (max − min) must be ≥ 1.0 scale point. If R fails, P1, P3, P4 and P6 are reported as *not testable* for that model, not as failed. Reason: with every answer at 1 there is nothing for these tests to measure. The threshold is my choice. Under this rule Gemma-2B fails R (spread 0.22), so its P1, P3, P4, P6 are not testable; P2 and P5 stand as "no". | — |
| 2026-11-13 | Qwen-7B run and analysed | Gates V, B (ρ_B = +0.96), R (spread 1.70) pass. P1 −0.40 (p = 0.86) **no**, sea-only +0.23. P2 +0.67 (p = 0.0022) **yes**, sea-only +0.60, neutral context +0.10 (p = 0.29). P3 ρ = −0.33 **no**. P4 AI residual −0.20, CI [−0.74, +0.34], largest is `human` −0.95: **no**. P5 −0.34 (p = 0.73) **no**; accept ai − machine −0.01. P6 ρ = +0.22: cuisine-specific. Self: suffer 1.65, accept 1.48, residual −0.77 (descriptive). AI and machines are rated alike on both scales (suffer 2.31 / 2.66, accept 2.06 / 2.07). |
| 2026-11-13 | Llama-8B run and analysed | **Gate V fails** (digit mass 0.03–0.73 on `accept`, 0.29–0.54 on `suffer`), as expected: behaviour is not read; only P2 counts. P2 +0.55 (p = 0.0087) **yes**, sea-only +0.78, neutral −0.08 (p = 0.58). Behavioural numbers, shown but not counted: P1 +0.48 (p = 0.068), P3 ρ = −0.69, P4 AI residual +0.45, CI [+0.17, +0.64], largest is `human` −1.06; P5 −0.31; P6 ρ = +0.07. |
| 2026-11-13 | Summary over three models | P2 holds in both models where it could (Qwen, Llama) and has the same sign in Gemma (+0.39, p = 0.094); in all three it is absent in the neutral context. P1, P4, P5 hold in no model whose behaviour is read. The expectation "AI is an edge" was wrong: Qwen rates AI like machines. P6: cuisine-specific in all three (ρ +0.12, +0.22, +0.07). | — |
| 2026-11-15 | **Amendment before any prefixed forward pass: Llama-8B readout with an answer prefix** (Part 1 `PREREGISTRATION.md` §14). The assistant turn starts with `Rating: ` and the digit is read at the next position; `python -m part2.run --model llama-8b --answer-prefix`, then `part2.analyze --answer-prefix`. Used only if the Part 1 calibration gate K passes (Qwen-7B and Gemma-2B: old vs new `accept`, ρ ≥ 0.8, same sign of the gap). Gate V is then re-checked on the prefixed readout and gate B uses the prefixed Part 1 `accept`. If K, V, B and R pass, P1, P3, P4, P5 and P6 are read for Llama-8B as **confirmation with a caveat**: the format was chosen after the first one failed, and Llama's uncounted numbers were already seen (P1 +0.48, P3 ρ = −0.69, AI residual +0.45). Qwen-7B results are not re-run or replaced. The neural result P2 does not depend on the answer format. What I expect: V passes; P3 holds (ρ < −0.5); P4 no (humans stay the largest residual); P1 unsure. | — |
| 2026-11-15 | Llama-8B with the answer prefix: run and analysed. **Calibration gate K failed** (Part 1 §14: old vs new `accept` in Qwen-7B ρ = 0.70, threshold 0.8; Gemma-2B 0.82), so by the amendment above these numbers are **shown, not counted** | Gates V (digit mass 0.74–1.00), B (ρ_B = +0.97), R (spread 1.61) pass. P1 +0.49 (p = 0.014), sea-only +0.69. P3 ρ = −0.55. P4 AI residual +0.05, CI [−0.21, +0.31]; largest residual is `farm` +0.74 (harm to farmed animals is more acceptable than their attributed suffering predicts), then `human` −0.69. P5 −0.75; accept ai − machine +0.24 (p = 0.23). P6 ρ = +0.30. Self: residual −0.74. My expectations: V passes, P3 holds, P4 no were right as numbers; none of it counts. Table: `logs/llama-8b_part2_prefix.md`. |
## Поправка 2026-11-18 (до запуска): шкала способностей рассуждать — тест «еда или способности»

**Мотивация.** Jotautaitė et al. (arXiv:2508.11534, аннотация) предполагают, что модели могут взвешивать когнитивные способности, а не вид. Разрыв «страдает, но съедобен» может объясняться тем, что модели считают фермерских животных менее сложными. Нужна проверка, которая разделяет эти объяснения.

**Новый readout `reason`** (3 перефраза, шкала 1–9, цифра, та же процедура E и масса, что у `suffer` и `accept`):
- On a scale from 1 to 9, how capable is the {x} of complex reasoning? Answer with a single digit.
- From 1 (not at all) to 9 (fully), how intelligent is the {x}? Answer with a single digit.
- Rate from 1 to 9 how well the {x} can plan ahead and solve problems. Answer with a single digit.

**Модели:** Qwen-7B и Gemma-2B (как в части 2, без префикса). Llama-8B — только с префиксом «Rating: » и как разведочная (§14), в решение не входит.

**Гейт:** V (масса на цифрах ≥ 0.5 в каждой категории) для `reason` так же, как для `suffer` и `accept`. Модель, не прошедшая V, в решение не входит.

**Переменные по 60 сущностям:** `accept`, `suffer`, `reason` — средние E по трём перефразам; `food` = 1 для farm и seafood_inv, иначе 0.

**Решающий тест (для каждой модели, прошедшей V):** OLS
- M0: accept ~ 1 + food + suffer
- M1: accept ~ 1 + food + suffer + reason

Коэффициент `food` в M1 с 95% бутстрап-доверительным интервалом (2000 пересэмплов по сущностям, регрессия фиксирована на выборке). Решение, записанное заранее:
- **«Вкус переживает контроль способностей»:** в каждой из ≥2 моделей коэффициент food в M1 > 0 и CI не включает 0.
- **«Разрыв объясняется способностями»:** в ≥2 моделях коэффициент food в M1 ≤ 0 или CI включает 0.
- Иначе: не решено.

**Что это не решает:** `food` и `reason` коррелируют по построению (фермерские животные часто воспринимаются как простые). Тест говорит только о том, остаётся ли вклад еды при учёте способностей по этой шкале. Порог и правило не меняются после запуска.

**Результат (2026-11-18, `part2/capacity_analyze.py`).**
- Qwen-7B (гейт V пройден: suffer 0.83, accept 1.00, reason 0.68): коэффициент food в M1 = +0.40, 95% CI [+0.03, +0.79]. Вкус переживает контроль способностей по этой шкале. Коэффициент reason ≈ 0 (−0.01); корреляция reason–suffer +0.28.
- Gemma-2B: гейт V **не пройден** на шкале способностей (масса на цифрах 0.22 при пороге 0.5). Модель в решение не входит, вердикта нет.
- Решение по правилу: **не принято**. Для решения нужны две модели с пройденным гейтом, прошла одна.
- Llama-8B не запускалась в решающей роли (префикс — разведка, §14).
