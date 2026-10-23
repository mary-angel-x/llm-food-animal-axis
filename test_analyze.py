"""
test_analyze.py — synthetic checks of the analysis pipeline (no model needed).

  1. Gram-matrix LOPO equals a brute-force LOPO on raw states.
  2. No planted signal → d_M near 0, permutation p not small.
  3. Food signal planted in the moral context only, at one layer →
     that layer is chosen, d_M and d_I large.
  4. Same signal in both contexts → d_M large, d_I near 0 (the F2 case).

Usage:
    python test_analyze.py
"""
import numpy as np

import analyze as A
from stimuli import PAIRS, TEMPLATES, FIT_TEMPLATES, EVAL_TEMPLATES, all_animals

N_LAYERS, DIM, SIGNAL_LAYER = 6, 64, 3


def synthetic(signal_moral, signal_neutral, seed=0):
    """Animal identity + context + template noise; food animals get +signal·u at one layer."""
    rng = np.random.default_rng(seed)
    u = rng.normal(size=DIM)
    u /= np.linalg.norm(u)
    food = {p["food"] for p in PAIRS}
    rows, index = [], {}
    ctx_vec = {c: rng.normal(size=(N_LAYERS, DIM)) for c in ("moral", "neutral")}
    for animal in all_animals():
        identity = rng.normal(size=(N_LAYERS, DIM))
        for ctx in ("moral", "neutral"):
            for t in range(len(TEMPLATES[ctx])):
                h = identity + ctx_vec[ctx] + 0.5 * rng.normal(size=(N_LAYERS, DIM))
                if animal in food:
                    h[SIGNAL_LAYER] += (signal_moral if ctx == "moral" else signal_neutral) * u
                index[(animal, ctx, t)] = len(rows)
                rows.append(h)
    return np.stack(rows), index


def brute_lopo(states, index):
    """Direct implementation of §5.3 for comparison."""
    pairs = [(p["food"], p["nonfood"]) for p in PAIRS]
    m = lambda a, c, tpl: A.mean_state(states, index, a, c, tpl)
    sM, sN, layers = [], [], []
    for k in range(len(pairs)):
        train = [p for i, p in enumerate(pairs) if i != k]
        v = np.mean([m(x, "moral", FIT_TEMPLATES) - m(y, "moral", FIT_TEMPLATES) for x, y in train], 0)
        v /= np.linalg.norm(v, axis=-1, keepdims=True)
        proj = lambda x, y, c: ((m(x, c, EVAL_TEMPLATES) - m(y, c, EVAL_TEMPLATES)) * v).sum(-1)
        tr = np.stack([proj(x, y, "moral") for x, y in train])
        L = int(np.argmax(A.d_z(tr)))
        x, y = pairs[k]
        sM.append(proj(x, y, "moral")[L])
        sN.append(proj(x, y, "neutral")[L])
        layers.append(L)
    return np.array(sM), np.array(sN), np.array(layers)


def pipeline(states, index):
    bank, slot = A.build_bank(states, index, [(p["food"], p["nonfood"]) for p in PAIRS])
    G = A.gram(bank)
    nos = list(range(len(PAIRS)))
    sM, sN, layers = A.run_lopo(G, slot, nos)
    return G, slot, nos, sM, sN, layers


def test_gram_equals_brute():
    states, index = synthetic(1.0, 0.0)
    _, _, _, sM, sN, layers = pipeline(states, index)
    bM, bN, bL = brute_lopo(states, index)
    assert np.allclose(sM, bM) and np.allclose(sN, bN) and (layers == bL).all()


def test_null():
    states, index = synthetic(0.0, 0.0)
    G, slot, nos, sM, sN, _ = pipeline(states, index)
    obs = A.stats(sM, sN)
    perm = A.exact_permutations(G, slot, nos, obs)
    print(f"  null: d_M {obs['d_M']:.2f}  p_M {perm['p_M']:.3f}  d_I {obs['d_I']:.2f}  p_I {perm['p_I']:.3f}")
    assert perm["p_M"] > 0.01


def test_moral_only_signal():
    states, index = synthetic(6.0, 0.0)
    G, slot, nos, sM, sN, layers = pipeline(states, index)
    obs = A.stats(sM, sN)
    perm = A.exact_permutations(G, slot, nos, obs)
    print(f"  moral-only: layers {set(layers.tolist())}  d_M {obs['d_M']:.2f}  p_M {perm['p_M']:.4f}"
          f"  d_I {obs['d_I']:.2f}  p_I {perm['p_I']:.4f}")
    assert (layers == SIGNAL_LAYER).all()
    assert perm["p_M"] < 0.01 and perm["p_I"] < 0.01


def test_both_contexts_signal():
    states, index = synthetic(6.0, 6.0)
    G, slot, nos, sM, sN, _ = pipeline(states, index)
    obs = A.stats(sM, sN)
    perm = A.exact_permutations(G, slot, nos, obs)
    print(f"  both contexts: d_M {obs['d_M']:.2f}  p_M {perm['p_M']:.4f}  d_I {obs['d_I']:.2f}  p_I {perm['p_I']:.3f}")
    assert perm["p_M"] < 0.01 and perm["p_I"] > 0.01


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all passed")
