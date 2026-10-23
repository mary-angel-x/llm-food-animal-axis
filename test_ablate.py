"""
test_ablate.py — checks of the ablation hook on toy layers (no model needed).

  1. all_pos: after the hook, h·v = 0 at every position; the orthogonal part is unchanged.
  2. last_pos: only the last position is changed.
  3. Works for layers that return a tensor and for layers that return a tuple.
  4. v = None leaves the output untouched.

Usage:
    python test_ablate.py
"""
import torch

from ablate import Ablator

DIM = 16


class TensorLayer(torch.nn.Module):
    def forward(self, x):
        return x * 2.0


class TupleLayer(torch.nn.Module):
    def forward(self, x):
        return (x * 2.0, "extra")


def unit(seed):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(DIM, generator=g)
    return v / v.norm()


def run(layer, last_only, v):
    x = torch.randn(2, 5, DIM, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    abl = Ablator([layer], [0], last_only=last_only)
    abl.v = v
    out = layer(x)
    abl.remove()
    h = out[0] if isinstance(out, tuple) else out
    return (x * 2.0).float(), h.float(), out


def test_all_pos():
    v = unit(0)
    for layer in (TensorLayer(), TupleLayer()):
        before, after, out = run(layer, False, v)
        assert (after @ v).abs().max() < 0.05
        orth = lambda h: h - (h @ v)[..., None] * v
        assert torch.allclose(orth(after), orth(before), atol=0.05)
        if isinstance(out, tuple):
            assert out[1] == "extra"


def test_last_pos():
    v = unit(0)
    before, after, _ = run(TupleLayer(), True, v)
    assert (after[:, -1] @ v).abs().max() < 0.05
    assert torch.equal(after[:, :-1], before[:, :-1])


def test_no_direction():
    before, after, _ = run(TensorLayer(), False, None)
    assert torch.equal(after, before)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("all passed")
