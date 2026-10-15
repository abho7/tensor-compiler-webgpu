"""The PyTorch oracle. Offline only: CI never runs this, it checks the output.

Reads golden/cases.json, rebuilds every graph op for op in PyTorch float64,
and records the outputs and autograd's gradients for each case's cotangents.
For the softmax composites it also records torch.softmax / torch.log_softmax
applied directly to the input, so our decomposition is checked against
PyTorch's own implementation rather than against a translation of itself.

    .venv/Scripts/python oracle/torch_golden.py
"""
import json
import pathlib
import platform

import torch

torch.set_default_dtype(torch.float64)
ROOT = pathlib.Path(__file__).resolve().parent.parent
F64 = torch.float64


def t(values, shape):
    return torch.tensor(values, dtype=F64).reshape(shape)


def run_graph(graph, leaves):
    vals = []
    for n in graph["nodes"]:
        op, ins, a = n["op"], [vals[i] for i in n["inputs"]], n["attrs"]
        if op == "input":
            v = leaves[a["name"]]
        elif op == "const":
            v = a["value"]
            v = torch.full(a["shape"], float(v), dtype=F64) if isinstance(v, (int, float)) else t(v, a["shape"])
        elif op in ("add", "sub", "mul", "div"):
            assert ins[0].shape == ins[1].shape, (op, ins[0].shape, ins[1].shape)
            v = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}[op](ins[0], ins[1])
        elif op == "neg":
            v = torch.neg(ins[0])
        elif op == "exp":
            v = torch.exp(ins[0])
        elif op == "log":
            v = torch.log(ins[0])
        elif op == "relu":
            v = torch.relu(ins[0])
        elif op == "tanh":
            v = torch.tanh(ins[0])
        elif op == "sqrt":
            v = torch.sqrt(ins[0])
        elif op == "step":
            v = (ins[0] > 0).to(F64)
        elif op == "stopGradient":
            v = ins[0].detach()
        elif op == "matmul":
            v = torch.matmul(ins[0], ins[1])
        elif op == "sum":
            v = torch.sum(ins[0], dim=a["axis"], keepdim=a["keepDims"])
        elif op == "max":
            v = torch.amax(ins[0], dim=a["axis"], keepdim=a["keepDims"])
        elif op == "broadcast":
            v = torch.broadcast_to(ins[0], a["shape"])
        elif op == "reshape":
            v = torch.reshape(ins[0], a["shape"])
        elif op == "transpose":
            v = torch.permute(ins[0], a["perm"])
        else:
            raise ValueError(f"unknown op {op}")
        assert list(v.shape) == n["shape"], (op, list(v.shape), n["shape"])
        vals.append(v)
    return vals


def flat(x):
    return x.detach().reshape(-1).tolist()


def evaluate(case):
    graph = case["graph"]
    names = [n["attrs"]["name"] for n in graph["nodes"] if n["op"] == "input"]
    shapes = {n["attrs"]["name"]: n["shape"] for n in graph["nodes"] if n["op"] == "input"}
    leaves = {k: t(case["inputs"][k], shapes[k]).requires_grad_(True) for k in names}
    vals = run_graph(graph, leaves)
    outs = [vals[i] for i in case["outputs"]]
    cots = [t(c, list(o.shape)) for c, o in zip(case["cotangents"], outs)]
    # An output computed from constants alone has no grad_fn; it contributes
    # nothing to any gradient, and autograd refuses to be handed it.
    live = [(o, c) for o, c in zip(outs, cots) if o.requires_grad]
    if live and names:
        grads = torch.autograd.grad([o for o, _ in live], [leaves[k] for k in names],
                                    grad_outputs=[c for _, c in live], allow_unused=True)
    else:
        grads = [None] * len(names)
    result = {
        "name": case["name"],
        "outputs": [flat(o) for o in outs],
        "grads": {k: flat(g) if g is not None else [0.0] * leaves[k].numel() for k, g in zip(names, grads)},
    }
    native = case.get("native")
    if native:
        x = t(case["inputs"][native["input"]], shapes[native["input"]]).requires_grad_(True)
        fn = torch.softmax if native["fn"] == "softmax" else torch.log_softmax
        y = fn(x, dim=native["axis"])
        (gx,) = torch.autograd.grad(y, x, grad_outputs=t(case["cotangents"][0], list(y.shape)))
        result["native"] = {"output": flat(y), "grad": flat(gx)}
    return result


def main():
    # Explicit paths, and no overwriting a golden file by accident: an earlier
    # run with defaults rewrote golden/torch.json (identically, as it turned
    # out) when only milestone 3a's file was meant.
    #   torch_golden.py golden/cases.json golden/torch.json --force
    #   torch_golden.py golden/cases-m3.json golden/torch-m3.json
    import sys
    args = [a for a in sys.argv[1:] if a != "--force"]
    if len(args) != 2:
        sys.exit("usage: torch_golden.py <cases.json> <out.json> [--force]")
    src, dst = ROOT / args[0], ROOT / args[1]
    if dst.exists() and "--force" not in sys.argv:
        sys.exit(f"{dst} exists; pass --force to overwrite a golden file")
    cases = json.loads(src.read_text())["cases"]
    out = {
        "producedBy": "oracle/torch_golden.py",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "dtype": "float64",
        "cases": [evaluate(c) for c in cases],
    }
    dst.write_text(json.dumps(out) + "\n")
    print(f"wrote {len(out['cases'])} cases with torch {torch.__version__}")


if __name__ == "__main__":
    main()
