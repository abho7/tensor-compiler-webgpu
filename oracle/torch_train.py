"""The PyTorch training oracle for milestone 3a. Offline only; CI checks its output.

Reproduces the runs in golden/train-spec-m3.json in float64 with
torch.optim.SGD and torch.optim.Adam themselves (not a translation of this
project's optimizer graphs): the same initial parameters, the same batches in
the same order. Records, for the synthetic golden runs, the loss and every
parameter after every step; for the demo run, every step's loss and the final
test loss and accuracy.

    .venv/Scripts/python oracle/torch_train.py golden/train-spec-m3.json golden/torch-train-m3.json
"""
import hashlib
import json
import pathlib
import platform
import sys

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
ROOT = pathlib.Path(__file__).resolve().parent.parent
PARAMS = ["W1", "b1", "W2", "b2"]


def tensors(init, dims):
    shapes = {"W1": (dims["inputs"], dims["hidden"]), "b1": (dims["hidden"],),
              "W2": (dims["hidden"], dims["classes"]), "b2": (dims["classes"],)}
    return {k: torch.tensor(init[k], dtype=torch.float64).reshape(shapes[k]).requires_grad_(True) for k in PARAMS}


def optimizer(spec, params):
    p = [params[k] for k in PARAMS]
    if spec["kind"] == "sgd":
        return torch.optim.SGD(p, lr=spec["lr"], momentum=spec["momentum"], weight_decay=spec["weightDecay"], foreach=False)
    return torch.optim.Adam(p, lr=spec["lr"], betas=(spec["beta1"], spec["beta2"]), eps=spec["eps"],
                            weight_decay=spec["weightDecay"], foreach=False)


def logits(params, x):
    return torch.relu(x @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]


def run(x_all, y_all, dims, cfg, keep_params):
    params = tensors(cfg["init"], dims)
    opt = optimizer(cfg["optimizer"], params)
    losses, kept = [], []
    for idx in cfg["schedule"][: cfg["steps"]]:
        idx = torch.tensor(idx)
        opt.zero_grad()
        loss = F.cross_entropy(logits(params, x_all[idx]), y_all[idx])
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if keep_params:
            kept.append({k: params[k].detach().reshape(-1).tolist() for k in PARAMS})
    return params, losses, kept


def read_optdigits(entry):
    raw = (ROOT / entry["path"]).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != entry["sha256"]:
        sys.exit(f"{entry['path']}: sha256 {digest} does not match {entry['sha256']}")
    rows = [list(map(int, line.split(","))) for line in raw.decode().splitlines() if line.strip()]
    x = torch.tensor([r[:64] for r in rows], dtype=torch.float64) / 16
    y = torch.tensor([r[64] for r in rows])
    return x, y


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    if len(args) != 2:
        sys.exit("usage: torch_train.py <train-spec.json> <out.json> [--force]")
    src, dst = ROOT / args[0], ROOT / args[1]
    if dst.exists() and "--force" not in sys.argv:
        sys.exit(f"{dst} exists; pass --force to overwrite a golden file")
    spec = json.loads(src.read_text())

    dims = spec["dims"]
    syn = spec["synthetic"]["train"]
    x = torch.tensor(syn["x"], dtype=torch.float64).reshape(-1, dims["inputs"])
    y = torch.tensor(syn["y"])
    configs = []
    for cfg in spec["configs"]:
        _, losses, kept = run(x, y, dims, cfg, keep_params=True)
        configs.append({"name": cfg["name"], "losses": losses, "params": kept})

    demo = spec["demo"]
    xtr, ytr = read_optdigits(demo["files"]["train"])
    xte, yte = read_optdigits(demo["files"]["test"])
    params, losses, _ = run(xtr, ytr, demo["dims"], demo, keep_params=False)
    with torch.no_grad():
        out = logits(params, xte)
        test_loss = F.cross_entropy(out, yte).item()
        test_acc = (out.argmax(1) == yte).double().mean().item()

    result = {
        "producedBy": "oracle/torch_train.py",
        "torch": torch.__version__, "python": platform.python_version(), "dtype": "float64",
        "configs": configs,
        "demo": {"losses": losses, "testLoss": test_loss, "testAccuracy": test_acc,
                 "final": {k: params[k].detach().reshape(-1).tolist() for k in PARAMS}},
    }
    dst.write_text(json.dumps(result) + "\n")
    print(f"wrote {len(configs)} golden runs and the demo ({len(losses)} steps, test accuracy {test_acc:.4f}) with torch {torch.__version__}")


if __name__ == "__main__":
    main()
