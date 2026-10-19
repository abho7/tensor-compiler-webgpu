"""The PyTorch oracle for milestone 3b (DESIGN.md amendments 20 to 26). Offline only.

Written with PyTorch's own primitives, not as a translation of this project's
graph: F.embedding, F.layer_norm, F.scaled_dot_product_attention(is_causal=True),
F.gelu(approximate='tanh'), F.cross_entropy and torch.optim.Adam.

Initial parameters and projection vectors are regenerated with a port of the
project's RNG (src/testing/rng.js) and must hash to the values in the spec.

    .venv/Scripts/python oracle/torch_transformer.py golden/spec-m3b.json golden/torch-m3b [--force]

writes golden/torch-m3b.json (metadata, losses, projections, statistics) and
golden/torch-m3b.f64 (float64 tensors, indexed by the JSON).
"""
import hashlib
import json
import math
import pathlib
import platform
import struct
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parent.parent
M32 = 0xFFFFFFFF
torch.set_num_threads(4)


class Rng:
    """mulberry32, as src/testing/rng.js computes it, in unsigned 32-bit arithmetic."""

    def __init__(self, seed):
        self.state = seed & M32

    def next(self):
        self.state = (self.state + 0x6D2B79F5) & M32
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & M32
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & M32)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296

    def uniform(self, lo, hi):
        return lo + (hi - lo) * self.next()

    def array(self, n, lo, hi):
        return [self.uniform(lo, hi) for _ in range(n)]


def numel(shape):
    return math.prod(shape) if shape else 1


def sha_f64(arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(struct.pack(f"<{len(a)}d", *a))
    return h.hexdigest()


def init_params(order, seed):
    rng = Rng((seed * 7001 + 13) & M32)
    out = {}
    for p in order:
        n = numel(p["shape"])
        kind = p["init"][0]
        out[p["name"]] = rng.array(n, -p["init"][1], p["init"][1]) if kind == "uniform" else [1.0 if kind == "ones" else 0.0] * n
    return out


def fnv(name):
    h = 0x811C9DC5
    for ch in name:
        h ^= ord(ch)
        h = (h * 0x01000193) & M32
    return h


def projection(name, n, k):
    return Rng((fnv(name) + 7919 * k) & M32).array(n, -1, 1)


def order_for(order, V):
    """The parameter order and shapes for vocabulary V (the spec stores V = 16's)."""
    out = []
    for p in order:
        shape = list(p["shape"])
        if p["name"] == "tok":
            shape = [V, shape[1]]
        elif p["name"] == "head":
            shape = [shape[0], V]
        elif p["name"] == "headb":
            shape = [V]
        out.append({**p, "shape": shape})
    return out


def tensors(order, flat, dtype):
    return {p["name"]: torch.tensor(flat[p["name"]], dtype=torch.float64).reshape(p["shape"]).to(dtype).requires_grad_(True) for p in order}


def forward(P, tokens, cfg):
    T, d, H, dh, L, eps = cfg["T"], cfg["d"], cfg["H"], cfg["dh"], cfg["L"], cfg["eps"]
    B = tokens.shape[0]
    h = F.embedding(tokens, P["tok"]) + P["pos"]
    for l in range(L):
        p = lambda k: P[f"b{l}.{k}"]
        a = F.layer_norm(h, (d,), p("ln1.g"), p("ln1.b"), eps)
        split = lambda w: (a @ w).reshape(B, T, H, dh).transpose(1, 2)
        o = F.scaled_dot_product_attention(split(p("wq")), split(p("wk")), split(p("wv")), is_causal=True)
        h = h + o.transpose(1, 2).reshape(B, T, d) @ p("wo") + p("bo")
        m = F.layer_norm(h, (d,), p("ln2.g"), p("ln2.b"), eps)
        h = h + F.gelu(m @ p("w1") + p("b1"), approximate="tanh") @ p("w2") + p("b2")
    return F.layer_norm(h, (d,), P["lnf.g"], P["lnf.b"], eps) @ P["head"] + P["headb"]


def loss_of(P, seqs, cfg):
    x, y = seqs[:, :-1], seqs[:, 1:]
    logits = forward(P, x, cfg)
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)), logits


def adam(P, opt):
    return torch.optim.Adam(list(P.values()), lr=opt["lr"], betas=(opt["beta1"], opt["beta2"]), eps=opt["eps"],
                            weight_decay=opt["weightDecay"], foreach=False)


class Blob:
    """float64 tensors appended to one file, indexed by name in the JSON."""

    def __init__(self):
        self.parts, self.index, self.offset = [], {}, 0

    def add(self, key, values):
        values = [float(v) for v in values]
        self.parts.append(struct.pack(f"<{len(values)}d", *values))
        self.index[key] = [self.offset, len(values)]
        self.offset += len(values)


def flat(t):
    return t.detach().to(torch.float64).reshape(-1).tolist()


def val_loss(P, seqs, cfg):
    with torch.no_grad():
        P64 = {k: v.detach().to(torch.float64) for k, v in P.items()}
        total, n = 0.0, 0
        for i in range(0, seqs.shape[0], 32):
            l, _ = loss_of(P64, seqs[i:i + 32], cfg)
            total += l.item() * seqs[i:i + 32].shape[0]
            n += seqs[i:i + 32].shape[0]
        return total / n


def main():
    args = [a for a in sys.argv[1:] if a != "--force"]
    if len(args) != 2:
        sys.exit("usage: torch_transformer.py <spec.json> <out-prefix> [--force]")
    src = ROOT / args[0]
    out_json, out_bin = ROOT / (args[1] + ".json"), ROOT / (args[1] + ".f64")
    if (out_json.exists() or out_bin.exists()) and "--force" not in sys.argv:
        sys.exit(f"{out_json} or {out_bin} exists; pass --force to overwrite a golden file")
    spec = json.loads(src.read_text(encoding="utf-8"))
    cfg, opt = spec["config"], spec["optimizer"]
    order16 = spec["paramOrder"]

    # The RNG port must reproduce the projection vectors and every initialisation exactly.
    k = spec["projections"]["k"]
    proj = {p["name"]: [projection(p["name"], numel(p["shape"]), j) for j in range(k)] for p in order16}
    if sha_f64([v for p in order16 for v in proj[p["name"]]]) != spec["projections"]["sha256"]:
        sys.exit("projection vectors do not match the spec's SHA-256: the RNG port is wrong")

    def init_checked(order, seed, want):
        flat_init = init_params(order, seed)
        if sha_f64([flat_init[p["name"]] for p in order]) != want:
            sys.exit(f"initialisation for seed {seed} does not match the spec's SHA-256")
        return flat_init

    syn = spec["synthetic"]
    train = torch.tensor(syn["train"])
    val = torch.tensor(syn["val"])
    blob = Blob()
    out = {"producedBy": "oracle/torch_transformer.py", "torch": torch.__version__, "python": platform.python_version()}

    # (i) at initialisation: logits, loss, every gradient.
    text = (ROOT / spec["text"]["path"]).read_bytes()
    if hashlib.sha256(text).hexdigest() != spec["text"]["sha256"]:
        sys.exit("data/alice/alice.txt does not match its SHA-256")
    text = text.decode("utf-8")
    vocab = sorted(set(text))
    if vocab != spec["text"]["vocab"]:
        sys.exit("vocabulary differs from the spec")
    index = {c: i for i, c in enumerate(vocab)}
    ids = torch.tensor([index[c] for c in text])
    Vt = len(vocab)
    order_t = order_for(order16, Vt)
    T1 = cfg["T"] + 1
    windows = lambda offs: torch.stack([ids[o:o + T1] for o in offs])

    init_cases = [(g["name"], order16, g["seed"], g["initSha"], train[torch.tensor(g["schedule"][0])]) for g in spec["golden"]]
    ti = spec["text"]["init"]
    init_cases.append(("text-seed1", order_t, ti["seed"], ti["initSha"], windows(ti["offsets"])))
    out["init"] = []
    for name, order, seed, want, seqs in init_cases:
        P = tensors(order, init_checked(order, seed, want), torch.float64)
        loss, logits = loss_of(P, seqs, cfg)
        loss.backward()
        blob.add(f"init/{name}/logits", flat(logits))
        for p in order:
            blob.add(f"init/{name}/grad/{p['name']}", flat(P[p["name"]].grad))
        out["init"].append({"name": name, "loss": loss.item()})
        print(f"init {name}: loss {loss.item():.6f}")

    # (ii) training, N steps: losses, projections every step, every parameter after the last.
    out["training"] = []
    for g in spec["golden"]:
        P = tensors(order16, init_checked(order16, g["seed"], g["initSha"]), torch.float64)
        o = adam(P, opt)
        losses, projections = [], []
        for t in range(g["steps"]):
            o.zero_grad()
            loss, _ = loss_of(P, train[torch.tensor(g["schedule"][t])], cfg)
            loss.backward()
            o.step()
            losses.append(loss.item())
            projections.append({p["name"]: [sum(a * b for a, b in zip(flat(P[p["name"]]), proj[p["name"]][j])) for j in range(k)] for p in order16})
        for p in order16:
            blob.add(f"train/{g['name']}/final/{p['name']}", flat(P[p["name"]]))
        out["training"].append({"name": g["name"], "losses": losses, "projections": projections})
        print(f"train {g['name']}: loss {losses[0]:.6f} -> {losses[-1]:.6f}")

    # (iv) statistical: 24 seeds in float64, and in float32 for context; final validation loss in float64.
    st = spec["statistical"]
    out["statistical"] = {"float64": [], "float32": []}
    for dtype, key in ((torch.float64, "float64"), (torch.float32, "float32")):
        for run in st["runs"]:
            P = tensors(order16, init_checked(order16, run["seed"], run["initSha"]), dtype)
            o = adam(P, opt)
            losses = []
            for idx in run["schedule"]:
                o.zero_grad()
                loss, _ = loss_of(P, train[torch.tensor(idx)], cfg)
                loss.backward()
                o.step()
                losses.append(loss.item())
            entry = {"seed": run["seed"], "valLoss": val_loss(P, val, cfg), "finalTrainLoss": losses[-1]}
            if run["seed"] == 1:
                entry["losses"] = losses
            out["statistical"][key].append(entry)
            print(f"statistical {key} seed {run['seed']}: val {entry['valLoss']:.5f}")

    # The demo: float64 and float32 curves from the same start, and their validation loss.
    dm = spec["demo"]
    vwin = windows(dm["valOffsets"])
    out["demo"] = {}
    for dtype, key in ((torch.float64, "float64"), (torch.float32, "float32")):
        P = tensors(order_t, init_checked(order_t, dm["seed"], dm["initSha"]), dtype)
        o = adam(P, opt)
        losses = []
        for offs in dm["offsets"]:
            o.zero_grad()
            loss, _ = loss_of(P, windows(offs), cfg)
            loss.backward()
            o.step()
            losses.append(loss.item())
        out["demo"][key] = {"losses": losses, "valLoss": val_loss(P, vwin, cfg)}
        print(f"demo {key}: {losses[0]:.4f} -> {losses[-1]:.4f}, val {out['demo'][key]['valLoss']:.4f}")

    out["blob"] = {"file": out_bin.name, "index": blob.index, "count": blob.offset}
    out_bin.write_bytes(b"".join(blob.parts))
    out_json.write_text(json.dumps(out) + "\n", encoding="utf-8")
    print(f"wrote {out_json.name} and {out_bin.name} ({blob.offset} float64 values) with torch {torch.__version__}")


if __name__ == "__main__":
    main()
