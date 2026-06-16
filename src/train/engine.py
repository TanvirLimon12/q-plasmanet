"""Phase 3 — shared training loop.

Seeding, device logging, train/val passes, best-checkpoint selection, history.
Reused by classical (Phase 3) and quantum (Phase 4) training.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import compute_metrics  # noqa: E402
from src.utils.seed import log_device, set_seed  # noqa: E402


def get_device() -> torch.device:
    """GPU for classical/backbone work: CUDA, else Apple MPS (M-series), else CPU.
    Quantum training must stay on CPU (lightning.qubit + TorchLayer); callers there
    pass torch.device('cpu') explicitly."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(params, name: str, lr: float, weight_decay: float = 0.0):
    name = (name or "adam").lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob_class1) over a loader."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob1 = torch.softmax(logits, dim=1)[:, 1]
        ps.append(prob1.cpu().numpy())
        ys.append(np.asarray(y))
    if not ys:
        return np.array([]), np.array([])
    return np.concatenate(ys), np.concatenate(ps)


def train_loop(
    model,
    train_dl,
    val_dl,
    *,
    epochs: int,
    lr: float,
    seed: int,
    device=None,
    optimizer: str = "adam",
    weight_decay: float = 0.0,
    select_metric: str = "auc",
    verbose: bool = True,
) -> tuple[dict, list[dict]]:
    """Train ``model``; return (best_state_dict, history). Selects on val metric."""
    set_seed(seed)
    device = device or get_device()
    model.to(device)
    if verbose:
        print(f"[engine] device={log_device()} seed={seed} epochs={epochs}")

    opt = make_optimizer((p for p in model.parameters() if p.requires_grad), optimizer, lr, weight_decay)
    crit = nn.CrossEntropyLoss()

    best_score = -np.inf
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for x, y in train_dl:
            x = x.to(device)
            y = torch.as_tensor(y, dtype=torch.long, device=device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = running / max(n, 1)

        # Validation
        if val_dl is not None and len(getattr(val_dl, "dataset", [])) > 0:
            y_true, y_prob = predict(model, val_dl, device)
            val_m = compute_metrics(y_true, y_prob)
            score = val_m.get(select_metric, val_m["f1"])
        else:
            val_m = {}
            score = -train_loss            # no val: track by loss

        rec = {"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(rec)
        if verbose:
            extra = f" val_{select_metric}={score:.4f}" if val_m else ""
            print(f"  epoch {epoch}: train_loss={train_loss:.4f}{extra}")

        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return best_state, history
