from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import matplotlib
import matplotlib.pyplot as plt

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Callable
from torch.utils.data import Dataset, DataLoader

matplotlib.use("Agg")

from logging_utils import setup_logger
from datasets import pad_and_mask
from metrics import metrics_dict


logger = setup_logger()


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    batch_size: int = 6
    patience: int = 5
    seed: int = 2024
    device: str = "cuda"
    loss_type: str = "ccc"  # 'mse'|'ccc'|'hybrid'


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class SongSequenceDataset(Dataset):
    def __init__(self, items: List[Tuple[np.ndarray, np.ndarray]]):
        self.items = items
        self.input_dim = items[0][0].shape[1]

    def __len__(self): return len(self.items)

    def __getitem__(self, i): return self.items[i]


def masked_mse(pred, target, mask):
    mask = mask.unsqueeze(-1)
    num = ((pred - target) ** 2 * mask).sum()
    den = mask.sum() * pred.shape[-1]
    return num / (den + 1e-8)


def masked_ccc_loss(pred, target, mask):
    eps = 1e-8
    m = mask.unsqueeze(-1)
    valid = m > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    losses = []
    for d in range(pred.shape[-1]):
        x = pred[..., d][valid[..., 0]]
        y = target[..., d][valid[..., 0]]
        mx = torch.mean(x)
        my = torch.mean(y)
        vx = torch.var(x, unbiased=False)
        vy = torch.var(y, unbiased=False)
        cov = torch.mean((x - mx) * (y - my))
        ccc = (2 * cov) / (vx + vy + (mx - my) ** 2 + eps)
        losses.append(1.0 - ccc)
    return torch.mean(torch.stack(losses))


def make_loss_fn(loss_type: str) -> Callable:
    lt = str(loss_type).lower()
    if lt == "mse":
        return masked_mse
    if lt == "ccc":
        return masked_ccc_loss
    if lt == "hybrid":
        def hybrid(pred, target, mask):
            return 0.5 * masked_mse(pred, target, mask) + 0.5 * masked_ccc_loss(pred, target, mask)
        return hybrid
    logger.warning(f"Unknown loss '{loss_type}', using CCC.")
    return masked_ccc_loss


def evaluate_model(model, dl, device):
    model.eval()
    Ys = []
    Ps = []
    # performance improvement only, as no backward propagation is done anyway
    with torch.no_grad():
        for Xb, Yb, Mb in dl:
            Xb = Xb.to(device)
            Yb = Yb.to(device)
            Mb = Mb.to(device)
            P = model(Xb)
            M = Mb.unsqueeze(-1)
            Ys.append((Yb * M).reshape(-1, 2)[Mb.reshape(-1) > 0].cpu().numpy())
            Ps.append((P * M).reshape(-1, 2)[Mb.reshape(-1) > 0].cpu().numpy())
    Y = np.concatenate(Ys, 0)
    P = np.concatenate(Ps, 0)

    def agg(yc, pc):
        m = metrics_dict(yc, pc)
        return m["CCC"], m["Pearson"], m["R2"], m["RMSE"]

    vccc, vp, vr2, vrmse = agg(Y[:, 0], P[:, 0])
    accc, ap, ar2, armse = agg(Y[:, 1], P[:, 1])
    return {"Valence_CCC": vccc, "Valence_Pearson": vp, "Valence_R2": vr2, "Valence_RMSE": vrmse,
            "Arousal_CCC": accc, "Arousal_Pearson": ap, "Arousal_R2": ar2, "Arousal_RMSE": armse,
            "CCC_mean": (vccc + accc) / 2.0}


class Trainer:
    def __init__(self, results_dir: Path, config: TrainConfig):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.results_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        self.plots_dir = self.results_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir = self.results_dir / "checkpoints"
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.cfg = config
        set_seed(config.seed)

    def train_fold(self, fold_idx: int, train_items, val_items, model_builder):
        device = self.cfg.device if torch.cuda.is_available() and self.cfg.device == "cuda" else "cpu"
        ds_tr = SongSequenceDataset(train_items)
        ds_va = SongSequenceDataset(val_items)
        dl_tr = DataLoader(ds_tr, batch_size=self.cfg.batch_size, shuffle=True, collate_fn=pad_and_mask)
        dl_va = DataLoader(ds_va, batch_size=self.cfg.batch_size, shuffle=False, collate_fn=pad_and_mask)

        model = model_builder(ds_tr.input_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=self.cfg.lr)
        loss_fn = make_loss_fn(self.cfg.loss_type)

        best = -1e9
        patience = self.cfg.patience
        best_path = self.ckpt_dir / f"fold{fold_idx}_best.pt"
        last_path = self.ckpt_dir / f"fold{fold_idx}_last.pt"
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            model.train()
            losses = []
            for Xb, Yb, Mb in dl_tr:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                Mb = Mb.to(device)
                P = model(Xb)
                loss = loss_fn(P, Yb, Mb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
            train_loss = float(np.mean(losses))
            train_m = evaluate_model(model, dl_tr, device)
            val_m = evaluate_model(model, dl_va, device)

            row = {"epoch": epoch, "train_loss": train_loss}
            row.update({f"train_{k}": v for k, v in train_m.items()})
            row.update({f"val_{k}": v for k, v in val_m.items()})
            history.append(row)

            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, last_path)

            score = val_m["CCC_mean"]
            if score > best:
                best = score
                patience = self.cfg.patience
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch}, best_path)
            else:
                patience -= 1
                if patience <= 0:
                    logger.info(f"[fold {fold_idx}] Early stopping at epoch {epoch}")
                    break

            logger.info(f"[fold {fold_idx}][epoch {epoch}] loss={train_loss:.4f} val_CCC_mean={score:.3f}")

        df = pd.DataFrame(history)
        df.to_csv(self.logs_dir / f"fold{fold_idx}_history.csv", index=False)
        plt.figure(figsize=(10, 5))
        if "train_CCC_mean" in df and "val_CCC_mean" in df:
            plt.plot(df["epoch"], df["train_CCC_mean"], label="train CCC mean")
            plt.plot(df["epoch"], df["val_CCC_mean"], label="val CCC mean")
        plt.xlabel("Epoch")
        plt.ylabel("CCC mean")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.plots_dir / f"fold{fold_idx}_ccc_curve.png")
        plt.close()

        return str(best_path)
