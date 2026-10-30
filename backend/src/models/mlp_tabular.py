"""
MLP on Engineered Tabular Features — Category D.
Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from backend.src.models.base import BaseAcademicModel, register

logger = logging.getLogger("AcademicIntelligence.TabularMLP")


class _MLPNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@register
class MLPTabularModel(BaseAcademicModel):
    MODEL_KEY = "mlp_tabular"
    CATEGORY = "D"
    LABEL = "MLP (Engineered Features)"
    IS_PROPOSAL_MODEL = False
    IS_SEQUENCE_MODEL = False

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        self._net: Optional[_MLPNet] = None
        self._device: torch.device = torch.device("cpu")
        self._input_dim: int = 0

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        p = self.params
        t0 = self._start_timer()

        from backend.src.config import config
        self._device = config.device

        seed = int(p.get("random_state", p.get("seed", 42)))
        torch.manual_seed(seed)
        np.random.seed(seed)

        X_t = torch.tensor(np.asarray(X_train, dtype=np.float32))
        y_t = torch.tensor(y_train.astype(np.float32))
        self._input_dim = X_t.shape[1]

        hidden_dims = p.get("hidden_dims", [128, 64, 32])
        dropout = float(p.get("dropout", 0.3))
        lr = float(p.get("lr", 1e-3))
        wd = float(p.get("weight_decay", 1e-4))
        epochs = int(p.get("epochs", 50))
        batch_size = int(p.get("batch_size", 256))
        patience = int(p.get("early_stopping_patience", 10))

        # Pos weight for class imbalance
        n_pos = int(np.sum(y_train == 1))
        n_neg = int(np.sum(y_train == 0))
        pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

        self._net = _MLPNet(self._input_dim, hidden_dims, dropout).to(self._device)
        optimizer = optim.AdamW(self._net.parameters(), lr=lr, weight_decay=wd)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=batch_size, shuffle=True
        )

        best_val_loss = float("inf")
        patience_counter = 0
        self.training_history = []

        for epoch in range(1, epochs + 1):
            self._net.train()
            train_loss = 0.0
            for xb, yb in loader:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
            train_loss /= len(X_t)
            scheduler.step()

            val_loss = None
            if X_val is not None:
                self._net.eval()
                with torch.no_grad():
                    Xv = torch.tensor(np.asarray(X_val, np.float32)).to(self._device)
                    yv = torch.tensor(y_val.astype(np.float32)).to(self._device)
                    val_loss = criterion(self._net(Xv), yv).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"MLP early stop at epoch {epoch}")
                        break

            self.training_history.append({
                "epoch": epoch, "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6) if val_loss is not None else None
            })

        # Restore best weights
        if hasattr(self, "_best_state"):
            self._net.load_state_dict(self._best_state)

        self.is_fitted = True
        self._stop_timer(t0)
        logger.info(f"MLP trained {len(self.training_history)} epochs in {self.fit_time_seconds:.1f}s")
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MLPTabularModel not fitted yet.")
        self._net.eval()
        with torch.no_grad():
            X_t = torch.tensor(np.asarray(X, dtype=np.float32)).to(self._device)
            logits = self._net(X_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.MODEL_KEY}.pt"
        torch.save({
            "state_dict": self._net.state_dict(),
            "input_dim": self._input_dim,
            "params": self.params,
            "training_history": self.training_history,
        }, path)
        (directory / f"{self.MODEL_KEY}_meta.json").write_text(
            json.dumps(self.metadata(), indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, directory: Path):
        path = directory / f"{cls.MODEL_KEY}.pt"
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(params=ckpt["params"])
        p = obj.params
        obj._input_dim = ckpt["input_dim"]
        obj._net = _MLPNet(
            obj._input_dim,
            p.get("hidden_dims", [128, 64, 32]),
            float(p.get("dropout", 0.3))
        )
        obj._net.load_state_dict(ckpt["state_dict"])
        obj.training_history = ckpt.get("training_history", [])
        obj.is_fitted = True
        return obj
