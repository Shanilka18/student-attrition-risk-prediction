"""
Category E — Deep Sequence Models (PyTorch).
All models receive:
  tensors   : (B, T, D) float32
  mask      : (B, T)  bool  — True where real data, False where padded
  lengths   : (B,)    long

Shared training loop: early stopping, gradient clipping, mixed precision (if GPU),
cosine LR scheduler, dropout, weight decay, focal-loss / BCE+pos_weight.

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

from backend.src.models.base import BaseAcademicModel, register

logger = logging.getLogger("AcademicIntelligence.SequenceModels")


# ──────────────────────────────────────────────────────────────────────────────
# Shared training loop
# ──────────────────────────────────────────────────────────────────────────────

def _train_sequence_model(
    model_obj: "BaseSequenceModel",
    X_train: torch.Tensor,
    mask_train: torch.Tensor,
    len_train: torch.Tensor,
    y_train: np.ndarray,
    X_val: Optional[torch.Tensor],
    mask_val: Optional[torch.Tensor],
    len_val: Optional[torch.Tensor],
    y_val: Optional[np.ndarray],
) -> None:
    p = model_obj.params
    device = model_obj._device

    seed = int(p.get("random_state", p.get("seed", 42)))
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = int(p.get("epochs", 50))
    batch_size = int(p.get("batch_size", 256))
    lr = float(p.get("lr", 1e-3))
    wd = float(p.get("weight_decay", 1e-4))
    patience = int(p.get("early_stopping_patience", 10))
    grad_clip = float(p.get("grad_clip", 1.0))
    use_amp = device.type == "cuda"

    # Class imbalance
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    y_t = torch.tensor(y_train.astype(np.float32), device=device)
    dataset = TensorDataset(X_train, mask_train, len_train, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.AdamW(model_obj._net.parameters(), lr=lr, weight_decay=wd)
    scheduler_name = p.get("scheduler", "cosine")
    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    scaler = GradScaler("cuda", enabled=use_amp)
    best_val_loss = float("inf")
    patience_counter = 0
    best_state: Optional[Dict] = None
    model_obj.training_history = []

    for epoch in range(1, epochs + 1):
        model_obj._net.train()
        total_loss = 0.0

        for xb, mb, lb, yb in loader:
            xb = xb.to(device)
            mb = mb.to(device)
            lb = lb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp):
                logits = model_obj._net(xb, mb, lb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model_obj._net.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(xb)

        train_loss = total_loss / len(y_train)
        scheduler.step()

        val_loss = None
        if X_val is not None and y_val is not None:
            model_obj._net.eval()
            with torch.no_grad():
                Xv = X_val.to(device)
                Mv = mask_val.to(device)
                Lv = len_val.to(device)
                yv = torch.tensor(y_val.astype(np.float32), device=device)
                with autocast("cuda", enabled=use_amp):
                    val_logits = model_obj._net(Xv, Mv, Lv)
                val_loss = criterion(val_logits, yv).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model_obj._net.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"{model_obj.LABEL}: early stopping at epoch {epoch}")
                    break

        model_obj.training_history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6) if val_loss is not None else None,
        })

    if best_state is not None:
        model_obj._net.load_state_dict(best_state)


# ──────────────────────────────────────────────────────────────────────────────
# Base class for all sequence models
# ──────────────────────────────────────────────────────────────────────────────

class BaseSequenceModel(BaseAcademicModel):
    IS_SEQUENCE_MODEL = True
    _net: Optional[nn.Module] = None

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        super().__init__(params)
        from backend.src.config import config
        self._device: torch.device = config.device

    def _build_net(self, input_dim: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X_train, y_train, X_val=None, y_val=None, class_weights=None):
        t0 = self._start_timer()
        # X_train may be dict payload from sequence tensors or raw tensor
        def _unpack(X):
            if isinstance(X, dict):
                return X["tensors"], X["padding_mask"], X["seq_lengths"]
            if isinstance(X, torch.Tensor):
                # Expect (N, T, D); create trivial mask
                T = X.shape[1]
                mask = torch.ones(X.shape[0], T, dtype=torch.bool)
                lengths = torch.full((X.shape[0],), T, dtype=torch.long)
                return X, mask, lengths
            raise TypeError(f"Unsupported X type: {type(X)}")

        X_t, mask_t, len_t = _unpack(X_train)
        input_dim = X_t.shape[2]

        self._net = self._build_net(input_dim).to(self._device)

        Xv_t = mv_t = lv_t = None
        if X_val is not None:
            Xv_t, mv_t, lv_t = _unpack(X_val)

        _train_sequence_model(
            self, X_t, mask_t, len_t, y_train, Xv_t, mv_t, lv_t, y_val
        )
        self.is_fitted = True
        self._stop_timer(t0)
        logger.info(f"{self.LABEL} trained in {self.fit_time_seconds:.1f}s")
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(f"{self.LABEL} not fitted yet.")
        self._net.eval()

        def _unpack(X):
            if isinstance(X, dict):
                return X["tensors"], X["padding_mask"], X["seq_lengths"]
            if isinstance(X, torch.Tensor):
                T = X.shape[1]
                return X, torch.ones(X.shape[0], T, dtype=torch.bool), \
                       torch.full((X.shape[0],), T, dtype=torch.long)
            raise TypeError(f"Unsupported X type: {type(X)}")

        Xt, Mt, Lt = _unpack(X)
        with torch.no_grad():
            logits = self._net(Xt.to(self._device), Mt.to(self._device), Lt.to(self._device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs.astype(np.float32)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.MODEL_KEY}.pt"
        torch.save({
            "state_dict": self._net.state_dict(),
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
        # Rebuild net — input dim saved in state dict's first weight shape
        first_key = next(iter(ckpt["state_dict"]))
        first_shape = ckpt["state_dict"][first_key].shape
        input_dim = first_shape[1] if len(first_shape) in (2, 3) else first_shape[-1]
        obj._net = obj._build_net(input_dim)
        obj._net.load_state_dict(ckpt["state_dict"])
        obj.training_history = ckpt.get("training_history", [])
        obj.is_fitted = True
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Concrete network architectures
# ──────────────────────────────────────────────────────────────────────────────

class _RNNBase(nn.Module):
    """Shared base for Vanilla RNN / LSTM / GRU / BiLSTM / BiGRU."""

    def __init__(self, rnn_cls, input_dim, hidden_size, num_layers,
                 bidirectional, dropout, use_attention=False):
        super().__init__()
        self.use_attention = use_attention
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        if use_attention:
            self.attn = nn.Linear(out_dim, 1)
        self.head = nn.Linear(out_dim, 1)

    def forward(self, x, mask, lengths):
        # Pack for efficiency
        lens_cpu = lengths.cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lens_cpu, batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        if self.use_attention:
            # Mask-aware soft attention
            scores = self.attn(out).squeeze(-1)  # (B, T)
            scores = scores.masked_fill(~mask[:, :out.shape[1]], float("-inf"))
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            context = (out * weights).sum(dim=1)
        else:
            # Take last real token output
            idx = (lens_cpu - 1).clamp(min=0).to(x.device)
            context = out[torch.arange(out.shape[0], device=x.device), idx]

        return self.head(self.dropout(context)).squeeze(-1)


def _make_rnn_net(rnn_cls, params, input_dim, use_attention=False, bidirectional=False):
    p = params
    return _RNNBase(
        rnn_cls=rnn_cls,
        input_dim=input_dim,
        hidden_size=int(p.get("hidden_size", 64)),
        num_layers=int(p.get("num_layers", 1)),
        bidirectional=p.get("bidirectional", bidirectional),
        dropout=float(p.get("dropout", 0.3)),
        use_attention=p.get("use_attention", use_attention),
    )


@register
class VanillaRNNModel(BaseSequenceModel):
    MODEL_KEY = "vanilla_rnn"
    CATEGORY = "E"
    LABEL = "Vanilla RNN"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.RNN, self.params, input_dim)


@register
class LSTMModel(BaseSequenceModel):
    MODEL_KEY = "lstm"
    CATEGORY = "E"
    LABEL = "LSTM (Proposal Model)"
    IS_PROPOSAL_MODEL = True

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.LSTM, self.params, input_dim)


@register
class GRUModel(BaseSequenceModel):
    MODEL_KEY = "gru"
    CATEGORY = "E"
    LABEL = "GRU (Proposal Model)"
    IS_PROPOSAL_MODEL = True

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.GRU, self.params, input_dim)


@register
class BiLSTMModel(BaseSequenceModel):
    MODEL_KEY = "bilstm"
    CATEGORY = "E"
    LABEL = "BiLSTM"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.LSTM, self.params, input_dim, bidirectional=True)


@register
class BiGRUModel(BaseSequenceModel):
    MODEL_KEY = "bigru"
    CATEGORY = "E"
    LABEL = "BiGRU"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.GRU, self.params, input_dim, bidirectional=True)


@register
class LSTMAttentionModel(BaseSequenceModel):
    MODEL_KEY = "lstm_attention"
    CATEGORY = "E"
    LABEL = "LSTM + Attention"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.LSTM, self.params, input_dim, use_attention=True)


@register
class GRUAttentionModel(BaseSequenceModel):
    MODEL_KEY = "gru_attention"
    CATEGORY = "E"
    LABEL = "GRU + Attention"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        return _make_rnn_net(nn.GRU, self.params, input_dim, use_attention=True)


# ── 1D-CNN ──────────────────────────────────────────────────────────────────

class _CNN1D(nn.Module):
    def __init__(self, input_dim, num_filters, kernel_sizes, dropout):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(num_filters * len(kernel_sizes), 1)

    def forward(self, x, mask, lengths):
        # x: (B, T, D) → (B, D, T) for Conv1d
        x = x.permute(0, 2, 1)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))          # (B, F, T)
            p = c.max(dim=-1).values     # global max pool
            pooled.append(p)
        out = torch.cat(pooled, dim=-1)  # (B, F * n_kernels)
        return self.head(self.dropout(out)).squeeze(-1)


@register
class CNN1DModel(BaseSequenceModel):
    MODEL_KEY = "cnn_1d"
    CATEGORY = "E"
    LABEL = "1D-CNN"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        p = self.params
        return _CNN1D(
            input_dim=input_dim,
            num_filters=int(p.get("num_filters", 64)),
            kernel_sizes=list(p.get("kernel_sizes", [2, 3])),
            dropout=float(p.get("dropout", 0.3)),
        )


# ── CNN-LSTM hybrid ──────────────────────────────────────────────────────────

class _CNNLSTM(nn.Module):
    def __init__(self, input_dim, num_filters, lstm_hidden, num_layers, dropout):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size=2, padding=1)
        self.lstm = nn.LSTM(
            num_filters, lstm_hidden, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(lstm_hidden, 1)

    def forward(self, x, mask, lengths):
        x_c = F.relu(self.conv(x.permute(0, 2, 1))).permute(0, 2, 1)
        T_out = x_c.shape[1]
        lens_cpu = lengths.clamp(max=T_out, min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x_c, lens_cpu, batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        idx = (lens_cpu - 1).clamp(min=0).to(x.device)
        last = out[torch.arange(out.shape[0], device=x.device), idx]
        return self.head(self.dropout(last)).squeeze(-1)


@register
class CNNLSTMModel(BaseSequenceModel):
    MODEL_KEY = "cnn_lstm"
    CATEGORY = "E"
    LABEL = "CNN-LSTM Hybrid"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        p = self.params
        return _CNNLSTM(
            input_dim=input_dim,
            num_filters=int(p.get("num_filters", 64)),
            lstm_hidden=int(p.get("lstm_hidden", 64)),
            num_layers=int(p.get("num_layers", 1)),
            dropout=float(p.get("dropout", 0.3)),
        )


# ── Temporal Convolutional Network (TCN) ────────────────────────────────────

class _TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
        self._pad = pad

    def forward(self, x):
        out = self.relu(self.conv1(x)[..., :-self._pad] if self._pad else self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out)[..., :-self._pad] if self._pad else self.conv2(out))
        out = self.dropout(out)
        res = x if self.downsample is None else self.downsample(x)
        # Match lengths
        if out.shape[-1] != res.shape[-1]:
            min_len = min(out.shape[-1], res.shape[-1])
            out = out[..., :min_len]
            res = res[..., :min_len]
        return self.relu(out + res)


class _TCN(nn.Module):
    def __init__(self, input_dim, num_channels, kernel_size, dropout):
        super().__init__()
        blocks = []
        for i, ch in enumerate(num_channels):
            dilation = 2 ** i
            in_ch = input_dim if i == 0 else num_channels[i - 1]
            blocks.append(_TemporalBlock(in_ch, ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*blocks)
        self.head = nn.Linear(num_channels[-1], 1)

    def forward(self, x, mask, lengths):
        out = self.network(x.permute(0, 2, 1))  # (B, C, T)
        out = out.permute(0, 2, 1)              # (B, T, C)
        # Last real timestep
        idx = (lengths.clamp(min=1) - 1).clamp(max=out.shape[1] - 1).to(x.device)
        last = out[torch.arange(out.shape[0], device=x.device), idx]
        return self.head(last).squeeze(-1)


@register
class TCNModel(BaseSequenceModel):
    MODEL_KEY = "tcn"
    CATEGORY = "E"
    LABEL = "Temporal Convolutional Network"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        p = self.params
        return _TCN(
            input_dim=input_dim,
            num_channels=list(p.get("num_channels", [64, 64, 64])),
            kernel_size=int(p.get("kernel_size", 2)),
            dropout=float(p.get("dropout", 0.2)),
        )


# ── Small Transformer Encoder ─────────────────────────────────────────────────

class _TransformerEnc(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, mask, lengths):
        # src_key_padding_mask: True where PADDED
        pad_mask = ~mask[:, :x.shape[1]]  # (B, T)
        x = self.proj(x)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        # CLS-like: mean over real tokens
        real_mask = mask[:, :x.shape[1]].unsqueeze(-1).float()
        pooled = (out * real_mask).sum(1) / real_mask.sum(1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


@register
class TransformerEncModel(BaseSequenceModel):
    MODEL_KEY = "transformer_enc"
    CATEGORY = "E"
    LABEL = "Transformer Encoder"
    IS_PROPOSAL_MODEL = False

    def _build_net(self, input_dim):
        p = self.params
        return _TransformerEnc(
            input_dim=input_dim,
            d_model=int(p.get("d_model", 32)),
            nhead=int(p.get("nhead", 4)),
            num_layers=int(p.get("num_layers", 2)),
            dim_feedforward=int(p.get("dim_feedforward", 128)),
            dropout=float(p.get("dropout", 0.2)),
        )
