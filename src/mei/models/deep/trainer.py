"""PyTorch two-stage training loop, early stopping, and metric tracking for MEI."""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DeepEngagementDataset(Dataset):
    """PyTorch Dataset supporting sequence, tabular, or dual multimodal inputs."""

    def __init__(
        self,
        X_tab: Optional[np.ndarray] = None,
        X_seq: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> None:
        self.X_tab = (
            torch.tensor(X_tab, dtype=torch.float32) if X_tab is not None else None
        )
        self.X_seq = (
            torch.tensor(X_seq, dtype=torch.float32) if X_seq is not None else None
        )
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

        if self.X_tab is not None:
            self.n_samples = len(self.X_tab)
        elif self.X_seq is not None:
            self.n_samples = len(self.X_seq)
        else:
            raise ValueError(
                "At least one of X_tab or X_seq must be provided to DeepEngagementDataset."
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Any, Optional[torch.Tensor]]:
        if self.X_tab is not None and self.X_seq is not None:
            x_val = (self.X_seq[idx], self.X_tab[idx])
        elif self.X_seq is not None:
            x_val = self.X_seq[idx]
        else:
            x_val = self.X_tab[idx]

        target = self.y[idx] if self.y is not None else None
        return x_val, target


def _collate_batch(batch: List[Tuple[Any, Any]]) -> Tuple[Any, Optional[torch.Tensor]]:
    """Custom collator handling tuple inputs (sequence, tabular) and tensor batches."""
    x_first = batch[0][0]
    if isinstance(x_first, tuple):
        seqs = torch.stack([item[0][0] for item in batch])
        tabs = torch.stack([item[0][1] for item in batch])
        x_batch = (seqs, tabs)
    else:
        x_batch = torch.stack([item[0] for item in batch])

    if batch[0][1] is not None:
        y_batch = torch.stack([item[1] for item in batch])
    else:
        y_batch = None

    return x_batch, y_batch


def evaluate_deep_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Evaluate loss, accuracy, AUC-ROC, precision, and recall on a DataLoader."""
    model.eval()
    total_loss = 0.0
    all_targets: List[float] = []
    all_probs: List[float] = []
    all_preds: List[int] = []
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item())
            n_batches += 1

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = y_batch.cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_targets.extend(targets.tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    y_true = np.array(all_targets)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds)

    mean_loss = total_loss / max(1, n_batches)
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = 0.5

    return {
        "loss": mean_loss,
        "acc": acc,
        "auc": auc,
        "prec": prec,
        "rec": rec,
    }


def train_two_stage(
    model: nn.Module,
    train_dataset: DeepEngagementDataset,
    val_dataset: DeepEngagementDataset,
    batch_size: int = 64,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs_stage1: int = 150,
    patience_stage1: int = 15,
    max_epochs_stage2: int = 60,
    patience_stage2: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute research-grade Two-Stage Training Protocol.

    Stage 1:
      - Adam optimizer, initial lr = 1e-3, weight_decay = 1e-4 (L2).
      - EarlyStopping with patience 15 monitoring val_loss.
      - ReduceLROnPlateau with patience 6, factor 0.5.
    Stage 2 (Refinement):
      - Start from best Stage 1 weights.
      - Adam optimizer with lr = 1e-4.
      - Max 60 epochs, patience 10.
      - Accepted only if best val_loss improves upon Stage 1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Calculate class weight pos_weight
    y_train = train_dataset.y.numpy()
    n_pos = float(np.sum(y_train == 1.0))
    n_neg = float(np.sum(y_train == 0.0))
    pos_weight = n_neg / max(1.0, n_pos)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_batch,
    )

    # -------------------------------------------------------------
    # STAGE 1: Broad Convergence
    # -------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_stage1, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=6, min_lr=1e-5
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_auc": [],
        "val_auc": [],
        "train_prec": [],
        "val_prec": [],
        "train_rec": [],
        "val_rec": [],
    }

    best_val_loss_s1 = float("inf")
    best_epoch_s1 = 0
    best_state_s1: Optional[Dict[str, Any]] = None
    patience_counter = 0

    for epoch in range(1, max_epochs_stage1 + 1):
        model.train()
        total_train_loss = 0.0
        n_train_batches = 0
        tr_preds_list: List[int] = []
        tr_probs_list: List[float] = []
        tr_targets_list: List[float] = []

        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())
            n_train_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = y_b.cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            tr_probs_list.extend(probs.tolist())
            tr_targets_list.extend(targets.tolist())
            tr_preds_list.extend(preds.tolist())

        # Evaluate at epoch end
        val_metrics = evaluate_deep_epoch(model, val_loader, criterion)

        # Vectorized train metrics from training pass
        y_tr_arr = np.array(tr_targets_list)
        p_tr_arr = np.array(tr_probs_list)
        pred_tr_arr = np.array(tr_preds_list)

        tr_loss = total_train_loss / max(1, n_train_batches)
        tr_acc = float(accuracy_score(y_tr_arr, pred_tr_arr))
        tr_prec = float(
            precision_score(y_tr_arr, pred_tr_arr, pos_label=1, zero_division=0)
        )
        tr_rec = float(
            recall_score(y_tr_arr, pred_tr_arr, pos_label=1, zero_division=0)
        )
        try:
            tr_auc = float(roc_auc_score(y_tr_arr, p_tr_arr))
        except Exception:
            tr_auc = 0.5

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_metrics["acc"])
        history["train_auc"].append(tr_auc)
        history["val_auc"].append(val_metrics["auc"])
        history["train_prec"].append(tr_prec)
        history["val_prec"].append(val_metrics["prec"])
        history["train_rec"].append(tr_rec)
        history["val_rec"].append(val_metrics["rec"])

        scheduler.step(val_metrics["loss"])

        if val_metrics["loss"] < best_val_loss_s1:
            best_val_loss_s1 = val_metrics["loss"]
            best_epoch_s1 = epoch
            best_state_s1 = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience_stage1:
                break

    # Restore best Stage 1 weights
    if best_state_s1 is not None:
        model.load_state_dict(best_state_s1)

    stage_1_epochs = len(history["train_loss"])
    stage_2_start_epoch = stage_1_epochs

    # -------------------------------------------------------------
    # STAGE 2: Refinement
    # -------------------------------------------------------------
    optimizer_s2 = torch.optim.Adam(
        model.parameters(), lr=lr_stage2, weight_decay=weight_decay
    )
    best_val_loss_s2 = best_val_loss_s1
    best_epoch_s2 = best_epoch_s1
    best_state_s2 = copy.deepcopy(best_state_s1)
    patience_counter_s2 = 0
    stage_2_improved = False

    s2_history: Dict[str, List[float]] = {k: [] for k in history}

    for epoch_offset in range(1, max_epochs_stage2 + 1):
        current_epoch = stage_1_epochs + epoch_offset
        model.train()
        total_s2_train_loss = 0.0
        n_s2_batches = 0
        s2_preds_list: List[int] = []
        s2_probs_list: List[float] = []
        s2_targets_list: List[float] = []

        for x_b, y_b in train_loader:
            optimizer_s2.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer_s2.step()
            total_s2_train_loss += float(loss.item())
            n_s2_batches += 1

            probs = torch.sigmoid(logits).detach().cpu().numpy()
            targets = y_b.cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            s2_probs_list.extend(probs.tolist())
            s2_targets_list.extend(targets.tolist())
            s2_preds_list.extend(preds.tolist())

        val_metrics = evaluate_deep_epoch(model, val_loader, criterion)

        y_s2_arr = np.array(s2_targets_list)
        p_s2_arr = np.array(s2_probs_list)
        pred_s2_arr = np.array(s2_preds_list)

        tr_s2_loss = total_s2_train_loss / max(1, n_s2_batches)
        tr_s2_acc = float(accuracy_score(y_s2_arr, pred_s2_arr))
        tr_s2_prec = float(
            precision_score(y_s2_arr, pred_s2_arr, pos_label=1, zero_division=0)
        )
        tr_s2_rec = float(
            recall_score(y_s2_arr, pred_s2_arr, pos_label=1, zero_division=0)
        )
        try:
            tr_s2_auc = float(roc_auc_score(y_s2_arr, p_s2_arr))
        except Exception:
            tr_s2_auc = 0.5

        s2_history["train_loss"].append(tr_s2_loss)
        s2_history["val_loss"].append(val_metrics["loss"])
        s2_history["train_acc"].append(tr_s2_acc)
        s2_history["val_acc"].append(val_metrics["acc"])
        s2_history["train_auc"].append(tr_s2_auc)
        s2_history["val_auc"].append(val_metrics["auc"])
        s2_history["train_prec"].append(tr_s2_prec)
        s2_history["val_prec"].append(val_metrics["prec"])
        s2_history["train_rec"].append(tr_s2_rec)
        s2_history["val_rec"].append(val_metrics["rec"])

        if val_metrics["loss"] < best_val_loss_s2:
            best_val_loss_s2 = val_metrics["loss"]
            best_epoch_s2 = current_epoch
            best_state_s2 = copy.deepcopy(model.state_dict())
            stage_2_improved = True
            patience_counter_s2 = 0
        else:
            patience_counter_s2 += 1
            if patience_counter_s2 >= patience_stage2:
                break

    # Concatenate history for complete lifecycle visualization
    for k in history:
        history[k].extend(s2_history[k])

    if stage_2_improved and best_state_s2 is not None:
        model.load_state_dict(best_state_s2)
        stage_2_used = True
        stage_2_note = f"Stage 2 refined best val_loss to {best_val_loss_s2:.4f} at epoch {best_epoch_s2}"
        final_best_loss = best_val_loss_s2
        final_best_epoch = best_epoch_s2
    else:
        if best_state_s1 is not None:
            model.load_state_dict(best_state_s1)
        stage_2_used = False
        stage_2_note = "stage 2 not used"
        final_best_loss = best_val_loss_s1
        final_best_epoch = best_epoch_s1

    # Identify Overfitting Zone: epochs after best_epoch where val_loss > min_loss * 1.03 while train_loss falls
    val_loss_arr = np.array(history["val_loss"])
    train_loss_arr = np.array(history["train_loss"])
    overfitting_zone_indices: List[int] = []

    for ep_idx in range(final_best_epoch, len(val_loss_arr)):
        if (
            val_loss_arr[ep_idx] > final_best_loss * 1.03
            and train_loss_arr[ep_idx] <= train_loss_arr[final_best_epoch - 1]
        ):
            overfitting_zone_indices.append(ep_idx + 1)

    return {
        "model": model,
        "history": history,
        "best_epoch": final_best_epoch,
        "best_val_loss": float(final_best_loss),
        "stage_2_start_epoch": stage_2_start_epoch,
        "stage_2_used": stage_2_used,
        "stage_2_note": stage_2_note,
        "epochs_run": len(history["train_loss"]),
        "overfitting_zone": overfitting_zone_indices,
    }


def predict_proba_deep(
    model: nn.Module,
    dataset: DeepEngagementDataset,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute predicted probabilities for an entire dataset."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )
    all_probs: List[float] = []

    with torch.no_grad():
        for x_b, _ in loader:
            logits = model(x_b)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    return np.array(all_probs, dtype=float)
