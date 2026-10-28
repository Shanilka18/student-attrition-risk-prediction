"""PyTorch two-stage training loop, early stopping, and regression metric tracking for MEI Track B."""

import copy
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from torch.utils.data import DataLoader

from mei.models.deep.trainer import DeepEngagementDataset, _collate_batch

logger = logging.getLogger(__name__)


def evaluate_deep_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> Dict[str, float]:
    """Evaluate loss, R², RMSE, and MAE on a DataLoader for continuous regression."""
    model.eval()
    total_loss = 0.0
    all_targets: List[float] = []
    all_preds: List[float] = []
    n_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in loader:
            out = model(x_batch).squeeze(-1)
            loss = criterion(out, y_batch)
            total_loss += float(loss.item())
            n_batches += 1

            preds = out.cpu().numpy()
            targets = y_batch.cpu().numpy()
            all_targets.extend(targets.tolist())
            all_preds.extend(preds.tolist())

    y_true = np.array(all_targets)
    y_pred = np.clip(np.array(all_preds), 0.0, 100.0)

    mean_loss = total_loss / max(1, n_batches)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(root_mean_squared_error(y_true, y_pred))

    return {
        "loss": mean_loss,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
    }


def train_two_stage_regression(
    model: nn.Module,
    train_dataset: DeepEngagementDataset,
    val_dataset: DeepEngagementDataset,
    batch_size: int = 64,
    lr_stage1: float = 1e-3,
    lr_stage2: float = 1e-4,
    weight_decay: float = 1e-4,
    max_epochs_stage1: int = 80,
    max_epochs_stage2: int = 40,
    patience_stage1: int = 12,
    patience_stage2: int = 8,
    lr_reduce_patience: int = 4,
    lr_reduce_factor: float = 0.5,
    min_lr: float = 1e-6,
    huber_delta: float = 2.0,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """Execute two-stage training loop for deep regression architectures on continuous intent score."""
    torch.manual_seed(random_seed)
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

    criterion = nn.HuberLoss(delta=huber_delta)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_r2": [],
        "val_loss": [],
        "val_r2": [],
        "val_mae": [],
        "val_rmse": [],
        "learning_rate": [],
    }

    best_val_loss = float("inf")
    best_weights: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0

    # -------------------------------------------------------------
    # STAGE 1: Standard Coarse Optimization
    # -------------------------------------------------------------
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr_stage1, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_reduce_factor,
        patience=lr_reduce_patience,
        min_lr=min_lr,
    )

    no_improve_count = 0

    for epoch in range(1, max_epochs_stage1 + 1):
        model.train()
        epoch_train_loss = 0.0
        n_batches = 0
        train_targets = []
        train_preds = []

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch).squeeze(-1)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            epoch_train_loss += float(loss.item())
            n_batches += 1
            train_targets.extend(y_batch.detach().cpu().numpy().tolist())
            train_preds.extend(out.detach().cpu().numpy().tolist())

        mean_train_loss = epoch_train_loss / max(1, n_batches)
        y_tr_clipped = np.clip(np.array(train_preds), 0.0, 100.0)
        train_r2 = (
            float(r2_score(np.array(train_targets), y_tr_clipped))
            if len(train_targets) > 1
            else 0.0
        )

        val_metrics = evaluate_deep_regression_epoch(model, val_loader, criterion)
        scheduler.step(val_metrics["loss"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(mean_train_loss)
        history["train_r2"].append(train_r2)
        history["val_loss"].append(val_metrics["loss"])
        history["val_r2"].append(val_metrics["r2"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_rmse"].append(val_metrics["rmse"])
        history["learning_rate"].append(current_lr)

        if val_metrics["loss"] < best_val_loss - 1e-4:
            best_val_loss = val_metrics["loss"]
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= patience_stage1:
            break

    # Restore best checkpoint from Stage 1
    if best_weights is not None:
        model.load_state_dict(best_weights)

    stage_1_end_epoch = len(history["train_loss"])
    stage_1_best_val_loss = best_val_loss
    stage_2_used = False
    stage_2_note = "Stage 2 skipped"

    # -------------------------------------------------------------
    # STAGE 2: Conditional Fine-Tuning
    # -------------------------------------------------------------
    if max_epochs_stage2 > 0 and stage_1_best_val_loss < float("inf"):
        stage_2_weights_backup = copy.deepcopy(model.state_dict())
        optimizer_s2 = torch.optim.Adam(
            model.parameters(), lr=lr_stage2, weight_decay=weight_decay
        )
        no_improve_count_s2 = 0
        stage_2_improved = False

        for s2_epoch in range(1, max_epochs_stage2 + 1):
            global_epoch = stage_1_end_epoch + s2_epoch
            model.train()
            epoch_train_loss = 0.0
            n_batches = 0
            train_targets = []
            train_preds = []

            for x_batch, y_batch in train_loader:
                optimizer_s2.zero_grad()
                out = model(x_batch).squeeze(-1)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer_s2.step()

                epoch_train_loss += float(loss.item())
                n_batches += 1
                train_targets.extend(y_batch.detach().cpu().numpy().tolist())
                train_preds.extend(out.detach().cpu().numpy().tolist())

            mean_train_loss = epoch_train_loss / max(1, n_batches)
            y_tr_clipped = np.clip(np.array(train_preds), 0.0, 100.0)
            train_r2 = (
                float(r2_score(np.array(train_targets), y_tr_clipped))
                if len(train_targets) > 1
                else 0.0
            )

            val_metrics = evaluate_deep_regression_epoch(model, val_loader, criterion)

            history["train_loss"].append(mean_train_loss)
            history["train_r2"].append(train_r2)
            history["val_loss"].append(val_metrics["loss"])
            history["val_r2"].append(val_metrics["r2"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["learning_rate"].append(lr_stage2)

            if val_metrics["loss"] < best_val_loss - 1e-4:
                best_val_loss = val_metrics["loss"]
                best_weights = copy.deepcopy(model.state_dict())
                best_epoch = global_epoch
                stage_2_improved = True
                no_improve_count_s2 = 0
            else:
                no_improve_count_s2 += 1

            if no_improve_count_s2 >= patience_stage2:
                break

        if stage_2_improved and best_weights is not None:
            model.load_state_dict(best_weights)
            stage_2_used = True
            stage_2_note = f"Stage 2 refined best val_loss from {stage_1_best_val_loss:.4f} to {best_val_loss:.4f} at epoch {best_epoch}"
        else:
            # Conditional rollback: Stage 2 did not improve upon Stage 1
            model.load_state_dict(stage_2_weights_backup)
            stage_2_used = False
            stage_2_note = f"Stage 2 did not improve on Stage 1 (best: {stage_1_best_val_loss:.4f}); rolled back cleanly"

    # -------------------------------------------------------------
    # Overfitting Zone Detection
    # -------------------------------------------------------------
    overfitting_zone: List[int] = []
    window = 3
    t_losses = history["train_loss"]
    v_losses = history["val_loss"]

    for i in range(window, len(t_losses)):
        t_drop = t_losses[i] < t_losses[i - window]
        v_rise = v_losses[i] > v_losses[i - window]
        if t_drop and v_rise:
            overfitting_zone.append(i + 1)

    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "stage_1_end_epoch": stage_1_end_epoch,
        "stage_2_used": stage_2_used,
        "stage_2_note": stage_2_note,
        "overfitting_zone": overfitting_zone,
    }


def predict_deep_regressor(
    model: nn.Module,
    dataset: DeepEngagementDataset,
    batch_size: int = 64,
) -> np.ndarray:
    """Generate raw model predictions on dataset."""
    model.eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate_batch
    )
    preds: List[float] = []

    with torch.no_grad():
        for x_batch, _ in loader:
            out = model(x_batch).squeeze(-1)
            preds.extend(out.cpu().numpy().tolist())

    return np.array(preds, dtype=np.float64)
