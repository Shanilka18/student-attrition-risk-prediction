"""Comprehensive 15-fold cross-validation and benchmarking engine for PyTorch deep models."""

import io
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from mei.evaluation.metrics import (
    compute_leaderboard_score,
    evaluate_predictions,
    f1_score,
    roc_auc_score,
)
from mei.models.deep.architectures import build_deep_model
from mei.models.deep.trainer import (
    DeepEngagementDataset,
    evaluate_deep_epoch,
    predict_proba_deep,
    train_two_stage,
)
from mei.models.deep.tuner import tune_deep_model
from mei.models.registry import ModelSpec

logger = logging.getLogger(__name__)


def prepare_fold_datasets(
    spec: ModelSpec,
    X_dict: Dict[str, Any],
    tr_idx: np.ndarray,
    val_idx: np.ndarray,
    y: np.ndarray,
) -> Tuple[
    DeepEngagementDataset, DeepEngagementDataset, Dict[str, StandardScaler], int, int
]:
    """Fit scalers on the training fold strictly and build PyTorch datasets."""
    y_tr, y_val = y[tr_idx], y[val_idx]
    scalers: Dict[str, StandardScaler] = {}
    tab_dim = 126
    seq_dim = 7

    if spec.feature_set == "sequence":
        raw_seq_tr = X_dict["sequence"][tr_idx]
        raw_seq_v = X_dict["sequence"][val_idx]
        n_tr, t, d = raw_seq_tr.shape
        seq_dim = d

        scaler = StandardScaler()
        scaled_seq_tr = (
            scaler.fit_transform(raw_seq_tr.reshape(-1, d))
            .reshape(n_tr, t, d)
            .astype(np.float32)
        )
        scaled_seq_v = (
            scaler.transform(raw_seq_v.reshape(-1, d))
            .reshape(len(raw_seq_v), t, d)
            .astype(np.float32)
        )
        scalers["seq"] = scaler

        ds_tr = DeepEngagementDataset(X_seq=scaled_seq_tr, y=y_tr)
        ds_val = DeepEngagementDataset(X_seq=scaled_seq_v, y=y_val)

    elif spec.feature_set == "hybrid_multimodal":
        raw_seq_tr = X_dict["sequence"][tr_idx]
        raw_seq_v = X_dict["sequence"][val_idx]
        raw_tab_tr = X_dict["agg_states"][tr_idx]
        raw_tab_v = X_dict["agg_states"][val_idx]

        n_tr, t, d = raw_seq_tr.shape
        seq_dim = d
        seq_scaler = StandardScaler()
        scaled_seq_tr = (
            seq_scaler.fit_transform(raw_seq_tr.reshape(-1, d))
            .reshape(n_tr, t, d)
            .astype(np.float32)
        )
        scaled_seq_v = (
            seq_scaler.transform(raw_seq_v.reshape(-1, d))
            .reshape(len(raw_seq_v), t, d)
            .astype(np.float32)
        )
        scalers["seq"] = seq_scaler

        tab_scaler = StandardScaler()
        scaled_tab_tr = tab_scaler.fit_transform(raw_tab_tr).astype(np.float32)
        scaled_tab_v = tab_scaler.transform(raw_tab_v).astype(np.float32)
        scalers["tab"] = tab_scaler
        tab_dim = scaled_tab_tr.shape[1]

        ds_tr = DeepEngagementDataset(X_tab=scaled_tab_tr, X_seq=scaled_seq_tr, y=y_tr)
        ds_val = DeepEngagementDataset(X_tab=scaled_tab_v, X_seq=scaled_seq_v, y=y_val)

    else:  # agg or agg_states
        raw_tab_tr = X_dict[spec.feature_set][tr_idx]
        raw_tab_v = X_dict[spec.feature_set][val_idx]

        tab_scaler = StandardScaler()
        scaled_tab_tr = tab_scaler.fit_transform(raw_tab_tr).astype(np.float32)
        scaled_tab_v = tab_scaler.transform(raw_tab_v).astype(np.float32)
        scalers["tab"] = tab_scaler
        tab_dim = scaled_tab_tr.shape[1]

        ds_tr = DeepEngagementDataset(X_tab=scaled_tab_tr, y=y_tr)
        ds_val = DeepEngagementDataset(X_tab=scaled_tab_v, y=y_val)

    return ds_tr, ds_val, scalers, tab_dim, seq_dim


def evaluate_single_deep_model(
    spec: ModelSpec,
    X_dict: Dict[str, Any],
    y: np.ndarray,
    cv_folds: List[Dict[str, List[int]]],
    seed: int = 42,
    tuned_params: Optional[Dict[str, Any]] = None,
    quick: bool = False,
) -> Dict[str, Any]:
    """Train and benchmark a single deep model across identical 15 CV folds."""
    logger.info(
        f"Evaluating deep model: {spec.id} ({spec.display_name}) across {len(cv_folds)} folds..."
    )
    t_start = time.time()
    n_samples = len(y)

    # 1. Hyperparameter Optimization on Fold 0 training partition
    if tuned_params is not None:
        best_params = tuned_params
        tuning_summary = {"preset": True, "params": best_params}
    else:
        fold_0_train_idx = np.array(
            cv_folds[0].get("train_idx", cv_folds[0].get("train_indices"))
        )
        n_trials = 3 if quick else spec.n_tuning_trials
        best_params, tuning_summary = tune_deep_model(
            spec=spec,
            X_dict={
                k: v[fold_0_train_idx] if hasattr(v, "__getitem__") else v
                for k, v in X_dict.items()
            },
            y=y[fold_0_train_idx],
            n_trials=n_trials,
            time_budget=spec.time_budget_seconds,
            seed=seed,
        )
        print(f"      [Tuning Complete] Best params: {best_params}", flush=True)

    # 2. Out-of-Fold Evaluation Across All 15 Outer Folds
    oof_probs_accum = np.zeros(n_samples, dtype=float)
    oof_counts = np.zeros(n_samples, dtype=int)
    fold_metrics: List[Dict[str, Any]] = []
    fold_histories: List[Dict[str, Any]] = []

    batch_size = int(best_params.get("batch_size", 64))
    lr = float(best_params.get("lr", 1e-3))
    weight_decay = float(best_params.get("weight_decay", 1e-4))

    for f_idx, fold in enumerate(cv_folds):
        tr_idx = np.array(fold.get("train_idx", fold.get("train_indices")))
        val_idx = np.array(fold.get("val_idx", fold.get("val_indices")))

        ds_tr, ds_val, scalers, tab_dim, seq_dim = prepare_fold_datasets(
            spec, X_dict, tr_idx, val_idx, y
        )

        model = build_deep_model(
            spec.id,
            tab_input_dim=tab_dim,
            seq_input_dim=seq_dim,
            hyperparams=best_params,
        )

        # Train model with two-stage protocol
        train_res = train_two_stage(
            model=model,
            train_dataset=ds_tr,
            val_dataset=ds_val,
            batch_size=batch_size,
            lr_stage1=lr,
            lr_stage2=lr * 0.1,
            weight_decay=weight_decay,
            max_epochs_stage1=150,
            patience_stage1=15,
            max_epochs_stage2=60,
            patience_stage2=10,
            seed=seed + f_idx * 17,
        )

        fold_histories.append(train_res)

        # Predict probabilities with best restored weights
        p_tr = predict_proba_deep(model, ds_tr, batch_size=batch_size)
        p_val = predict_proba_deep(model, ds_val, batch_size=batch_size)

        oof_probs_accum[val_idx] += p_val
        oof_counts[val_idx] += 1

        y_tr, y_v = y[tr_idx], y[val_idx]
        pred_tr = (p_tr >= 0.5).astype(int)
        pred_v = (p_val >= 0.5).astype(int)

        f_tr_f1 = float(f1_score(y_tr, pred_tr, pos_label=1, zero_division=0))
        f_val_f1 = float(f1_score(y_v, pred_v, pos_label=1, zero_division=0))
        try:
            f_tr_auc = float(roc_auc_score(y_tr, p_tr))
        except Exception:
            f_tr_auc = 0.5
        try:
            f_val_auc = float(roc_auc_score(y_v, p_val))
        except Exception:
            f_val_auc = 0.5

        fold_metrics.append(
            {
                "fold": f_idx,
                "train_f1": f_tr_f1,
                "val_f1": f_val_f1,
                "train_auc": f_tr_auc,
                "val_auc": f_val_auc,
                "best_epoch": train_res["best_epoch"],
                "best_val_loss": train_res["best_val_loss"],
                "stage_2_used": train_res["stage_2_used"],
                "gap": f_tr_f1 - f_val_f1,
            }
        )

        if (f_idx + 1) % 5 == 0 or f_idx == 0:
            print(
                f"      -> Fold {f_idx + 1:2d}/{len(cv_folds)}: Val F1={f_val_f1:.4f}, AUC={f_val_auc:.4f}, Loss={train_res['best_val_loss']:.4f} (Ep {train_res['best_epoch']})",
                flush=True,
            )

    # 3. Final Aggregated OOF Probabilities
    nonzero_mask = oof_counts > 0
    oof_probs = np.zeros(n_samples, dtype=float)
    oof_probs[nonzero_mask] = oof_probs_accum[nonzero_mask] / oof_counts[nonzero_mask]

    # Full Evaluation on OOF Predictions
    full_eval = evaluate_predictions(y, oof_probs, optimal_threshold=None)

    # 4. Compute Generalization Gap and Leaderboard Score
    val_f1 = full_eval.metrics_optimal.f1
    val_auc = full_eval.probabilistic.auc_roc
    val_bal_acc = full_eval.metrics_optimal.balanced_accuracy
    val_mcc = full_eval.metrics_optimal.mcc
    train_f1_mean = float(np.mean([f["train_f1"] for f in fold_metrics]))

    lb_score, gen_gap, penalty, status = compute_leaderboard_score(
        val_f1=val_f1,
        val_auc_roc=val_auc,
        val_balanced_acc=val_bal_acc,
        val_mcc=val_mcc,
        train_f1=train_f1_mean,
    )

    # 5. Fit Final Full-Cohort Model (for production inference & latency)
    all_idx = np.arange(n_samples)
    ds_full, ds_subval, scalers_full, tab_dim_f, seq_dim_f = prepare_fold_datasets(
        spec, X_dict, all_idx, all_idx[:300], y
    )
    final_model = build_deep_model(
        spec.id,
        tab_input_dim=tab_dim_f,
        seq_input_dim=seq_dim_f,
        hyperparams=best_params,
    )
    _ = train_two_stage(
        model=final_model,
        train_dataset=ds_full,
        val_dataset=ds_subval,
        batch_size=batch_size,
        lr_stage1=lr,
        lr_stage2=lr * 0.1,
        weight_decay=weight_decay,
        max_epochs_stage1=40,
        patience_stage1=10,
        max_epochs_stage2=15,
        patience_stage2=5,
        seed=seed,
    )

    # 6. Benchmark Inference Latency & Model Size
    sample_1000 = (
        ds_full.X_tab[:1000] if ds_full.X_tab is not None else ds_full.X_seq[:1000]
    )
    if len(sample_1000) < 1000:
        repeats = int(np.ceil(1000 / len(sample_1000)))
        if ds_full.X_tab is not None and ds_full.X_seq is not None:
            t_sub = (
                torch.tile(ds_full.X_tab, (repeats, 1))[:1000],
                torch.tile(ds_full.X_seq, (repeats, 1, 1))[:1000],
            )
            sample_ds = DeepEngagementDataset(
                X_tab=t_sub[0].numpy(), X_seq=t_sub[1].numpy()
            )
        elif ds_full.X_seq is not None:
            s_sub = torch.tile(ds_full.X_seq, (repeats, 1, 1))[:1000]
            sample_ds = DeepEngagementDataset(X_seq=s_sub.numpy())
        else:
            t_sub = torch.tile(ds_full.X_tab, (repeats, 1))[:1000]
            sample_ds = DeepEngagementDataset(X_tab=t_sub.numpy())
    else:
        sample_ds = ds_full

    t_inf_0 = time.time()
    _ = predict_proba_deep(final_model, sample_ds, batch_size=batch_size)
    inference_time_ms = float((time.time() - t_inf_0) * 1000.0)

    buf = io.BytesIO()
    torch.save(final_model.state_dict(), buf)
    serialized_size_bytes = int(buf.tell())

    training_time_s = float(time.time() - t_start)

    # 7. Cross-Fold Epoch Aggregation (Mean ± 1 SD curves across 15 folds)
    max_epochs_seen = max(len(h["history"]["train_loss"]) for h in fold_histories)
    curve_keys = [
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_auc",
        "val_auc",
    ]
    mean_curves: Dict[str, List[float]] = {}
    std_curves: Dict[str, List[float]] = {}

    for k in curve_keys:
        epoch_matrix = np.full((len(fold_histories), max_epochs_seen), np.nan)
        for i_f, h in enumerate(fold_histories):
            vals = h["history"][k]
            epoch_matrix[i_f, : len(vals)] = vals
            if len(vals) < max_epochs_seen:
                # Forward-fill final value
                epoch_matrix[i_f, len(vals) :] = vals[-1]

        mean_curves[k] = [float(val) for val in np.nanmean(epoch_matrix, axis=0)]
        std_curves[k] = [float(val) for val in np.nanstd(epoch_matrix, axis=0)]

    # Representative fold history (Fold 0)
    rep_history = fold_histories[0]

    leaderboard_row = {
        "model_id": spec.id,
        "display_name": spec.display_name,
        "plain_name": spec.plain_name,
        "category": spec.category,
        "feature_set": spec.feature_set,
        "leaderboard_score": float(lb_score),
        "generalization_gap": float(gen_gap),
        "overfit_penalty": float(penalty),
        "overfit_status": status,
        "val_f1_optimal": float(val_f1),
        "val_auc_roc": float(val_auc),
        "val_balanced_accuracy": float(val_bal_acc),
        "val_mcc": float(val_mcc),
        "val_precision": float(full_eval.metrics_optimal.precision),
        "val_recall": float(full_eval.metrics_optimal.recall),
        "val_specificity": float(full_eval.metrics_optimal.specificity),
        "val_pr_auc": float(full_eval.probabilistic.pr_auc),
        "val_brier_score": float(full_eval.probabilistic.brier_score),
        "val_ece": float(full_eval.probabilistic.ece),
        "train_f1_mean": float(train_f1_mean),
        "optimal_threshold": float(full_eval.optimal_threshold),
        "training_time_s": float(training_time_s),
        "inference_time_ms": float(inference_time_ms),
        "model_size_kb": round(float(serialized_size_bytes / 1024.0), 1),
    }

    result = {
        "spec": {
            "id": spec.id,
            "display_name": spec.display_name,
            "plain_name": spec.plain_name,
            "category": spec.category,
            "feature_set": spec.feature_set,
            "needs_scaling": spec.needs_scaling,
            "supports_class_weight": spec.supports_class_weight,
        },
        "hyperparameters": best_params,
        "tuning_summary": tuning_summary,
        "leaderboard_metrics": leaderboard_row,
        "fold_metrics": fold_metrics,
        "confusion_matrix_raw": full_eval.metrics_optimal.confusion_matrix_raw,
        "confusion_matrix_norm": full_eval.metrics_optimal.confusion_matrix_norm,
        "oof_predictions": [float(p) for p in oof_probs],
        "final_model": final_model,
        "scalers": scalers_full,
        "curves": {
            "roc": {
                "fpr": [float(x) for x in full_eval.curves.roc_fpr],
                "tpr": [float(y_) for y_ in full_eval.curves.roc_tpr],
            },
            "pr": {
                "recall": [float(r) for r in full_eval.curves.pr_recall],
                "precision": [float(p) for p in full_eval.curves.pr_precision],
            },
            "calibration": {
                "prob_true": [float(pt) for pt in full_eval.curves.cal_prob_true],
                "prob_pred": [float(pp) for pp in full_eval.curves.cal_prob_pred],
            },
        },
        "representative_fold": {
            "history": rep_history["history"],
            "best_epoch": rep_history["best_epoch"],
            "best_val_loss": rep_history["best_val_loss"],
            "stage_2_start_epoch": rep_history["stage_2_start_epoch"],
            "stage_2_used": rep_history["stage_2_used"],
            "stage_2_note": rep_history["stage_2_note"],
            "overfitting_zone": rep_history["overfitting_zone"],
        },
        "cross_fold_curves": {
            "mean": mean_curves,
            "std": std_curves,
            "epochs": list(range(1, max_epochs_seen + 1)),
        },
    }

    return result


def train_overfit_demonstration_model(
    X_tab: np.ndarray,
    y: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train deliberately unregularized OverparameterizedMLP for 300 epochs.

    Demonstrates classic severe overfitting: zero dropout, zero weight decay, 512-256-128 units,
    no early stopping. Produces the viva centerpiece figure comparing regularized vs unregularized learning.
    """
    logger.info(
        "Training deliberately over-parameterized model for overfitting demonstration..."
    )
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 70/30 split on cohort
    n = len(y)
    split_pt = int(0.7 * n)
    tr_idx = np.arange(split_pt)
    val_idx = np.arange(split_pt, n)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tab[tr_idx]).astype(np.float32)
    X_val = scaler.transform(X_tab[val_idx]).astype(np.float32)

    ds_tr = DeepEngagementDataset(X_tab=X_tr, y=y[tr_idx])
    ds_val = DeepEngagementDataset(X_tab=X_val, y=y[val_idx])

    model = build_deep_model("overfit_demo", tab_input_dim=X_tr.shape[1])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0.0
    )  # L2 = 0
    criterion = nn.BCEWithLogitsLoss()

    train_loader = torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=False)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(1, 301):
        model.train()
        for x_b, y_b in train_loader:
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

        tr_metrics = evaluate_deep_epoch(model, train_loader, criterion)
        val_metrics = evaluate_deep_epoch(model, val_loader, criterion)

        history["train_loss"].append(tr_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(tr_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

    return {
        "history": history,
        "epochs": list(range(1, 301)),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "min_val_loss": float(np.min(history["val_loss"])),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
    }


def compute_seed_variance_deep(
    model_id: str,
    spec: ModelSpec,
    X_dict: Dict[str, Any],
    y: np.ndarray,
    cv_folds: List[Dict[str, List[int]]],
    best_params: Dict[str, Any],
    seeds: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Train the model across 5 seeds and evaluate seed standard deviation."""
    if seeds is None:
        seeds = [42, 100, 2024, 777, 999]

    seed_f1s: List[float] = []
    seed_aucs: List[float] = []
    seed_scores: List[float] = []

    for s in seeds:
        # Evaluate on Fold 0
        fold = cv_folds[0]
        tr_idx = np.array(fold.get("train_idx", fold.get("train_indices")))
        val_idx = np.array(fold.get("val_idx", fold.get("val_indices")))

        ds_tr, ds_val, _, tab_dim, seq_dim = prepare_fold_datasets(
            spec, X_dict, tr_idx, val_idx, y
        )
        model = build_deep_model(
            model_id,
            tab_input_dim=tab_dim,
            seq_input_dim=seq_dim,
            hyperparams=best_params,
        )

        _ = train_two_stage(
            model=model,
            train_dataset=ds_tr,
            val_dataset=ds_val,
            batch_size=int(best_params.get("batch_size", 64)),
            lr_stage1=float(best_params.get("lr", 1e-3)),
            lr_stage2=float(best_params.get("lr", 1e-3)) * 0.1,
            weight_decay=float(best_params.get("weight_decay", 1e-4)),
            max_epochs_stage1=100,
            patience_stage1=12,
            max_epochs_stage2=40,
            patience_stage2=8,
            seed=s,
        )

        p_val = predict_proba_deep(
            model, ds_val, batch_size=int(best_params.get("batch_size", 64))
        )
        pred_val = (p_val >= 0.5).astype(int)
        y_v = y[val_idx]

        f1 = float(f1_score(y_v, pred_val, pos_label=1, zero_division=0))
        try:
            auc = float(roc_auc_score(y_v, p_val))
        except Exception:
            auc = 0.5

        seed_f1s.append(f1)
        seed_aucs.append(auc)
        seed_scores.append((f1 + auc) / 2.0)

    return {
        "model_id": model_id,
        "seeds": seeds,
        "f1_mean": float(np.mean(seed_f1s)),
        "f1_std": float(np.std(seed_f1s)),
        "auc_mean": float(np.mean(seed_aucs)),
        "auc_std": float(np.std(seed_aucs)),
        "seed_f1s": [float(x) for x in seed_f1s],
        "seed_aucs": [float(x) for x in seed_aucs],
    }
