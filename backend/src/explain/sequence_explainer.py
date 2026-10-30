"""
Sequence Explainability Engine: Integrated Gradients, Temporal Attention & Permutation Importance.

Provides:
  1. Temporal Attention Weights extraction (for LSTM/GRU + Attention models)
  2. Integrated Gradients attribution across (semester, feature) tensor grid
  3. Semester-by-Feature contribution matrix / heatmap
  4. Permutation importance by feature channel and by semester time-step

Component: Academic Intelligence Framework (IT23164680)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("AcademicIntelligence.SequenceExplainer")

SEQUENCE_FEATURE_NAMES = [
    "GPA (Semester)",
    "Exam Marks Average",
    "Attendance Percentage",
    "Assignment Submission Delay",
]


class SequenceModelExplainer:
    """
    Explainability engine for 3D temporal sequence models (PyTorch).
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.net = getattr(model, "_net", model)
        self.feature_names = feature_names or SEQUENCE_FEATURE_NAMES
        self.device = device or torch.device("cpu")
        self.net.to(self.device)
        self.net.eval()

    def extract_attention_weights(self, data_dict: Dict[str, Any]) -> np.ndarray:
        """
        Extracts temporal soft-attention weights (B, T) for models with attention.
        For models without attention, falls back to normalized gradient saliency across time.
        """
        x = data_dict["tensors"].to(self.device).float()
        mask = data_dict["padding_mask"].to(self.device).bool()
        lengths = data_dict["seq_lengths"].to(self.device).long()

        if hasattr(self.net, "use_attention") and self.net.use_attention and hasattr(self.net, "attn"):
            with torch.no_grad():
                lens_cpu = lengths.cpu().clamp(min=1)
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lens_cpu, batch_first=True, enforce_sorted=False
                )
                out, _ = self.net.rnn(packed)
                out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

                scores = self.net.attn(out).squeeze(-1)  # (B, T)
                scores = scores.masked_fill(~mask[:, :out.shape[1]], float("-inf"))
                weights = torch.softmax(scores, dim=1)  # (B, T)
                return weights.cpu().numpy()
        else:
            # Fallback: Temporal saliency via Integrated Gradients summed over features
            ig_attrs = self.compute_integrated_gradients(data_dict, steps=10)
            temporal_saliency = np.sum(np.abs(ig_attrs), axis=-1)  # (B, T)
            # Normalize per sample
            row_sums = np.sum(temporal_saliency, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return temporal_saliency / row_sums

    def compute_integrated_gradients(
        self,
        data_dict: Dict[str, Any],
        steps: int = 20,
    ) -> np.ndarray:
        """
        Computes Integrated Gradients attribution across the (B, T, F) tensor:
          Attr(x) = (x - x_base) * (1/M) * sum_{m=1}^M grad_x f(x_base + (m/M)*(x - x_base))
        """
        x = data_dict["tensors"].to(self.device).float()
        mask = data_dict["padding_mask"].to(self.device).bool()
        lengths = data_dict["seq_lengths"].to(self.device).long()

        baseline = torch.zeros_like(x)
        diff = x - baseline
        total_grads = torch.zeros_like(x)

        for step in range(1, steps + 1):
            alpha = float(step) / steps
            interpolated = baseline + alpha * diff
            interpolated.requires_grad_(True)

            logits = self.net(interpolated, mask, lengths)
            probs = torch.sigmoid(logits)

            grad = torch.autograd.grad(
                outputs=probs.sum(),
                inputs=interpolated,
                retain_graph=False,
                create_graph=False,
            )[0]
            total_grads += grad

        avg_grads = total_grads / steps
        integrated_grad = (diff * avg_grads).detach().cpu().numpy()
        return integrated_grad

    def get_semester_by_feature_matrix(
        self,
        data_dict: Dict[str, Any],
        steps: int = 15,
    ) -> pd.DataFrame:
        """
        Computes mean absolute attribution across all samples for each (semester, feature) cell.
        Returns a (max_semesters x n_features) DataFrame ready for heatmap visualization.
        """
        ig_attrs = self.compute_integrated_gradients(data_dict, steps=steps)
        mask = data_dict["padding_mask"].cpu().numpy()

        # Zero out padding cells
        for b in range(len(mask)):
            for t in range(mask.shape[1]):
                if not mask[b, t]:
                    ig_attrs[b, t, :] = 0.0

        # Mean absolute attribution across active samples per semester
        mean_attrs = np.mean(np.abs(ig_attrs), axis=0)  # (T, F)

        sem_labels = [f"Semester {t + 1}" for t in range(mean_attrs.shape[0])]
        df_matrix = pd.DataFrame(
            mean_attrs,
            index=sem_labels,
            columns=self.feature_names[:mean_attrs.shape[1]],
        )
        return df_matrix

    def compute_permutation_importance(
        self,
        data_dict: Dict[str, Any],
        y_true: np.ndarray,
        seed: int = 42,
    ) -> Dict[str, pd.DataFrame]:
        """
        Computes permutation importance:
          1. By feature channel (shuffling that channel across all semesters)
          2. By semester time-step (shuffling that semester across all features)
        """
        rng = np.random.default_rng(seed)
        tensors = data_dict["tensors"].clone()
        mask = data_dict["padding_mask"]
        lengths = data_dict["seq_lengths"]

        # Baseline AUROC
        raw_probs = self.model.predict_proba({
            "tensors": tensors, "padding_mask": mask, "seq_lengths": lengths
        })
        base_auc = float(roc_auc_score(y_true, raw_probs))

        # 1. Permutation by Feature Channel
        feat_rows = []
        n_features = tensors.shape[2]
        for f in range(n_features):
            permuted_tensors = tensors.clone()
            idx = rng.permutation(len(tensors))
            permuted_tensors[:, :, f] = permuted_tensors[idx, :, f]

            perm_probs = self.model.predict_proba({
                "tensors": permuted_tensors, "padding_mask": mask, "seq_lengths": lengths
            })
            perm_auc = float(roc_auc_score(y_true, perm_probs))
            drop = max(0.0, base_auc - perm_auc)

            feat_rows.append({
                "channel_index": f,
                "feature_name": self.feature_names[f] if f < len(self.feature_names) else f"Channel {f}",
                "base_auc": round(base_auc, 4),
                "permuted_auc": round(perm_auc, 4),
                "auc_drop": round(drop, 4),
            })

        df_feat = pd.DataFrame(feat_rows).sort_values("auc_drop", reverse=True).reset_index(drop=True)

        # 2. Permutation by Semester Time-step
        sem_rows = []
        n_semesters = tensors.shape[1]
        for t in range(n_semesters):
            permuted_tensors = tensors.clone()
            idx = rng.permutation(len(tensors))
            permuted_tensors[:, t, :] = permuted_tensors[idx, t, :]

            perm_probs = self.model.predict_proba({
                "tensors": permuted_tensors, "padding_mask": mask, "seq_lengths": lengths
            })
            perm_auc = float(roc_auc_score(y_true, perm_probs))
            drop = max(0.0, base_auc - perm_auc)

            sem_rows.append({
                "semester_index": t,
                "semester_label": f"Semester {t + 1}",
                "base_auc": round(base_auc, 4),
                "permuted_auc": round(perm_auc, 4),
                "auc_drop": round(drop, 4),
            })

        df_sem = pd.DataFrame(sem_rows).sort_values("auc_drop", reverse=True).reset_index(drop=True)

        return {
            "by_feature": df_feat,
            "by_semester": df_sem,
        }
