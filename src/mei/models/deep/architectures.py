"""PyTorch neural network architectures for Multimodal Student Engagement Intelligence."""

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class GaussianNoise(nn.Module):
    """Additive Gaussian noise injection layer active strictly during training."""

    def __init__(self, sigma: float = 0.05) -> None:
        super().__init__()
        self.sigma = sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.sigma > 0.0:
            return x + torch.randn_like(x) * self.sigma
        return x


class MLPWide(nn.Module):
    """Wide Multilayer Perceptron with Batch Normalization and Dropout.

    Architecture: Dense(128)-BN-ReLU-Dropout -> Dense(64)-BN-ReLU-Dropout -> Dense(32)-BN-ReLU-Dropout -> Linear(1).
    Evaluated primarily on tabular aggregates or aggregate state features (agg_states).
    """

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        h1 = hidden_dim
        h2 = max(16, hidden_dim // 2)
        h3 = max(8, hidden_dim // 4)

        self.noise = GaussianNoise(sigma=noise_std)
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.BatchNorm1d(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.BatchNorm1d(h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, h3),
            nn.BatchNorm1d(h3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h3, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        return self.net(x_noisy).squeeze(-1)


class ResidualBlock(nn.Module):
    """Pre-activation or post-activation Residual Block with Layer Normalization."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class MLPResidual(nn.Module):
    """Residual Multilayer Perceptron with 2 skip-connected ResNet blocks and Layer Normalization."""

    def __init__(
        self,
        input_dim: int = 126,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        h = self.in_proj(x_noisy)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h).squeeze(-1)


class GRUModel(nn.Module):
    """Gated Recurrent Unit (GRU) for short 4-period longitudinal engagement sequences."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        _, h_n = self.gru(x_noisy)  # h_n shape: (1, B, hidden_dim)
        h_last = h_n[-1]  # shape: (B, hidden_dim)
        return self.head(h_last).squeeze(-1)


class LSTMModel(nn.Module):
    """Long Short-Term Memory (LSTM) sequence model."""

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        _, (h_n, _) = self.lstm(x_noisy)  # h_n shape: (1, B, hidden_dim)
        h_last = h_n[-1]
        return self.head(h_last).squeeze(-1)


class BiGRUModel(nn.Module):
    """Bidirectional Gated Recurrent Unit (Bi-GRU).

    Justification over Bi-LSTM: On 4-period sequences, GRU eliminates cell state gating
    redundancy, reducing parameter footprint by 25% and mitigating overparameterization.
    """

    def __init__(
        self,
        input_dim: int = 7,
        hidden_dim: int = 24,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        out, _ = self.gru(x_noisy)  # out shape: (B, 4, hidden_dim * 2)
        pooled = out.mean(dim=1)  # Average pooling over time steps
        return self.head(pooled).squeeze(-1)


class CNN1DModel(nn.Module):
    """1D Temporal Convolutional Neural Network with Global Max Pooling."""

    def __init__(
        self,
        input_dim: int = 7,
        filters: int = 32,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)
        # Conv1d expects (batch, in_channels, seq_len)
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=filters, kernel_size=2, padding="same"
        )
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            in_channels=filters, out_channels=filters, kernel_size=2, padding="same"
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(filters, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        # x is (B, 4, 7); permute to (B, 7, 4)
        x_conv = x_noisy.permute(0, 2, 1)
        h = self.drop(self.relu(self.conv1(x_conv)))
        h = self.relu(self.conv2(h))
        pooled = self.pool(h).squeeze(-1)  # (B, filters)
        return self.head(pooled).squeeze(-1)


class TinyTransformer(nn.Module):
    """Tiny Transformer Encoder with Multi-Head Self-Attention for 4-period sequences."""

    def __init__(
        self,
        input_dim: int = 7,
        d_model: int = 32,
        num_layers: int = 1,
        nhead: int = 2,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)
        self.in_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 4, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        x_noisy = self.noise(x)
        tokens = self.in_proj(x_noisy) + self.pos_embedding
        encoded = self.transformer(tokens)  # (B, 4, d_model)
        pooled = encoded.mean(dim=1)  # Global token pooling
        return self.head(pooled).squeeze(-1)


class HybridMultimodalModel(nn.Module):
    """Dual-Branch Hybrid Multimodal Deep Network.

    Combines temporal sequence dynamics (via sequential GRU) with longitudinal aggregate
    and engagement state features (via tabular MLP) into a unified attrition risk head.
    """

    def __init__(
        self,
        seq_input_dim: int = 7,
        tab_input_dim: int = 126,
        fusion_dim: int = 16,
        dropout: float = 0.2,
        noise_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.noise = GaussianNoise(sigma=noise_std)

        # Branch A: Sequence branch
        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=24,
            batch_first=True,
            bidirectional=False,
        )
        self.seq_proj = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(24, 16),
            nn.ReLU(),
        )

        # Branch B: Tabular state + aggregate branch
        self.tab_proj = nn.Sequential(
            nn.Linear(tab_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.ReLU(),
        )

        # Fusion Head: Concat(16 + 16 = 32) -> Head
        self.fusion = nn.Sequential(
            nn.Linear(32, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 1),
        )

    def forward(
        self, x: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x_seq, x_tab = x[0], x[1]
        else:
            raise ValueError(
                "HybridMultimodalModel requires a tuple (x_seq, x_tab) as input."
            )

        x_seq_noisy = self.noise(x_seq)
        x_tab_noisy = self.noise(x_tab)

        # Sequence embedding
        _, h_n = self.gru(x_seq_noisy)
        z_seq = self.seq_proj(h_n[-1])

        # Tabular embedding
        z_tab = self.tab_proj(x_tab_noisy)

        # Concatenate and fuse
        fused = torch.cat([z_seq, z_tab], dim=-1)
        return self.fusion(fused).squeeze(-1)


class OverparameterizedMLP(nn.Module):
    """Deliberately over-parameterized unregularized MLP for overfitting demonstration.

    Used specifically for research figures comparing regularized vs unregularized learning curves.
    No dropout, no batch normalization, no L2 decay, no early stopping (300 full epochs).
    """

    def __init__(self, input_dim: int = 126) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self.net(x).squeeze(-1)


def build_deep_model(
    model_id: str,
    tab_input_dim: int = 126,
    seq_input_dim: int = 7,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Factory creating an instance of the requested deep model architecture."""
    params = hyperparams or {}
    dropout = float(params.get("dropout", 0.2))
    noise_std = float(params.get("noise_std", 0.05))

    if model_id == "mlp_wide":
        hidden_dim = int(params.get("hidden_dim", 128))
        return MLPWide(
            input_dim=tab_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "mlp_residual":
        hidden_dim = int(params.get("hidden_dim", 64))
        return MLPResidual(
            input_dim=tab_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "gru":
        hidden_dim = int(params.get("hidden_dim", 32))
        return GRUModel(
            input_dim=seq_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "lstm":
        hidden_dim = int(params.get("hidden_dim", 32))
        return LSTMModel(
            input_dim=seq_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "bi_gru":
        hidden_dim = int(params.get("hidden_dim", 24))
        return BiGRUModel(
            input_dim=seq_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "cnn_1d":
        filters = int(params.get("filters", 32))
        return CNN1DModel(
            input_dim=seq_input_dim,
            filters=filters,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "tiny_transformer":
        d_model = int(params.get("d_model", 32))
        num_layers = int(params.get("num_layers", 1))
        return TinyTransformer(
            input_dim=seq_input_dim,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "hybrid_multimodal":
        fusion_dim = int(params.get("fusion_dim", 16))
        return HybridMultimodalModel(
            seq_input_dim=seq_input_dim,
            tab_input_dim=tab_input_dim,
            fusion_dim=fusion_dim,
            dropout=dropout,
            noise_std=noise_std,
        )
    elif model_id == "overfit_demo":
        return OverparameterizedMLP(input_dim=tab_input_dim)
    else:
        raise ValueError(f"Unknown deep model architecture ID: '{model_id}'")
