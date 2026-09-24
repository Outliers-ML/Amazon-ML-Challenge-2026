"""PyTorch neural network architectures for tabular and multimodal competition tasks.

Includes:
1. `TabularMLP`: Multi-layer perceptron with BatchNorm, Dropout, and Residual skips.
2. `MultimodalFusionNet`: Late-fusion architecture combining text transformer embeddings,
   vision embeddings, and tabular numeric features.
"""

from typing import List, Optional
import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Deep residual multilayer perceptron for tabular features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.2,
    ) -> None:
        """Initialize TabularMLP.

        Args:
            input_dim: Number of input features.
            output_dim: Number of target outputs (1 for binary/regression, >1 for multiclass).
            hidden_dims: List of hidden dimension sizes. Defaults to [256, 128, 64].
            dropout_rate: Dropout probability.
        """
        super().__init__()
        dims = hidden_dims or [256, 128, 64]
        layers = []
        prev_dim = input_dim

        for h_dim in dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.backbone(x)
        return self.head(features)


class MultimodalFusionNet(nn.Module):
    """Late-fusion multimodal network combining text, vision, and tabular inputs."""

    def __init__(
        self,
        text_dim: int = 768,
        image_dim: int = 512,
        tabular_dim: int = 32,
        fusion_dim: int = 256,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
    ) -> None:
        """Initialize MultimodalFusionNet.

        Args:
            text_dim: Dimension of text encoder output embeddings.
            image_dim: Dimension of image encoder output embeddings.
            tabular_dim: Dimension of tabular feature inputs.
            fusion_dim: Bottleneck dimension for joint multimodal representation.
            output_dim: Dimension of final output predictions.
            dropout_rate: Dropout regularization rate.
        """
        super().__init__()

        # Linear projection heads for each modality
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ) if text_dim > 0 else None

        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ) if image_dim > 0 else None

        self.tabular_proj = nn.Sequential(
            nn.Linear(tabular_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        ) if tabular_dim > 0 else None

        # Total fused representation dimension
        n_modalities = sum([self.text_proj is not None, self.image_proj is not None, self.tabular_proj is not None])
        total_dim = n_modalities * fusion_dim

        self.classifier = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, output_dim),
        )

    def forward(
        self,
        text_feats: Optional[torch.Tensor] = None,
        image_feats: Optional[torch.Tensor] = None,
        tab_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass fusing available modalities."""
        representations = []
        if self.text_proj and text_feats is not None:
            representations.append(self.text_proj(text_feats))
        if self.image_proj and image_feats is not None:
            representations.append(self.image_proj(image_feats))
        if self.tabular_proj and tab_feats is not None:
            representations.append(self.tabular_proj(tab_feats))

        if not representations:
            raise ValueError("At least one input modality must be provided to forward().")

        fused = torch.cat(representations, dim=1)
        return self.classifier(fused)
