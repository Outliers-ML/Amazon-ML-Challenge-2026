"""Image feature extraction module.

Extracts dense image embeddings using pretrained Vision models (e.g. EfficientNet,
ResNet, ConvNeXt, or Vision Transformers) via timm / torchvision.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.utils.logger import get_logger

logger = get_logger("ImageFeatures")


class ImagePathDataset(Dataset):
    """Simple PyTorch Dataset loading images from file paths."""

    def __init__(self, paths: List[str | Path], transform=None) -> None:
        self.paths = [str(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        try:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img
        except Exception:
            # Fallback black image
            return torch.zeros((3, 224, 224), dtype=torch.float32)


class ImageEmbeddingExtractor:
    """Extracts dense feature representations from images using pretrained CNN / ViT backbones."""

    def __init__(
        self,
        model_name: str = "resnet34",
        batch_size: int = 64,
        device: str = "cuda",
        num_workers: int = 4,
    ) -> None:
        """Initialize ImageEmbeddingExtractor.

        Args:
            model_name: Backbone architecture from timm or torchvision.
            batch_size: Inference batch size.
            device: 'cuda' or 'cpu'.
            num_workers: DataLoader workers.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.num_workers = num_workers
        self.model = None

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _init_model(self) -> None:
        """Load model dynamically via timm or torchvision."""
        try:
            import timm
            self.model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        except Exception:
            from torchvision.models import resnet34, ResNet34_Weights
            m = resnet34(weights=ResNet34_Weights.DEFAULT)
            m.fc = nn.Identity()
            self.model = m

        self.model.eval().to(self.device)

    def extract_features(self, image_paths: List[str | Path]) -> np.ndarray:
        """Extract dense embeddings for a list of image paths.

        Args:
            image_paths: List of absolute or relative image filepaths.

        Returns:
            2D numpy array of shape (N, feature_dim).
        """
        if self.model is None:
            self._init_model()

        dataset = ImagePathDataset(image_paths, transform=self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        embeddings = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                feats = self.model(batch)
                embeddings.append(feats.cpu().numpy())

        return np.vstack(embeddings)
