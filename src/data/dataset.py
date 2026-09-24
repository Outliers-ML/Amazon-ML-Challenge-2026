"""PyTorch Dataset implementations for tabular and multimodal competition tasks.

Provides efficient PyTorch `Dataset` wrappers supporting:
1. Pure tabular tensors (continuous + categorical)
2. Multimodal data: Text tokens, Image tensors, and Tabular features simultaneously.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class TabularDataset(Dataset):
    """PyTorch Dataset wrapper for tabular numerical and categorical features."""

    def __init__(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        targets: Optional[Union[pd.Series, np.ndarray]] = None,
        is_classification: bool = True,
    ) -> None:
        """Initialize TabularDataset.

        Args:
            features: 2D array or DataFrame of features.
            targets: Optional 1D array or Series of target values.
            is_classification: True for classification labels (long/int), False for regression (float).
        """
        if isinstance(features, pd.DataFrame):
            self.features = features.values.astype(np.float32)
        else:
            self.features = np.asarray(features, dtype=np.float32)

        self.targets = None
        if targets is not None:
            if isinstance(targets, pd.Series):
                y_arr = targets.values
            else:
                y_arr = np.asarray(targets)

            if is_classification:
                self.targets = torch.tensor(y_arr, dtype=torch.long)
            else:
                self.targets = torch.tensor(y_arr, dtype=torch.float32).unsqueeze(1)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = torch.from_numpy(self.features[idx])
        if self.targets is not None:
            return x, self.targets[idx]
        return x


class MultimodalDataset(Dataset):
    """PyTorch Dataset supporting Text token encodings, Image paths, and Tabular attributes."""

    def __init__(
        self,
        df: pd.DataFrame,
        text_cols: Optional[List[str]] = None,
        image_col: Optional[str] = None,
        tabular_cols: Optional[List[str]] = None,
        target_col: Optional[str] = None,
        tokenizer: Optional[any] = None,
        image_transform: Optional[any] = None,
        max_length: int = 128,
        is_classification: bool = True,
    ) -> None:
        """Initialize MultimodalDataset.

        Args:
            df: Pandas DataFrame containing dataset rows.
            text_cols: Names of columns containing text (concatenated with separator).
            image_col: Column name containing image filepaths.
            tabular_cols: Names of numeric or categorical feature columns.
            target_col: Target label column name if training set.
            tokenizer: HuggingFace Tokenizer instance.
            image_transform: Torchvision/Albumentations transform for images.
            max_length: Maximum token sequence length for tokenizer.
            is_classification: True for classification, False for regression.
        """
        self.df = df.reset_index(drop=True)
        self.text_cols = text_cols or []
        self.image_col = image_col
        self.tabular_cols = tabular_cols or []
        self.target_col = target_col
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.is_classification = is_classification

        # Precompute concatenated text strings
        if self.text_cols:
            self.texts = (
                self.df[self.text_cols]
                .fillna("")
                .astype(str)
                .agg(" | ".join, axis=1)
                .tolist()
            )
        else:
            self.texts = None

        # Precompute tabular matrix
        if self.tabular_cols:
            self.tabular_features = self.df[self.tabular_cols].fillna(0.0).values.astype(np.float32)
        else:
            self.tabular_features = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {}

        # 1. Text tokenization
        if self.texts is not None and self.tokenizer is not None:
            encoded = self.tokenizer(
                self.texts[idx],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            item["input_ids"] = encoded["input_ids"].squeeze(0)
            item["attention_mask"] = encoded["attention_mask"].squeeze(0)

        # 2. Image loading and transforms
        if self.image_col and self.image_col in self.df.columns:
            img_path = self.df.at[idx, self.image_col]
            try:
                image = Image.open(img_path).convert("RGB")
                if self.image_transform:
                    image = self.image_transform(image)
                item["image"] = image
            except Exception:
                # Return blank placeholder image if corrupt or not found
                item["image"] = torch.zeros((3, 224, 224), dtype=torch.float32)

        # 3. Tabular features
        if self.tabular_features is not None:
            item["tabular"] = torch.from_numpy(self.tabular_features[idx])

        # 4. Target label
        if self.target_col and self.target_col in self.df.columns:
            y_val = self.df.at[idx, self.target_col]
            if self.is_classification:
                item["target"] = torch.tensor(int(y_val), dtype=torch.long)
            else:
                item["target"] = torch.tensor(float(y_val), dtype=torch.float32)

        return item
