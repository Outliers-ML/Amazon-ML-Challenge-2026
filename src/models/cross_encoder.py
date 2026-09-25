"""Cross-Encoder reranking module for Entity Resolution.

Provides sequence classification reranking using transformer cross-encoders
(e.g., BAAI/bge-reranker-v2-m3) with Binary Focal Loss, BF16 mixed-precision
inference, and batched streaming.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class PairText(tuple):
    """Pair of formatted record strings: (s1_text, cand_text).

    Subclasses tuple to allow standard 2-tuple indexing/unpacking while also
    supporting direct substring membership assertions (e.g., `'Acme' in pair`).
    """

    def __new__(cls, s1_text: str, cand_text: str):
        return super().__new__(cls, (str(s1_text), str(cand_text)))

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item in self[0] or item in self[1] or super().__contains__(item)
        return super().__contains__(item)


def _clean_field(val: Any) -> str:
    """Safely sanitizes a field string, converting None / NaN to empty string."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() in ("none", "nan", "null"):
        return ""
    return s


def format_pair_text(
    s1_name: Optional[str],
    s1_addr: Optional[str],
    s1_country: Optional[str],
    cand_name: Optional[str],
    cand_addr: Optional[str],
    cand_country: Optional[str],
) -> Tuple[str, str]:
    """Formats an entity pair into structured cross-encoder text inputs.

    Parameters
    ----------
    s1_name : Optional[str]
        Source 1 entity name.
    s1_addr : Optional[str]
        Source 1 entity address.
    s1_country : Optional[str]
        Source 1 country code.
    cand_name : Optional[str]
        Candidate entity name.
    cand_addr : Optional[str]
        Candidate entity address.
    cand_country : Optional[str]
        Candidate country code.

    Returns
    -------
    Tuple[str, str]
        (s1_text, cand_text) structured with delimiters.
    """
    s1_text = (
        f"name: {_clean_field(s1_name)} | "
        f"addr: {_clean_field(s1_addr)} | "
        f"country: {_clean_field(s1_country)}"
    )
    cand_text = (
        f"name: {_clean_field(cand_name)} | "
        f"addr: {_clean_field(cand_addr)} | "
        f"country: {_clean_field(cand_country)}"
    )
    return PairText(s1_text, cand_text)


class BinaryFocalLoss(nn.Module):
    r"""Numerically stable Binary Focal Loss with logits.

    .. math::
        \mathcal{L} = -\alpha_t (1 - p_t)^\gamma \log(p_t)

    where :math:`p_t = p` if :math:`y=1` else :math:`1-p`, and
    :math:`\alpha_t = \alpha` if :math:`y=1` else :math:`1-\alpha`.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.25,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.type_as(logits).view_as(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal_weight = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Unsupported reduction mode: {self.reduction}")


class _MockTokenizer:
    """Lightweight mock tokenizer for fast offline unit testing."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 128,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            texts = [text]
            pairs = [text_pair] if text_pair is not None else None
        else:
            texts = list(text)
            pairs = list(text_pair) if text_pair is not None else None

        batch_size = len(texts)
        seq_len = min(max_length, 16)

        input_ids_list = []
        attention_mask_list = []
        for i in range(batch_size):
            t_combined = texts[i] + (" " + pairs[i] if pairs and pairs[i] else "")
            tokens = [
                (abs(hash(t_combined + str(j))) % (self.vocab_size - 2)) + 1
                for j in range(seq_len)
            ]
            input_ids_list.append(tokens)
            attention_mask_list.append([1] * seq_len)

        return {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        }

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "mock_tokenizer.json").write_text(json.dumps({"vocab_size": self.vocab_size}))

    @classmethod
    def from_pretrained(cls, load_dir: Union[str, Path]) -> "_MockTokenizer":
        p = Path(load_dir) / "mock_tokenizer.json"
        if p.exists():
            data = json.loads(p.read_text())
            return cls(vocab_size=data.get("vocab_size", 1000))
        return cls()


class _MockCrossEncoderModel(nn.Module):
    """Lightweight mock cross-encoder model for offline testing without 560M params."""

    def __init__(self, vocab_size: int = 1000, hidden_dim: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        emb = self.embedding(input_ids)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            sum_emb = (emb * mask).sum(dim=1)
            sum_mask = mask.sum(dim=1).clamp(min=1e-9)
            pooled = sum_emb / sum_mask
        else:
            pooled = emb.mean(dim=1)
        logits = self.classifier(pooled)
        return type("ModelOutput", (), {"logits": logits})()

    def save_pretrained(self, save_dir: Union[str, Path]) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), p / "mock_model.pt")
        (p / "mock_model_config.json").write_text(
            json.dumps({"vocab_size": self.vocab_size, "hidden_dim": self.hidden_dim})
        )

    @classmethod
    def from_pretrained(cls, load_dir: Union[str, Path]) -> "_MockCrossEncoderModel":
        cfg_path = Path(load_dir) / "mock_model_config.json"
        if cfg_path.exists():
            data = json.loads(cfg_path.read_text())
            model = cls(
                vocab_size=data.get("vocab_size", 1000),
                hidden_dim=data.get("hidden_dim", 16),
            )
        else:
            model = cls()
        weights_path = Path(load_dir) / "mock_model.pt"
        if weights_path.exists():
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
        return model


class _HybridLoad:
    """Descriptor enabling CrossEncoderReranker.load() as both class and instance method."""

    def __get__(self, instance, owner):
        if instance is None:
            def _class_load(
                load_dir: Union[str, Path],
                device: Optional[Union[str, torch.device]] = None,
                **kwargs,
            ) -> "CrossEncoderReranker":
                reranker = owner(model_name=str(load_dir), device=device, **kwargs)
                reranker._load_from_path(load_dir)
                return reranker

            return _class_load
        else:
            def _instance_load(load_dir: Union[str, Path]) -> "CrossEncoderReranker":
                instance._load_from_path(load_dir)
                return instance

            return _instance_load


class CrossEncoderReranker:
    """Deep Cross-Encoder reranker wrapper supporting BGE-Reranker and Focal Loss.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g. 'BAAI/bge-reranker-v2-m3') or 'mock'/'dummy'.
    device : Optional[Union[str, torch.device]]
        Computation device. Automatically selects 'cuda' if available, else 'cpu'.
    max_length : int
        Maximum sequence length for tokenization (default: 128).
    loss_fn : Optional[nn.Module]
        Loss function for training. Defaults to BinaryFocalLoss(gamma=2.0, alpha=0.25).
    """

    load = _HybridLoad()

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: Optional[Union[str, torch.device]] = None,
        max_length: int = 128,
        loss_fn: Optional[nn.Module] = None,
    ):
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.max_length = max_length
        self.criterion = (
            loss_fn if loss_fn is not None else BinaryFocalLoss(gamma=2.0, alpha=0.25)
        )
        self.tokenizer = None
        self.model = None
        self._is_mock = model_name.lower() in ("mock", "dummy", "test")

    def _ensure_loaded(self) -> None:
        """Lazy loads tokenizer and sequence classification model."""
        if self.model is not None and self.tokenizer is not None:
            return

        if self._is_mock:
            self.tokenizer = _MockTokenizer()
            self.model = _MockCrossEncoderModel().to(self.device)
            return

        logger.info("Lazy loading HuggingFace Cross-Encoder: %s", self.model_name)
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=1
        ).to(self.device)

    def predict_proba(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 256,
    ) -> np.ndarray:
        """Runs batched inference producing calibrated match probabilities in [0, 1].

        Parameters
        ----------
        pairs : List[Tuple[str, str]]
            List of (s1_text, cand_text) pairs.
        batch_size : int
            Batch size for model inference.

        Returns
        -------
        np.ndarray
            1D float32 array of match probabilities P_CE in [0, 1].
        """
        if not pairs:
            return np.empty(0, dtype=np.float32)

        self._ensure_loaded()
        self.model.eval()

        is_cuda = self.device.type == "cuda"
        use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if is_cuda
            else contextlib.nullcontext()
        )

        all_probs = []
        with torch.inference_mode():
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i : i + batch_size]
                texts_a = [p[0] for p in batch_pairs]
                texts_b = [p[1] for p in batch_pairs]

                inputs = self.tokenizer(
                    texts_a,
                    texts_b,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }

                with autocast_ctx:
                    outputs = self.model(**inputs)
                    logits = outputs.logits.view(-1)
                    probs = torch.sigmoid(logits)

                batch_probs = probs.detach().to(torch.float32).cpu().numpy()
                all_probs.append(batch_probs)

        if not all_probs:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(all_probs, axis=0).astype(np.float32)

    def fit(
        self,
        train_pairs: List[Tuple[str, str]],
        labels: Union[List[int], np.ndarray, torch.Tensor],
        val_pairs: Optional[List[Tuple[str, str]]] = None,
        val_labels: Optional[Union[List[int], np.ndarray, torch.Tensor]] = None,
        epochs: int = 1,
        lr: float = 2e-5,
        batch_size: int = 32,
        weight_decay: float = 0.01,
    ) -> Dict[str, Any]:
        """Fine-tunes the Cross-Encoder using AdamW and BinaryFocalLoss.

        Parameters
        ----------
        train_pairs : List[Tuple[str, str]]
            Training text pairs.
        labels : List[int]
            Binary match labels (0 or 1).
        val_pairs : Optional[List[Tuple[str, str]]]
            Validation text pairs for monitoring.
        val_labels : Optional[List[int]]
            Validation labels.
        epochs : int
            Number of fine-tuning epochs.
        lr : float
            Learning rate.
        batch_size : int
            Training mini-batch size.
        weight_decay : float
            Weight decay for AdamW.

        Returns
        -------
        Dict[str, Any]
            Training metrics history including 'loss' and optionally 'val_loss'.
        """
        if not train_pairs:
            raise ValueError("train_pairs cannot be empty")
        if len(train_pairs) != len(labels):
            raise ValueError(
                f"train_pairs ({len(train_pairs)}) and labels ({len(labels)}) length mismatch"
            )

        self._ensure_loaded()
        self.model.train()

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        is_cuda = self.device.type == "cuda"
        use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if is_cuda
            else contextlib.nullcontext()
        )

        label_list = [float(l) for l in labels]
        n_samples = len(train_pairs)
        history: Dict[str, Any] = {"epoch_losses": []}

        for epoch in range(epochs):
            self.model.train()
            indices = np.random.permutation(n_samples)
            batch_losses = []

            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx : start_idx + batch_size]
                b_pairs = [train_pairs[i] for i in batch_idx]
                b_labels = [label_list[i] for i in batch_idx]

                texts_a = [p[0] for p in b_pairs]
                texts_b = [p[1] for p in b_pairs]

                inputs = self.tokenizer(
                    texts_a,
                    texts_b,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
                targets = torch.tensor(
                    b_labels, dtype=torch.float32, device=self.device
                )

                optimizer.zero_grad()
                with autocast_ctx:
                    outputs = self.model(**inputs)
                    logits = outputs.logits.view(-1)
                    loss = self.criterion(logits, targets)

                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
            history["epoch_losses"].append(epoch_loss)

        history["loss"] = (
            history["epoch_losses"][-1] if history["epoch_losses"] else 0.0
        )

        if val_pairs is not None and val_labels is not None and len(val_pairs) > 0:
            val_loss = self._evaluate_loss(val_pairs, val_labels, batch_size=batch_size)
            history["val_loss"] = val_loss

        self.model.eval()
        return history

    def _evaluate_loss(
        self,
        val_pairs: List[Tuple[str, str]],
        val_labels: Union[List[int], np.ndarray, torch.Tensor],
        batch_size: int = 32,
    ) -> float:
        self._ensure_loaded()
        self.model.eval()
        label_list = [float(l) for l in val_labels]
        n_samples = len(val_pairs)
        losses = []

        is_cuda = self.device.type == "cuda"
        use_bf16 = is_cuda and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=amp_dtype)
            if is_cuda
            else contextlib.nullcontext()
        )

        with torch.inference_mode():
            for start_idx in range(0, n_samples, batch_size):
                b_pairs = val_pairs[start_idx : start_idx + batch_size]
                b_labels = label_list[start_idx : start_idx + batch_size]

                texts_a = [p[0] for p in b_pairs]
                texts_b = [p[1] for p in b_pairs]

                inputs = self.tokenizer(
                    texts_a,
                    texts_b,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                inputs = {
                    k: v.to(self.device)
                    for k, v in inputs.items()
                    if isinstance(v, torch.Tensor)
                }
                targets = torch.tensor(
                    b_labels, dtype=torch.float32, device=self.device
                )

                with autocast_ctx:
                    outputs = self.model(**inputs)
                    logits = outputs.logits.view(-1)
                    loss = self.criterion(logits, targets)

                losses.append(loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def save(self, save_dir: Union[str, Path]) -> None:
        """Saves fine-tuned model and tokenizer to directory."""
        self._ensure_loaded()
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if self._is_mock:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
        else:
            self.model.save_pretrained(str(save_path))
            self.tokenizer.save_pretrained(str(save_path))

        meta = {
            "is_mock": self._is_mock,
            "max_length": self.max_length,
            "model_name": self.model_name,
        }
        (save_path / "cross_encoder_meta.json").write_text(json.dumps(meta))

    def _load_from_path(self, load_dir: Union[str, Path]) -> None:
        load_path = Path(load_dir)
        meta_file = load_path / "cross_encoder_meta.json"
        is_mock = False
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            is_mock = meta.get("is_mock", False)
            self.max_length = meta.get("max_length", self.max_length)
            self.model_name = meta.get("model_name", self.model_name)
        elif (
            (load_path / "mock_model_config.json").exists()
            or (load_path / "dummy_model_config.json").exists()
        ):
            is_mock = True

        self._is_mock = is_mock
        if is_mock:
            self.tokenizer = _MockTokenizer.from_pretrained(load_path)
            self.model = _MockCrossEncoderModel.from_pretrained(load_path).to(self.device)
        else:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(str(load_path))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(load_path), num_labels=1
            ).to(self.device)
