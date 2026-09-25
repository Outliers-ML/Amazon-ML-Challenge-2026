import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch


def test_pair_sequence_formatting():
    from src.models.cross_encoder import format_pair_text

    formatted = format_pair_text(
        s1_name="Acme Corp",
        s1_addr="10 Main St",
        s1_country="US",
        cand_name="Acme Corporation",
        cand_addr="10 Main Street",
        cand_country="US",
    )
    # Check structure
    assert isinstance(formatted, tuple)
    assert len(formatted) == 2
    assert formatted[0] == "name: Acme Corp | addr: 10 Main St | country: US"
    assert formatted[1] == "name: Acme Corporation | addr: 10 Main Street | country: US"
    # Plan assertion: substring checks directly on formatted pair
    assert "[CLS]" in formatted or "Acme Corp" in formatted
    assert "Acme Corporation" in formatted

    # Safe handling of empty / None / <NA> values
    empty_res = format_pair_text(
        s1_name=None,
        s1_addr=None,
        s1_country="US",
        cand_name="Acme",
        cand_addr="",
        cand_country=None,
    )
    assert empty_res[0] == "name:  | addr:  | country: US"
    assert empty_res[1] == "name: Acme | addr:  | country: "

    na_res = format_pair_text(
        s1_name="<NA>",
        s1_addr="<na>",
        s1_country="US",
        cand_name="Acme",
        cand_addr="<Na>",
        cand_country="<na>",
    )
    assert na_res[0] == "name:  | addr:  | country: US"
    assert na_res[1] == "name: Acme | addr:  | country: "


def test_focal_loss_computation():
    from src.models.cross_encoder import BinaryFocalLoss

    loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.25)
    logits = torch.tensor([2.5, -2.5], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([1.0, 0.0], dtype=torch.float32)

    loss = loss_fn(logits, targets)
    assert loss.item() > 0.0

    # Test gradients exist and flow properly
    loss.backward()
    assert logits.grad is not None
    # For target=1 with positive logit, loss should decrease if logit increases (grad < 0)
    assert logits.grad[0].item() < 0.0
    # For target=0 with negative logit, loss should decrease if logit decreases (grad > 0)
    assert logits.grad[1].item() > 0.0

    # Numerical stability with extreme and infinite logits
    extreme_logits = torch.tensor([100.0, -100.0, 50.0, -50.0], dtype=torch.float32)
    extreme_targets = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    extreme_loss = loss_fn(extreme_logits, extreme_targets)
    assert torch.isfinite(extreme_loss)
    assert not torch.isnan(extreme_loss)
    assert extreme_loss.item() > 0.0

    inf_logits = torch.tensor([float("inf"), float("-inf"), 1e9, -1e9], dtype=torch.float32)
    inf_targets = torch.tensor([1.0, 0.0, 0.0, 1.0], dtype=torch.float32)
    inf_loss = loss_fn(inf_logits, inf_targets)
    assert torch.isfinite(inf_loss)
    assert not torch.isnan(inf_loss)
    assert inf_loss.item() > 0.0

    # Reduction options
    none_loss = BinaryFocalLoss(gamma=2.0, alpha=0.25, reduction="none")(logits.detach(), targets)
    assert none_loss.shape == (2,)
    sum_loss = BinaryFocalLoss(gamma=2.0, alpha=0.25, reduction="sum")(logits.detach(), targets)
    assert pytest.approx(sum_loss.item()) == none_loss.sum().item()


def test_cross_encoder_predict_and_fit_lightweight():
    from src.models.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="mock", device="cpu", max_length=128)

    pairs = [
        ("name: Acme Corp | addr: 10 Main St | country: US", "name: Acme Corporation | addr: 10 Main Street | country: US"),
        ("name: Tata Motors | addr: Mumbai | country: IN", "name: Google Inc | addr: Mountain View | country: US"),
    ]

    # Predict proba
    probs = reranker.predict_proba(pairs, batch_size=2)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (2,)
    assert probs.dtype == np.float32
    assert np.all((probs >= 0.0) & (probs <= 1.0))

    # Empty pairs handling
    empty_probs = reranker.predict_proba([])
    assert isinstance(empty_probs, np.ndarray)
    assert len(empty_probs) == 0

    # Fit lightweight model
    train_pairs = pairs * 8
    labels = [1, 0] * 8

    # Validation errors when mismatched or one is None
    with pytest.raises(ValueError, match="val_pairs and val_labels must both be provided and have matching lengths"):
        reranker.fit(train_pairs, labels, val_pairs=pairs, val_labels=[1])
    with pytest.raises(ValueError, match="val_pairs and val_labels must both be provided and have matching lengths"):
        reranker.fit(train_pairs, labels, val_pairs=pairs, val_labels=None)
    with pytest.raises(ValueError, match="val_pairs and val_labels must both be provided and have matching lengths"):
        reranker.fit(train_pairs, labels, val_pairs=None, val_labels=[1, 0])

    history = reranker.fit(
        train_pairs,
        labels,
        val_pairs=pairs,
        val_labels=[1, 0],
        epochs=2,
        lr=1e-3,
        batch_size=4,
    )
    assert history is not None
    assert "loss" in history
    assert "val_loss" in history

    # Inference after fit
    probs_after = reranker.predict_proba(pairs)
    assert probs_after.shape == (2,)
    assert np.all((probs_after >= 0.0) & (probs_after <= 1.0))


def test_cross_encoder_save_and_load():
    from src.models.cross_encoder import CrossEncoderReranker

    pairs = [
        ("name: Acme Corp | addr: 10 Main St | country: US", "name: Acme Corporation | addr: 10 Main Street | country: US"),
        ("name: Tata Motors | addr: Mumbai | country: IN", "name: Google Inc | addr: Mountain View | country: US"),
    ]

    reranker = CrossEncoderReranker(model_name="mock", device="cpu", max_length=128)
    probs_orig = reranker.predict_proba(pairs)

    with tempfile.TemporaryDirectory() as tmp_dir:
        reranker.save(tmp_dir)

        # Verify saved files exist
        assert len(list(Path(tmp_dir).iterdir())) > 0

        # Classmethod load
        loaded_reranker = CrossEncoderReranker.load(tmp_dir, device="cpu")
        probs_loaded = loaded_reranker.predict_proba(pairs)
        np.testing.assert_allclose(probs_orig, probs_loaded, rtol=1e-5, atol=1e-5)

        # Instance load with explicit device
        fresh_reranker = CrossEncoderReranker(model_name="mock", device="cpu")
        fresh_reranker.load(tmp_dir, device="cpu")
        probs_instance_loaded = fresh_reranker.predict_proba(pairs)
        np.testing.assert_allclose(probs_orig, probs_instance_loaded, rtol=1e-5, atol=1e-5)



@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cross_encoder_cuda_autocast():
    from src.models.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name="mock", device="cuda", max_length=128)
    pairs = [
        ("name: Acme Corp | addr: 10 Main St | country: US", "name: Acme Corporation | addr: 10 Main Street | country: US"),
    ]
    probs = reranker.predict_proba(pairs, batch_size=1)
    assert isinstance(probs, np.ndarray)
    assert probs.shape == (1,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
