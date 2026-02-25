"""Tests for reproducibility hardening (R5.19 + R5.20).

Validates CUDA determinism configuration and DataLoader worker seeding.
"""

from __future__ import annotations

from unittest.mock import patch

import torch

from minivess.utils.seed import set_global_seed

# ---------------------------------------------------------------------------
# T1: set_global_seed configures CUDA determinism
# ---------------------------------------------------------------------------


class TestCudaDeterminism:
    """Test that set_global_seed enables CUDA deterministic mode."""

    def test_cudnn_deterministic_enabled(self) -> None:
        """set_global_seed should set cudnn.deterministic = True."""
        set_global_seed(42)
        assert torch.backends.cudnn.deterministic is True

    def test_cudnn_benchmark_disabled(self) -> None:
        """set_global_seed should set cudnn.benchmark = False."""
        set_global_seed(42)
        assert torch.backends.cudnn.benchmark is False

    def test_seed_return_value(self) -> None:
        """set_global_seed should return the seed."""
        assert set_global_seed(123) == 123

    def test_different_seeds_produce_different_tensors(self) -> None:
        """Different seeds should produce different random tensors."""
        set_global_seed(1)
        t1 = torch.randn(10)
        set_global_seed(2)
        t2 = torch.randn(10)
        assert not torch.allclose(t1, t2)

    def test_same_seed_reproduces_tensors(self) -> None:
        """Same seed should produce identical random tensors."""
        set_global_seed(42)
        t1 = torch.randn(10)
        set_global_seed(42)
        t2 = torch.randn(10)
        assert torch.allclose(t1, t2)


# ---------------------------------------------------------------------------
# T2: DataLoader worker_init_fn
# ---------------------------------------------------------------------------


class TestDataLoaderWorkerSeeding:
    """Test that DataLoader worker_init_fn is provided for reproducibility."""

    def test_worker_init_fn_exists(self) -> None:
        """data/loader.py should define a _worker_init_fn."""
        from minivess.data.loader import _worker_init_fn

        assert callable(_worker_init_fn)

    def test_worker_init_fn_accepts_worker_id(self) -> None:
        """_worker_init_fn should accept a worker_id argument."""
        from minivess.data.loader import _worker_init_fn

        # Should not raise
        _worker_init_fn(0)
        _worker_init_fn(3)

    def test_train_loader_uses_worker_init_fn(self) -> None:
        """build_train_loader should pass worker_init_fn to DataLoader."""
        from minivess.data import loader

        with patch.object(loader, "CacheDataset"), \
             patch.object(loader, "DataLoader") as mock_dl, \
             patch.object(loader, "build_train_transforms"):
            from minivess.config.models import DataConfig

            config = DataConfig(dataset_name="test", num_workers=2)
            loader.build_train_loader([{"image": "a", "label": "b"}], config)

            # DataLoader should be called with worker_init_fn kwarg
            dl_kwargs = mock_dl.call_args
            assert "worker_init_fn" in dl_kwargs.kwargs or \
                   (len(dl_kwargs.args) > 5) or \
                   any("worker_init_fn" in str(k) for k in dl_kwargs.kwargs)
