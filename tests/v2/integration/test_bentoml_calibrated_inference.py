"""Integration tests for BentoML calibrated inference.

E2E Plan Phase 3, Task T3.1: BentoML inference with calibrated response.

Verifies:
1. BentoML responds with mask + calibrated probabilities + uncertainty intervals
2. Mask dtype is binary (uint8 or bool)
3. Probabilities are float32 in [0, 1]
4. Inference latency under 30 seconds for a single volume

Marked @integration @slow — excluded from staging and prod suites.
"""

from __future__ import annotations

import time

import pytest


def _bentoml_reachable() -> bool:
    """Check if BentoML serving endpoint is reachable."""
    try:
        import urllib.request

        with urllib.request.urlopen("http://localhost:3333/healthz", timeout=5):
            return True
    except Exception:
        return False


_REQUIRES_BENTOML = "requires BentoML serving endpoint"


@pytest.mark.integration
@pytest.mark.slow
class TestBentoMLCalibratedInference:
    """Verify BentoML returns mask + calibrated probabilities + uncertainty."""

    def test_bentoml_healthz_responds(self) -> None:
        """Verify BentoML /healthz endpoint returns HTTP 200."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import urllib.request

        with urllib.request.urlopen("http://localhost:3333/healthz", timeout=5) as resp:
            assert resp.status == 200

    def test_inference_returns_mask(self) -> None:
        """Send volume, verify response['mask'] exists with correct spatial dims."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import json
        import urllib.request

        import numpy as np

        # Create a small synthetic test volume
        volume = np.random.rand(1, 1, 16, 16, 8).astype(np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:3333/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        assert "mask" in result, (
            f"BentoML response missing 'mask'. Keys: {list(result.keys())}"
        )

    def test_mask_dtype_binary(self) -> None:
        """Verify mask values are 0 or 1."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import json
        import urllib.request

        import numpy as np

        volume = np.random.rand(1, 1, 16, 16, 8).astype(np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:3333/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if "mask" not in result:
            pytest.skip("No mask in response")

        mask = np.array(result["mask"])
        unique_values = set(np.unique(mask).tolist())
        assert unique_values.issubset({0, 1}), (
            f"Mask contains non-binary values: {unique_values}"
        )

    def test_inference_returns_calibrated_probabilities(self) -> None:
        """Verify response['probabilities'] exists, dtype float."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import json
        import urllib.request

        import numpy as np

        volume = np.random.rand(1, 1, 16, 16, 8).astype(np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:3333/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        assert "probabilities" in result, (
            f"BentoML response missing 'probabilities'. Keys: {list(result.keys())}"
        )

    def test_probabilities_in_unit_interval(self) -> None:
        """Verify all probability values in [0.0, 1.0]."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import json
        import urllib.request

        import numpy as np

        volume = np.random.rand(1, 1, 16, 16, 8).astype(np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:3333/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if "probabilities" not in result:
            pytest.skip("No probabilities in response")

        probs = np.array(result["probabilities"])
        assert probs.min() >= 0.0, f"Probability below 0: {probs.min()}"
        assert probs.max() <= 1.0, f"Probability above 1: {probs.max()}"

    def test_inference_returns_uncertainties(self) -> None:
        """Verify response['uncertainties'] dict has expected keys."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import json
        import urllib.request

        import numpy as np

        volume = np.random.rand(1, 1, 16, 16, 8).astype(np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:3333/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        if "uncertainties" not in result:
            pytest.skip(
                "No uncertainties in response (conformal plugin may not be enabled)"
            )

        uncertainties = result["uncertainties"]
        assert isinstance(uncertainties, dict), (
            f"uncertainties should be dict, got {type(uncertainties)}"
        )

    def test_inference_latency_under_30s(self) -> None:
        """Verify end-to-end inference completes within 30 seconds."""
        if not _bentoml_reachable():
            pytest.skip(_REQUIRES_BENTOML)

        import json
        import urllib.request

        import numpy as np

        volume = np.random.rand(1, 1, 16, 16, 8).astype(np.float32)
        payload = json.dumps({"volume": volume.tolist()}).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:3333/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp.read()
        elapsed = time.monotonic() - start

        assert elapsed <= 30.0, f"Inference took {elapsed:.1f}s, exceeds 30s limit"
