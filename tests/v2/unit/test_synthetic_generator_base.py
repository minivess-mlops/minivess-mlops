"""Tests for SyntheticGeneratorAdapter ABC and registry.

TDD RED phase for Task T-D1 (Issue #768).
"""

from __future__ import annotations

import numpy as np
import pytest


class TestSyntheticGeneratorAdapterABC:
    """Test the abstract base class contract."""

    def test_abc_cannot_be_instantiated(self) -> None:
        """SyntheticGeneratorAdapter is abstract — cannot instantiate directly."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        with pytest.raises(TypeError, match="abstract"):
            SyntheticGeneratorAdapter()  # type: ignore[abstract]

    def test_abc_requires_generate_stack(self) -> None:
        """Subclasses must implement generate_stack."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class BadGenerator(SyntheticGeneratorAdapter):
            @property
            def name(self) -> str:
                return "bad"

            @property
            def requires_training(self) -> bool:
                return False

        with pytest.raises(TypeError, match="abstract"):
            BadGenerator()  # type: ignore[abstract]

    def test_abc_requires_name(self) -> None:
        """Subclasses must implement name property."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class BadGenerator(SyntheticGeneratorAdapter):
            def generate_stack(
                self, n_volumes: int, config: dict | None = None
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                return []

            @property
            def requires_training(self) -> bool:
                return False

        with pytest.raises(TypeError, match="abstract"):
            BadGenerator()  # type: ignore[abstract]

    def test_abc_requires_requires_training(self) -> None:
        """Subclasses must implement requires_training property."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class BadGenerator(SyntheticGeneratorAdapter):
            def generate_stack(
                self, n_volumes: int, config: dict | None = None
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                return []

            @property
            def name(self) -> str:
                return "bad"

        with pytest.raises(TypeError, match="abstract"):
            BadGenerator()  # type: ignore[abstract]

    def test_concrete_subclass_instantiable(self) -> None:
        """A properly implemented subclass can be instantiated."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class GoodGenerator(SyntheticGeneratorAdapter):
            def generate_stack(
                self, n_volumes: int, config: dict | None = None
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                return [
                    (np.zeros((32, 32, 32)), np.zeros((32, 32, 32)))
                    for _ in range(n_volumes)
                ]

            @property
            def name(self) -> str:
                return "good_test"

            @property
            def requires_training(self) -> bool:
                return False

        gen = GoodGenerator()
        assert gen.name == "good_test"
        assert gen.requires_training is False

    def test_generate_stack_returns_list_of_tuples(self) -> None:
        """generate_stack must return list of (image, mask) tuples."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class SimpleGenerator(SyntheticGeneratorAdapter):
            def generate_stack(
                self, n_volumes: int, config: dict | None = None
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                return [
                    (
                        np.random.rand(32, 32, 32).astype(np.float32),
                        np.zeros((32, 32, 32), dtype=np.uint8),
                    )
                    for _ in range(n_volumes)
                ]

            @property
            def name(self) -> str:
                return "simple"

            @property
            def requires_training(self) -> bool:
                return False

        gen = SimpleGenerator()
        result = gen.generate_stack(n_volumes=3)
        assert len(result) == 3
        for image, mask in result:
            assert isinstance(image, np.ndarray)
            assert isinstance(mask, np.ndarray)
            assert image.ndim == 3, "Image must be 3D"
            assert mask.ndim == 3, "Mask must be 3D"

    def test_generate_stack_respects_n_volumes(self) -> None:
        """generate_stack returns exactly n_volumes pairs."""
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class CountGenerator(SyntheticGeneratorAdapter):
            def generate_stack(
                self, n_volumes: int, config: dict | None = None
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                return [
                    (np.zeros((16, 16, 16)), np.zeros((16, 16, 16)))
                    for _ in range(n_volumes)
                ]

            @property
            def name(self) -> str:
                return "count"

            @property
            def requires_training(self) -> bool:
                return False

        gen = CountGenerator()
        for n in [1, 5, 10]:
            result = gen.generate_stack(n_volumes=n)
            assert len(result) == n


class TestSyntheticGeneratorRegistry:
    """Test the registry pattern for synthetic generators."""

    def test_registry_exists(self) -> None:
        """SYNTHETIC_GENERATORS registry dict exists."""
        from minivess.data.synthetic import SYNTHETIC_GENERATORS

        assert isinstance(SYNTHETIC_GENERATORS, dict)

    def test_get_generator_by_name(self) -> None:
        """get_generator() retrieves a generator by method name."""
        from minivess.data.synthetic import get_generator

        # Should not raise for a registered method
        # At minimum, 'debug' (from existing debug_dataset.py logic) should exist
        gen = get_generator("debug")
        assert gen is not None

    def test_get_generator_unknown_raises(self) -> None:
        """get_generator() raises KeyError for unknown method."""
        from minivess.data.synthetic import get_generator

        with pytest.raises(KeyError, match="nonexistent_method"):
            get_generator("nonexistent_method")

    def test_list_generators(self) -> None:
        """list_generators() returns available method names."""
        from minivess.data.synthetic import list_generators

        names = list_generators()
        assert isinstance(names, list)
        assert len(names) >= 1
        assert all(isinstance(n, str) for n in names)

    def test_generate_stack_top_level_api(self) -> None:
        """generate_stack() top-level function dispatches to registry."""
        from minivess.data.synthetic import generate_stack

        result = generate_stack(method="debug", n_volumes=2)
        assert len(result) == 2
        for image, mask in result:
            assert isinstance(image, np.ndarray)
            assert isinstance(mask, np.ndarray)
            assert image.ndim == 3
            assert mask.ndim == 3

    def test_register_generator(self) -> None:
        """register_generator() adds a new generator to the registry."""
        from minivess.data.synthetic import register_generator
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        class CustomGen(SyntheticGeneratorAdapter):
            def generate_stack(
                self, n_volumes: int, config: dict | None = None
            ) -> list[tuple[np.ndarray, np.ndarray]]:
                return [
                    (np.ones((8, 8, 8)), np.ones((8, 8, 8))) for _ in range(n_volumes)
                ]

            @property
            def name(self) -> str:
                return "custom_test"

            @property
            def requires_training(self) -> bool:
                return False

        register_generator("custom_test", CustomGen)
        from minivess.data.synthetic import get_generator

        gen = get_generator("custom_test")
        assert gen is not None

    def test_registry_values_are_generator_types(self) -> None:
        """All registry values are SyntheticGeneratorAdapter subclasses."""
        from minivess.data.synthetic import SYNTHETIC_GENERATORS
        from minivess.data.synthetic.base import SyntheticGeneratorAdapter

        for method_name, gen_cls in SYNTHETIC_GENERATORS.items():
            assert issubclass(gen_cls, SyntheticGeneratorAdapter), (
                f"{method_name} is not a SyntheticGeneratorAdapter subclass"
            )
