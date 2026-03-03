"""Tests for MLflow param naming standardization (#276).

Covers:
- Param name validation against schema
- Required param checking
- Param prefix conventions
"""

from __future__ import annotations


class TestParamNameValidation:
    """Test param name validation against conventions."""

    def test_valid_param_names(self) -> None:
        from minivess.observability.mlflow_schema import validate_param_name

        assert validate_param_name("loss_name") is True
        assert validate_param_name("arch_filters") is True
        assert validate_param_name("sys_python_version") is True
        assert validate_param_name("data_n_volumes") is True
        assert validate_param_name("learning_rate") is True

    def test_invalid_param_names_with_slashes(self) -> None:
        from minivess.observability.mlflow_schema import validate_param_name

        # Slashes cause metric naming conflicts
        assert validate_param_name("sys/python_version") is False
        assert validate_param_name("loss/name") is False

    def test_invalid_param_names_with_dots(self) -> None:
        from minivess.observability.mlflow_schema import validate_param_name

        assert validate_param_name("model.family") is False


class TestRequiredParams:
    """Test required param checking."""

    def test_load_required_params(self) -> None:
        from minivess.observability.mlflow_schema import load_mlflow_schema

        schema = load_mlflow_schema()
        assert "required_params" in schema
        assert "loss_name" in schema["required_params"]
        assert "model_family" in schema["required_params"]

    def test_check_missing_params(self) -> None:
        from minivess.observability.mlflow_schema import check_required_params

        logged_params = {"loss_name": "dice_ce", "batch_size": "2"}
        missing = check_required_params(logged_params)
        # model_family should be missing
        assert "model_family" in missing

    def test_check_all_present(self) -> None:
        schema_module = __import__(
            "minivess.observability.mlflow_schema", fromlist=["load_mlflow_schema"]
        )
        schema = schema_module.load_mlflow_schema()
        # Provide all required params
        logged_params = {p: "value" for p in schema["required_params"]}
        missing = schema_module.check_required_params(logged_params)
        assert len(missing) == 0


class TestParamPrefixes:
    """Test param prefix conventions."""

    def test_known_prefixes(self) -> None:
        from minivess.observability.mlflow_schema import KNOWN_PARAM_PREFIXES

        assert "arch_" in KNOWN_PARAM_PREFIXES
        assert "sys_" in KNOWN_PARAM_PREFIXES
        assert "data_" in KNOWN_PARAM_PREFIXES
        assert "loss_" in KNOWN_PARAM_PREFIXES
        assert "eval_" in KNOWN_PARAM_PREFIXES
        assert "cfg_" in KNOWN_PARAM_PREFIXES

    def test_categorize_param(self) -> None:
        from minivess.observability.mlflow_schema import categorize_param

        assert categorize_param("arch_filters") == "architecture"
        assert categorize_param("sys_python_version") == "system"
        assert categorize_param("learning_rate") == "training"
        assert categorize_param("data_n_volumes") == "data"
