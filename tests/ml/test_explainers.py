from ml.explain.registry import (
    EXPLAINER_SPECS,
    explainer_mode,
    explainer_name,
    is_live_explainer,
)


def test_linear_and_tree_explainer_interface():
    assert explainer_name("linear_regression") == "LinearExplainer"
    assert explainer_name("random_forest") == "TreeExplainer"
    assert explainer_name("xgboost") == "TreeExplainer"
    assert is_live_explainer("linear_regression")
    assert is_live_explainer("random_forest")
    assert is_live_explainer("xgboost")


def test_sequence_models_are_offline_permutation_not_kernel():
    for name in ("lstm", "cnn_lstm"):
        assert explainer_name(name) == "PermutationExplainer"
        assert explainer_mode(name) == "offline"
        assert not is_live_explainer(name)
        spec = EXPLAINER_SPECS[name]
        assert "KernelSHAP" in spec["rationale"] or "KernelSHAP" in spec["rationale"]
        assert "not" in spec["rationale"].lower() or "too slow" in spec["rationale"].lower()


def test_no_model_uses_kernel_explainer_as_the_explainer():
    for name, spec in EXPLAINER_SPECS.items():
        assert spec["explainer"] != "KernelExplainer"
        assert "KernelExplainer" not in spec["explainer"]
