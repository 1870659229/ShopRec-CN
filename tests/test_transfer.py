from src.models.transfer import fuse, uncertainty_weight


def test_uncertainty_weight_sums_to_one():
    w = uncertainty_weight([0.9, 0.5, 0.1])
    assert round(sum(w), 6) == 1.0


def test_uncertainty_weight_penalizes_low_confidence():
    w = uncertainty_weight([0.9, 0.1])
    assert w[0] > w[1]


def test_fuse_prefers_target_when_confidence_high():
    out = fuse(aux=[1.0, 0.0], tgt=[0.0, 1.0], confs=[0.99, 0.99])
    assert out[0] < out[1]


def test_fuse_prefers_aux_when_confidence_low():
    out = fuse(aux=[1.0, 0.0], tgt=[0.0, 1.0], confs=[0.01, 0.01])
    assert out[0] > out[1]


def test_uncertainty_weight_empty_returns_empty():
    assert uncertainty_weight([]) == []


def test_uncertainty_weight_all_zero_is_uniform():
    assert uncertainty_weight([0.0, 0.0]) == [0.5, 0.5]
