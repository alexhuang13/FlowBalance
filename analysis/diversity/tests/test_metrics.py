from analysis.diversity.metrics import diversity_metrics


def test_balanced_two_cluster_metrics():
    result = diversity_metrics(["a", "a", "b", "b"])
    assert result["n"] == 4
    assert result["k"] == 2
    assert result["dominant_ratio"] == 0.5
    assert result["simpson"] == 0.5


def test_none_labels_are_excluded():
    result = diversity_metrics([1, None, 1, 2])
    assert result["n"] == 3
    assert result["cluster_sizes"] == {1: 2, 2: 1}


def test_empty_metrics():
    result = diversity_metrics([None, None])
    assert result["n"] == 0
    assert result["simpson"] is None
