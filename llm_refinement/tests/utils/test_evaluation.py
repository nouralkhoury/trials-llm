import pytest
from utils.evaluation import evaluate_predictions, threshold_accuracy, save_eval, get_metrics


@pytest.mark.parametrize("predicted_output, ground_truth, entity, expected_tp, expected_tn, expected_fp, expected_fn", [
    (
        {"inclusion_biomarker": [["ALK translocation"], ["EGFR mutation"]], "exclusion_biomarker": []},
        {"inclusion_biomarker": [["ALK translocation"], ["EGFR mutation"]], "exclusion_biomarker": []},
        "inclusion_biomarker",
        2, 0, 0, 0
    ),
    (
        {"inclusion_biomarker": [["ALK translocation"], ["EGFR mutation"]], "exclusion_biomarker": []},
        {"inclusion_biomarker": [["ALK translocation"]], "exclusion_biomarker": []},
        "inclusion_biomarker",
        1, 0, 1, 0
    ),
    (
        {"inclusion_biomarker": [], "exclusion_biomarker": []},
        {"inclusion_biomarker": [], "exclusion_biomarker": []},
        "inclusion_biomarker",
        0, 1, 0, 0
    ),
    (
        {"inclusion_biomarker": [["ALK translocation"], ["EGFR mutation"]], "exclusion_biomarker": []},
        {"inclusion_biomarker": [["ALK translocation", "EGFR mutation"]], "exclusion_biomarker": []},
        "inclusion_biomarker",
        0, 0, 2, 1
    ),
])
def test_evaluate_predictions_with_DNF(predicted_output, ground_truth, entity, expected_tp, expected_tn, expected_fp, expected_fn):
    tp, tn, fp, fn = evaluate_predictions(predicted_output, ground_truth, entity)
    assert tp == expected_tp, "True positives should match."
    assert tn == expected_tn, "True negatives should match."
    assert fp == expected_fp, "False positives should match."
    assert fn == expected_fn, "False negatives should match."


@pytest.mark.parametrize("predicted_output, ground_truth, entity, expected_tp, expected_tn, expected_fp, expected_fn", [
    (
        {"inclusion_biomarker": ["ALK translocation", "EGFR mutation"], "exclusion_biomarker": []},
        {"inclusion_biomarker": ["ALK translocation", "EGFR mutation"], "exclusion_biomarker": []},
        "inclusion_biomarker",
        2, 0, 0, 0
    ),
    (
        {"inclusion_biomarker": ["ALK translocation", "EGFR mutation"], "exclusion_biomarker": []},
        {"inclusion_biomarker": ["ALK translocation"], "exclusion_biomarker": []},
        "inclusion_biomarker",
        1, 0, 1, 0
    ),
    (
        {"inclusion_biomarker": [], "exclusion_biomarker": []},
        {"inclusion_biomarker": [], "exclusion_biomarker": []},
        "inclusion_biomarker",
        0, 1, 0, 0
    ),
    (
        {"inclusion_biomarker": ["ALK translocation", "EGFR mutation"], "exclusion_biomarker": []},
        {"inclusion_biomarker": ["ALK translocation", "EGFR mutation"], "exclusion_biomarker": []},
        "inclusion_biomarker",
        2, 0, 0, 0
    ),
])
def test_evaluate_predictions_without_DNF(predicted_output, ground_truth, entity, expected_tp, expected_tn, expected_fp, expected_fn):
    tp, tn, fp, fn = evaluate_predictions(predicted_output, ground_truth, entity)
    assert tp == expected_tp, "True positives should match."
    assert tn == expected_tn, "True negatives should match."
    assert fp == expected_fp, "False positives should match."
    assert fn == expected_fn, "False negatives should match."

@pytest.mark.parametrize("tp, tn, fp, fn, expected_precision, expected_recall, expected_f1_score, expected_accuracy, expected_f2_score", [
    (1, 0, 0, 0, 1, 1, 1, 1, 1),
    (0, 0, 1, 0, 0, 0, 0, 0, 0),
])
def test_get_metrics(tp, tn, fp, fn, expected_precision, expected_recall, expected_f1_score, expected_accuracy, expected_f2_score):
    precision, recall, f1_score, accuracy, f2_score = get_metrics(tp, tn, fp, fn)
    assert precision == expected_precision, "Precision should match."
    assert recall == expected_recall, "Recall should match."
    assert f1_score == expected_f1_score, "F1 score should match."
    assert accuracy == expected_accuracy, "Accuracy should match."
    assert f2_score == expected_f2_score, "F2 score should match."

@pytest.mark.parametrize("accuracy, threshold, expected_result", [
    (0.85, 80, True),
    (0.79, 80, False),
])
def test_threshold_accuracy(accuracy, threshold, expected_result):
    result = threshold_accuracy(accuracy, threshold)
    assert result == expected_result, "Threshold accuracy result should match."

@pytest.mark.parametrize("tp, tn, fp, fn, evals, expected_tp, expected_tn, expected_fp, expected_fn", [
    (
        [1], [0], [0], [0],
        (1, 0, 0, 0),
        [1, 1], [0, 0], [0, 0], [0, 0]
    ),
    (
        [], [], [], [],
        (0, 1, 1, 0),
        [0], [1], [1], [0]
    ),
])
def test_save_eval(tp, tn, fp, fn, evals, expected_tp, expected_tn, expected_fp, expected_fn):
    save_eval(tp, tn, fp, fn, evals)
    assert tp == expected_tp, "True positives should match."
    assert tn == expected_tn, "True negatives should match."
    assert fp == expected_fp, "False positives should match."
    assert fn == expected_fn, "False negatives should match."

