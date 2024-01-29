"""
Clinical Trial Evaluation Script

This script includes functions for evaluating predictions in clinical trial
inclusion and exclusion criteria.
The `evaluate_predictions` function calculates precision, recall, F1 score,
and accuracy based on predicted and ground truth criteria.
It supports flexible input formats, allowing for both flat lists and lists of
lists representing conjunctions of criteria.
The `threshold_accuracy` function determines if the accuracy meets or exceeds
a specified threshold percentage.
Both functions assume accuracy is represented as a float between 0 and 1,
and the threshold is a percentage.
"""


def evaluate_predictions(predicted_output, ground_truth, entity):
    """
    Evaluate the predictions for inclusion/exclusion criteria in a clinical trial.

    Parameters:
    - predicted_output (dict): A dictionary containing the predicted criteria.
        Example: {"inclusion": ["A", "B", "C"]} or {"inclusion": [["A", "B"], ["C"]]}
    - ground_truth (dict): A dictionary containing the ground truth inclusion criteria.
        Example: {"inclusion": ["A", "B", "C"]} or {"inclusion": [["A", "B"], ["C"]]}

    Returns:
    - precision (float): Precision score calculated as the number of correctly predicted criteria divided by the total number of predicted criteria.
    - recall (float): Recall score calculated as the number of correctly predicted criteria divided by the total number of true criteria.
    - f1_score (float): F1 score calculated based on precision and recall.
    - accuracy (float): Accuracy score calculated as the number of correctly predicted criteria divided by the total number of predicted criteria.

    Note:
    - The function assumes that each predicted criterion should match exactly one true criterion.
    - The input format is flexible, supporting both flat lists and lists of lists for criteria.
      For flat lists, each element is treated as an individual criterion. For lists of lists, each inner list is treated as a conjunction of criteria.

    Precision, Recall, and F1 Score Formulas:
    - Precision = Correctly Predicted Criteria / Total Predicted Criteria
    - Recall = Correctly Predicted Criteria / Total True Criteria
    - F1 Score = 2 * (Precision * Recall) / (Precision + Recall), with consideration for cases where Precision + Recall is zero.
    """
    correct_conjunctions = 0
    predicted_conjunctions = len(predicted_output[entity])

    for predicted_conjunction in predicted_output[entity]:
        for true_conjunction in ground_truth[entity]:
            if set(predicted_conjunction) == set(true_conjunction):
                correct_conjunctions += 1
                break  # Break if a match is found, as each predicted conjunction should match exactly one true conjunction

    # Calculate precision, recall, F1 score, and accuracy
    precision = correct_conjunctions / predicted_conjunctions if predicted_conjunctions != 0 else 0
    recall = correct_conjunctions / len(ground_truth[entity]) if len(ground_truth[entity]) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    accuracy = correct_conjunctions / predicted_conjunctions if predicted_conjunctions != 0 else 0

    return precision, recall, f1_score, accuracy


def threshold_accuracy(accuracy, threshold=50):
    """
    Determine if the accuracy meets or exceeds a specified threshold percentage.

    Parameters:
    - accuracy (float): The accuracy score calculated as the ratio of correctly predicted criteria to the total number of predicted criteria.
    - threshold (float): The minimum percentage of accuracy required to return True. Default is 50.

    Returns:
    - threshold_met (bool): True if the accuracy percentage meets or exceeds the specified threshold, False otherwise.

    Note:
    - The function assumes that accuracy is provided as a float value between 0 and 1, where 1 represents 100% accuracy.
    - The threshold is specified as a percentage, and the comparison is done by multiplying the accuracy by 100 and checking if it meets or exceeds the threshold.
    """
    threshold_met = accuracy * 100 >= threshold
    return threshold_met
