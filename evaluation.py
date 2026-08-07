import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import time

from sklearn.metrics import classification_report
from constants import LABEL2ID, ID2LABEL


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tags_to_spans(tags):
    """
    Converts IOB tags into spans.
    For evaluation on span-level, not token-level.

    Example:
    ["B-LOC", "I-LOC", "I-LOC", "O", "B-LOC", "I-LOC"]
    ->
    {(0, 3, "LOC"), (4, 6, "LOC")}
    """

    spans = set()
    start = None
    entity_type = None

    for i, tag in enumerate(tags):

        if tag == "O":
            if start is not None:
                spans.add((start, i, entity_type))
                start = None
                entity_type = None

        elif tag.startswith("B-"):
            # Close previous span
            if start is not None:
                spans.add((start, i, entity_type))

            start = i
            entity_type = tag[2:]

        elif tag.startswith("I-"):
            # Continue current span
            if start is None:
                # Robustness against malformed IOB sequence
                start = i
                entity_type = tag[2:]

    # Close span at end of sequence
    if start is not None:
        spans.add((start, len(tags), entity_type))

    return spans


# Evaluation
def test_model(model, test_data):
    model.eval()

    all_predictions = []
    all_true_tags = []

    all_pred_spans = []
    all_true_spans = []

    with torch.no_grad():

        for batch in test_data:
            embeddings = batch["embeddings"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            predictions = model.decode(embeddings, mask)

            for pred, gold, valid in zip(predictions, tags, mask):

                length = valid.sum().item()

                # Remove padding
                pred = pred[:length]

                gold = gold[:length].cpu().tolist()

                # Convert prediction IDs to labels
                pred_labels = [ID2LABEL[tag_id] for tag_id in pred]

                # Convert gold IDs to labels
                gold_labels = [
                    ID2LABEL[tag_id] for tag_id in gold]

                # Token-level evaluation
                all_predictions.extend(pred_labels)
                all_true_tags.extend(gold_labels)

                # Span-level evaluation
                pred_spans = tags_to_spans(pred_labels)
                gold_spans = tags_to_spans(gold_labels)

                all_pred_spans.append(pred_spans)
                all_true_spans.append(gold_spans)

    return all_true_tags, all_predictions, all_true_spans, all_pred_spans

def span_metrics(all_true_spans, all_pred_spans):

    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold, pred in zip(all_true_spans, all_pred_spans):

        true_positive += len(gold & pred)
        false_positive += len(pred - gold)
        false_negative += len(gold - pred)

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0
        else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return precision, recall, f1


def save_test_report(true_tags, predictions, true_spans, pred_spans, hyperparameters):

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    filename = f"data/test_reports/{current_time}.csv"

    # Token-level metrics
    report = classification_report(true_tags, predictions, target_names=LABEL2ID.keys(), output_dict=True)
    df_report = pd.DataFrame(report).T

    # Span-level metrics
    span_precision, span_recall, span_f1 = span_metrics(true_spans, pred_spans)

    df_report.loc["span_precision", "precision"] = span_precision
    df_report.loc["span_recall", "recall"] = span_recall
    df_report.loc["span_f1", "f1-score"] = span_f1

    for key, value in hyperparameters.items():
        df_report[key] = value
    df_report = df_report.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    df_report.to_csv(filename)