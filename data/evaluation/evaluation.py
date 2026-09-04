import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import os
import json

from sklearn.metrics import classification_report, precision_recall_fscore_support
from constants import LABEL2ID, ID2LABEL
from collections import defaultdict


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


# Transformer
def transformer_bio_predictions_to_spans(predicted_ids, offset_mapping, ):
    spans = []

    current_label = None
    current_start = None
    current_end = None

    for label_id, (token_start, token_end) in zip(
        predicted_ids,
        offset_mapping,
    ):
        # Special tokens
        if token_start == token_end:
            continue

        label = ID2LABEL[label_id]

        if label == "O":
            if current_label is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )
                current_label = None
                current_start = None
                current_end = None

            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            # close old span
            if current_label is not None:
                spans.append(
                    (current_start, current_end, current_label)
                )

            # start new span
            current_label = entity_type
            current_start = token_start
            current_end = token_end

        elif prefix == "I":
            if current_label == entity_type:
                # continue span
                current_end = token_end

            else:
                # unexpected I-Tag as new label
                if current_label is not None:
                    spans.append(
                        (current_start, current_end, current_label)
                    )

                current_label = entity_type
                current_start = token_start
                current_end = token_end

    # close last span
    if current_label is not None:
        spans.append(
            (current_start, current_end, current_label)
        )

    return spans


def pred_spans_to_fairytale(test_dataset, predicted_ids):
    """
    Takes BIO-predictions -> Character-Spans per Chunk.
    Takes all chunks of one fairytale.
    Only keeps single Prediction (because of set) if the same prediction is in overlapping chunks.

    :param test_dataset: RoBERTaSequenceTaggingDataset
    :param predicted_ids: output of trainer.predict, as labels (not probabilities)
    :return: predicted_spans_by_fairytale for all fairytales in testset
    """
    predicted_spans_by_fairytale = defaultdict(set)

    for i in range(len(test_dataset)):
        chunk = test_dataset[i]

        # add metadata to predictions for span eval
        fairytale_id = chunk["fairytale_id"]
        offset_mapping = chunk["offset_mapping"]

        # Trainer can return chunks padded
        # only take real token of chunk
        chunk_predictions = predicted_ids[i][:len(offset_mapping)]

        chunk_spans = transformer_bio_predictions_to_spans(
            predicted_ids=chunk_predictions,
            offset_mapping=offset_mapping,
        )

        predicted_spans_by_fairytale[fairytale_id].update(chunk_spans)

    return predicted_spans_by_fairytale


def prepare_test_spans(test_fairytales, predicted_spans_by_fairytale):

    all_true_spans = []
    all_pred_spans = []

    for fairytale in test_fairytales:
        gold_spans = {
            (span.start, span.end, span.label)
            for span in fairytale.resolve_nested_spans()
        }

        pred_spans = predicted_spans_by_fairytale.get(
            fairytale.khm,
            set(),
        )

        all_true_spans.append(gold_spans)
        all_pred_spans.append(pred_spans)

    return all_true_spans, all_pred_spans

def handle_token_prediction(test_dataset, predictions):
    all_true_labels = []
    all_pred_labels = []

    # take true labels from testset
    # still contain special tokens & padding
    for i in range(len(test_dataset)):
        chunk = test_dataset[i]
        gold_labels = chunk["labels"].tolist()
        all_true_labels.append(gold_labels)


    # take prediction labels
    # find token positions with character-offset: offset_mapping[token_position]
    # gather all double predictions (due to overlap): predicted_ids[chunk_index][token_position]
    # decide with pred score, which pred to keep: higher score from predictions.predictions


    return all_true_labels, all_pred_labels


def calculate_test_metrics(all_true_spans, all_pred_spans, all_true_labels, all_pred_labels):
    """
    Calculates metrics on span and on token level.
    :param all_true_spans: span level
    :param all_pred_spans: span level
    :param all_true_labels: token level
    :param all_pred_labels: token level
    :return:
    """
    precision, recall, f1 = span_metrics(all_true_spans, all_pred_spans,)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_true_labels, all_pred_labels, average='macro')

    metrics = {
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
        "token_precision": test_precision,
        "token_recall": test_recall,
        "token_f1": test_f1,
    }

    return metrics



def save_transformer_train_results(training_args, trainer, results_dir, current_date):
    """Save training arguments and log history to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    train_results_filename = f"{results_dir}/{current_date}_train_results.json"

    train_data = {
        'training_args': json.loads(training_args.to_json_string()),
        'log_history': trainer.state.log_history if hasattr(trainer.state, 'log_history') else []
    }

    with open(train_results_filename, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Training results saved to {train_results_filename}")



def save_transformer_test_results(test_results, results_dir, current_date):
    """Save validation metrics to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    test_results_filename = f"{results_dir}/{current_date}_test_results.json"
    with open(test_results_filename, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Validation results saved to {test_results_filename}")