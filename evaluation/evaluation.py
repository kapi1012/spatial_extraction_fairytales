import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter
import os
import json

from sklearn.metrics import classification_report, precision_recall_fscore_support
from constants import LABEL2ID, ID2LABEL, LABEL_MAP, ENTITY_TYPES
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

    prediction_documents = []

    with torch.no_grad():

        for batch in test_data:
            embeddings = batch["embeddings"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            predictions = model.decode(embeddings, mask)

            for i, (pred, gold, valid) in enumerate(zip(predictions, tags, mask)):

                length = valid.sum().item()

                # Remove padding
                pred = pred[:length]

                gold = gold[:length].cpu().tolist()

                # Convert prediction IDs to labels
                pred_labels = [ID2LABEL[tag_id] for tag_id in pred]

                # Convert gold IDs to labels
                gold_labels = [
                    ID2LABEL[tag_id] for tag_id in gold]
                tokens = batch["tokens"][i][:length]

                # Token-level evaluation
                all_predictions.extend(pred_labels)
                all_true_tags.extend(gold_labels)

                # Span-level evaluation
                pred_spans = tags_to_spans(pred_labels)
                gold_spans = tags_to_spans(gold_labels)

                all_pred_spans.append(pred_spans)
                all_true_spans.append(gold_spans)


                prediction_documents.append({
                    "khm": int(batch["khm"][i]),
                    "id": int(batch["id"][i]),
                    "tokens": list(tokens),
                    "true_tags": gold_labels,
                    "predicted_tags": pred_labels,
                })

    return all_true_tags, all_predictions, all_true_spans, all_pred_spans, prediction_documents

def span_micro_metrics(all_true_spans, all_pred_spans):

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


def span_metrics(all_true_spans, all_pred_spans, entity_types):
    # calculates on macro span level without O-Label
    counts = {
        entity_type: {"tp": 0, "fp": 0, "fn": 0}
        for entity_type in entity_types
    }

    for gold, pred in zip(all_true_spans, all_pred_spans):
        tp_spans = gold & pred
        fp_spans = pred - gold
        fn_spans = gold - pred

        for start, end, entity_type in tp_spans:
            if entity_type in counts:
                counts[entity_type]["tp"] += 1

        for start, end, entity_type in fp_spans:
            if entity_type in counts:
                counts[entity_type]["fp"] += 1

        for start, end, entity_type in fn_spans:
            if entity_type in counts:
                counts[entity_type]["fn"] += 1

    precisions = []
    recalls = []
    f1_scores = []

    for entity_type in entity_types:
        tp = counts[entity_type]["tp"]
        fp = counts[entity_type]["fp"]
        fn = counts[entity_type]["fn"]

        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return macro_precision, macro_recall, macro_f1


def save_test_report(true_tags, predictions, true_spans, pred_spans, hyperparameters):

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    filename = f"data/test_reports/{current_time}.csv"

    # exclude O-Label from report
    entity_labels = [
        label for label in LABEL2ID.keys()
        if label != "O"
    ]

    # Token-level metrics
    report = classification_report(
        true_tags, predictions,
        labels=entity_labels,
        target_names=entity_labels,
        output_dict=True,
        zero_division=0,)
    df_report = pd.DataFrame(report).T

    # Span-level metrics
    entity_types = sorted({label[2:] for label in ID2LABEL.values() if label.startswith(("B-", "I-"))})
    span_precision, span_recall, span_f1 = span_metrics(true_spans, pred_spans, entity_types)

    df_report.loc["span_precision", "precision"] = span_precision
    df_report.loc["span_recall", "recall"] = span_recall
    df_report.loc["span_f1", "f1-score"] = span_f1

    for key, value in hyperparameters.items():
        df_report[key] = value
    df_report = df_report.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    df_report.to_csv(filename)


def save_bilstm_predictions_by_khm(
    prediction_documents
):
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    filename = f"data/evaluation/bilstm_predictions_{current_time}.json"

    grouped = defaultdict(list)

    for doc in prediction_documents:
        grouped[doc["khm"]].append(doc)

    results = {}

    for khm, sentences in sorted(grouped.items()):
        sentence_results = []

        for sentence_id, doc in enumerate(sentences):
            tokens = doc["tokens"]
            true_tags = doc["true_tags"]
            predicted_tags = doc["predicted_tags"]

            def serialize_spans(spans):
                return [
                    {
                        "start": start,
                        "end": end,
                        "text": " ".join(tokens[start:end]),
                        "label": label,
                    }
                    for start, end, label in sorted(spans)
                ]

            true_spans = tags_to_spans(true_tags)
            predicted_spans = tags_to_spans(predicted_tags)

            sentence_results.append({
                "sentence_id": sentence_id,
                "id": doc["id"],
                "tokens": tokens,
                "true_tags": true_tags,
                "predicted_tags": predicted_tags,
                "token_errors": [
                    {
                        "token_index": i,
                        "token": token,
                        "true_tag": true_tag,
                        "predicted_tag": pred_tag,
                    }
                    for i, (token, true_tag, pred_tag) in enumerate(
                        zip(tokens, true_tags, predicted_tags)
                    )
                    if true_tag != pred_tag
                ],
                "true_spans": serialize_spans(true_spans),
                "predicted_spans": serialize_spans(predicted_spans),
            })

        results[str(khm)] = {
            "khm": khm,
            "sentences": sentence_results,
        }

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


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


def resolved_tokens_to_spans(resolved_tokens):
    """
    Reconstructs predicted spans from resolved tokens.

    Tokens were sorted by characteroffset per fairytale.
    Span recontruction after resolving overlap.
    """

    labels_by_fairytale = defaultdict(list)

    for (fairytale_id, start, end), observation in resolved_tokens.items():
        labels_by_fairytale[fairytale_id].append({
            "start": start,
            "end": end,
            "pred_label": observation["pred_label"],
        })

    predicted_spans_by_fairytale = defaultdict(set)

    for fairytale_id, tokens in labels_by_fairytale.items():
        tokens.sort(key=lambda token: (token["start"], token["end"]))

        offsets = [
            (token["start"], token["end"])
            for token in tokens
        ]

        predicted_ids = [
            token["pred_label"]
            for token in tokens
        ]

        #print(f"\nMärchen-ID: {fairytale_id}")
        #print("Anzahl Tokens:", len(tokens))
        #print(
        #    "Vorhergesagte Labels:",
        #    Counter(ID2LABEL[label_id] for label_id in predicted_ids),
        #)

        chunk_spans = transformer_bio_predictions_to_spans(
            predicted_ids=predicted_ids,
            offset_mapping=offsets,
        )

        #print("Anzahl rekonstruierter Spans:", len(chunk_spans))
        #print("Beispiel-Spans:", chunk_spans[:5])

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


def collect_token_labels(test_dataset, predicted_ids, predictions):
    token_observations = defaultdict(list)
    logits = predictions.predictions

    for chunk_idx in range(len(test_dataset)):
        chunk = test_dataset[chunk_idx]

        fairytale_id = chunk["fairytale_id"]
        offset_mapping = chunk["offset_mapping"]
        true_labels = chunk["labels"].tolist()

        for token_position, (start, end) in enumerate(offset_mapping):

            # Special tokens
            if start == end:
                continue

            # Prediction on same token position
            pred_label = int(predicted_ids[chunk_idx][token_position])

            # Gold label
            true_label = true_labels[token_position]

            # skip padding
            if true_label == -100:
                continue

            # Softmax probabilities
            probs = torch.softmax( torch.tensor( logits[chunk_idx, token_position], dtype=torch.float32 ), dim=-1 )
            confidence = probs[pred_label].item()

            token_key = (fairytale_id, start, end)

            token_observations[token_key].append({
                "chunk_idx": chunk["chunk_idx"],
                "token_position": token_position,
                "true_label": true_label,
                "pred_label": pred_label,
                "logit": logits[
                    chunk_idx,
                    token_position,
                    pred_label
                ],
                "confidence": confidence,
            })

    return dict(sorted(token_observations.items()))


def resolve_overlap_predictions(token_observations):
    """
    Labeled sequences can have an overlap due to chunking with a stride.
    3 Cases:
    1. Token only appears in one chunk -> keep.
    2. Token appears in two chunks, but have same label -> keep.
    3. Token appears in two chunks, but have different label -> keep token with higher softmax confidence.
        -> predictions contain logits (predicted_ids contain label)
    """
    resolved_tokens = {}
    same_prediction = 0
    different_prediction = 0

    for token_key, observations in token_observations.items():

        # Case 1
        if len(observations) == 1:
            resolved_tokens[token_key] = observations[0]

        # Case 2
        elif len(observations) == 2:
            obs1, obs2 = observations

            if obs1["true_label"] != obs2["true_label"]:
                raise ValueError(
                    f"Different gold labels for token {token_key}: "
                    f"{obs1['true_label']} vs. {obs2['true_label']}"
                )

            # same predictions
            if obs1["pred_label"] == obs2["pred_label"]:
                same_prediction += 1
                resolved_tokens[token_key] = obs1

            # case 3: higher logit
            else:
                different_prediction += 1

                #compare softmax
                if obs1["confidence"] >= obs2["confidence"]:
                    resolved_tokens[token_key] = obs1
                else:
                    resolved_tokens[token_key] = obs2

        else:
            raise ValueError(
                f"Token {token_key} kommt {len(observations)} x vor."
            )

    print("Same prediction in overlap:", same_prediction)
    print("Different prediction in overlap:", different_prediction)

    return resolved_tokens


def extract_token_labels(resolved_tokens):
    all_true_labels = []
    all_pred_labels = []

    for observation in resolved_tokens.values():
        all_true_labels.append(observation["true_label"])
        all_pred_labels.append(observation["pred_label"])


    return all_true_labels, all_pred_labels


def calculate_test_metrics(all_true_spans, all_pred_spans, all_true_labels, all_pred_labels):
    """
    Calculates metrics on span and on token level.
    :param all_true_spans: span-level
    :param all_pred_spans: span-level
    :param all_true_labels: token-level
    :param all_pred_labels: token-level
    :return:
    """
    span_precision, span_recall, span_f1 = span_metrics(all_true_spans, all_pred_spans, ENTITY_TYPES)

    # Exclude O from token-level classification report
    entity_label_ids = [
        label_id
        for label, label_id in LABEL2ID.items()
        if label != "O"
    ]

    target_names = [
        ID2LABEL[label_id]
        for label_id in entity_label_ids
    ]

    token_report = classification_report(
        all_true_labels,
        all_pred_labels,
        labels=entity_label_ids,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    # Combined metrics
    metrics = {
        "span_metrics": {
            "precision": float(span_precision),
            "recall": float(span_recall),
            "f1": float(span_f1),
        },
        "token_metrics": token_report,
    }

    return metrics

# compute metrics to validate on best span_f1 on training
def make_compute_metrics(val_dataset, val_fairytales):

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(logits, tuple):
            logits = logits[0]

        predicted_ids = np.argmax(logits, axis=-1)

        # 1. Token-Vorhersagen sammeln
        token_observations = collect_token_labels(
            val_dataset,
            predicted_ids,
            eval_pred,
        )

        # 2. Overlaps auflösen
        resolved_tokens = resolve_overlap_predictions(
            token_observations
        )

        # 3. Spans rekonstruieren
        predicted_spans_by_fairytale = resolved_tokens_to_spans(
            resolved_tokens
        )

        # 4. Gold- und vorhergesagte Spans vergleichen
        all_true_spans, all_pred_spans = prepare_test_spans(
            val_fairytales,
            predicted_spans_by_fairytale,
        )

        span_precision, span_recall, span_f1 = span_metrics(
            all_true_spans,
            all_pred_spans,
            ENTITY_TYPES,
        )

        return {
            "span_precision": span_precision,
            "span_recall": span_recall,
            "span_f1": span_f1,
        }

    return compute_metrics


def save_transformer_train_results(training_args, trainer, results_dir, current_date):
    """Save training arguments and log history to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    train_results_filename = f"{results_dir}/{current_date}_train_results.json"

    train_data = {
        'training_args': json.loads(training_args.to_json_string()),
        'log_history': trainer.state.log_history if hasattr(trainer.state, 'log_history') else []
    }

    with open(train_results_filename, 'w', encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)
    print(f"Training results saved to {train_results_filename}")



def save_transformer_test_results(test_results, training_args, results_dir, current_date):
    """Save test metrics to JSON file."""
    os.makedirs(results_dir, exist_ok=True)
    test_results_filename = f"{results_dir}/{current_date}_test_results.json"

    test_data = {
        "test_results": test_results,
        "training_args": json.loads(
            training_args.to_json_string()
        ),
    }

    with open(test_results_filename, 'w', encoding="utf-8") as f:
        json.dump(test_data, f, indent=2)
    print(f"Test results saved to {test_results_filename}")


def save_transformer_predictions_json(
        resolved_tokens,
        predicted_spans_by_fairytale,
        test_fairytales):
    """
    prediction_documents: Liste von Dictionaries mit:
        khm, title, sentences

    Jede Sentence enthält:
        tokens, true_tags, predicted_tags
    """
    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    filename = f"data/evaluation/roberta_predictions_{current_time}.json"

    # Lookup der Originalmärchen.
    fairytale_lookup = {
        ft.khm: ft for ft in test_fairytales
    }

    # Tokens nach Märchen gruppieren.
    tokens_by_fairytale = defaultdict(list)


    for (fairytale_id, start, end), observation in resolved_tokens.items():
        tokens_by_fairytale[fairytale_id].append({
            "start": start,
            "end": end,
            "text": (
                fairytale_lookup[fairytale_id].text[start:end]
                if fairytale_id in fairytale_lookup
                else None
            ),
            "true_tag": ID2LABEL[observation["true_label"]],
            "predicted_tag": ID2LABEL[observation["pred_label"]],
            "confidence": observation["confidence"],
        })

    results = {}

    for fairytale in test_fairytales:
        fairytale_id = fairytale.khm
        khm = fairytale.khm

        tokens = sorted(
            tokens_by_fairytale.get(fairytale_id, []),
            key=lambda token: (token["start"], token["end"]),
        )

        token_errors = [
            token for token in tokens
            if token["true_tag"] != token["predicted_tag"]
        ]

        # Gold-Spans aus den Originalannotationen.
        true_spans = [
            {
                "start": span.start,
                "end": span.end,
                "text": span.text,
                "label": span.label,
            }
            for span in fairytale.spans
        ]

        # Vorhergesagte Spans aus deiner bestehenden Rekonstruktion.
        predicted_spans = [
            {
                "start": start,
                "end": end,
                "text": fairytale.text[start:end],
                "label": label,
            }
            for start, end, label
            in sorted(predicted_spans_by_fairytale.get(fairytale_id, set()))
        ]

        results[str(fairytale.khm)] = {
            "khm": fairytale.khm,
            "fairytale_id": fairytale_id,
            "title": fairytale.title,
            "text": fairytale.text,
            "tokens": tokens,
            "token_errors": token_errors,
            "true_spans": true_spans,
            "predicted_spans": predicted_spans,
        }

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


# Local LLM
# mapping from old to new labels
def normalize_label(label):
    return LABEL_MAP.get(label, label)

def extract_test_spans(test_fairytales, annotated_fairytales):
    all_true_spans = []
    all_pred_spans = []
    evaluation_errors = []

    # sort preds with khm
    predicted_by_khm = {
        fairytale.khm: {
            (span.start, span.end, normalize_label(span.label))
            for span in fairytale.spans
        }
        for fairytale in annotated_fairytales
    }

    for fairytale in test_fairytales:

        gold_spans = {
            (span.start, span.end, normalize_label(span.label))
            for span in fairytale.resolve_nested_spans()
        }

        if fairytale.khm not in predicted_by_khm:
            evaluation_errors.append(
                {"khm": fairytale.khm,
                "reason": "prediction_missing",
             })

            pred_spans = set()
        else:
        # pred-spans for same fairytale
            pred_spans = predicted_by_khm[fairytale.khm]

        all_true_spans.append(gold_spans)
        all_pred_spans.append(pred_spans)

    return all_true_spans, all_pred_spans, evaluation_errors

def save_local_llm_test_results(precision, recall, f1, evaluation_errors, results_dir, model_name, annotation_time, number_sentences):
    """Save test metrics to JSON file."""
    current_date = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    os.makedirs(results_dir, exist_ok=True)
    test_results_filename = os.path.join(results_dir, f"test_results_{current_date}.json")

    test_data = {
        "span_precision": round(precision, 3),
        "span_recall": round(recall, 3),
        "span_f1": round(f1, 3),
        "model": model_name,
        "khm_without_preds": evaluation_errors,
        "annotation_time_min": round(annotation_time / 60, 3),
        "number_sentences_chunks": number_sentences,
    }


    with open(test_results_filename, "w", encoding="utf-8") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)