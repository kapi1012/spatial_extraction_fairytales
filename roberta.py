from collections import Counter
from datetime import datetime
import inspect
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

from data.evaluation.evaluation import save_transformer_train_results, save_transformer_test_results,  \
    prepare_test_spans, pred_spans_to_fairytale, collect_token_labels, calculate_test_metrics, extract_token_labels, \
    resolve_overlap_predictions
from dataset_preprocessing_transformer import *
from dataset_preprocessing import load_dataset
from constants import LABEL2ID
from data_split import data_split_on_khm, split_dataset_by_khm
import numpy as np

# https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification

# model: https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german
TOKENIZER_NAME = "FacebookAI/xlm-roberta-large-finetuned-conll03-german"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

model = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german", device_map="auto")

# check: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ

# RoBERTa Tokenizer

DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_3.7.26.json"
dataset = load_dataset(DATA_PATH)

# choose own train-val-test-split
ANALYSIS_PATH = "data/analysis_dataset/analysis_dataset.csv"
train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, 42)
train_fairytales, val_fairytales, test_fairytales = split_dataset_by_khm(dataset, train_khm, val_khm, test_khm,)

train_chunks = []
val_chunks = []
test_chunks = []

for fairytale in train_fairytales:
    train_chunks.extend(tokenize_fairytale(fairytale, tokenizer))

for fairytale in val_fairytales:
    val_chunks.extend(tokenize_fairytale(fairytale, tokenizer))

for fairytale in test_fairytales:
    test_chunks.extend(tokenize_fairytale(fairytale, tokenizer))

train_dataset = RoBERTaSequenceTaggingDataset(train_chunks, LABEL2ID)
val_dataset = RoBERTaSequenceTaggingDataset(val_chunks, LABEL2ID)
test_dataset = RoBERTaSequenceTaggingDataset(test_chunks, LABEL2ID)

# DataCollator for padding of input_ids, attention_mask and -100 labels
# added custom metadata
data_collator = DataCollatorForTokenClassification(tokenizer)

# Trainer
args = TrainingArguments(
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,

    num_train_epochs=10,

    optim="adamw_torch",
    warmup_ratio=0.1,
    weight_decay=0.01,

    output_dir="transformer/trainer_output",
    logging_dir="transformer/train_log",
    logging_strategy="steps",
    logging_steps=50,

    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    label_smoothing_factor=0.0, # controls how confident the model should be
    load_best_model_at_end=True, # for choosing the best model based on eval metrics -> save it at the end

    remove_unused_columns=True, #default, keeps only input_ids, attention_mask and labels from DataSet class
    seed=42,
    push_to_hub=False,

)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
)

#Checking Input
#print(inspect.signature(model.forward))
#batch = next(iter(trainer.get_train_dataloader()))
#print(batch.keys())

current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
train_results_dir = "transformer/train_log"
test_results_dir = "transformer/test_results"

#Training
trainer.train()
trainer.model.save_pretrained(f"transformer/models/{current_time}_best_model")
save_transformer_train_results(args, trainer, train_results_dir, current_time)

#Testing
predictions = trainer.predict(test_dataset)

# convert probability prediction to label
predicted_ids = np.argmax(predictions.predictions, axis=-1,)

predicted_spans_by_fairytale = pred_spans_to_fairytale(test_dataset, predicted_ids)
# labels for span-level evaluation
all_true_spans, all_pred_spans = prepare_test_spans(test_fairytales, predicted_spans_by_fairytale)

# labels for token-level evaluation
token_observations = collect_token_labels(test_dataset, predicted_ids, predictions)
resolved_tokens = resolve_overlap_predictions(token_observations)
all_true_labels, all_pred_labels = extract_token_labels(resolved_tokens)

test_results = calculate_test_metrics(all_true_spans, all_pred_spans, all_true_labels, all_pred_labels)

print("Number of test chunks:", len(test_dataset))
print("Number of token observations:", len(token_observations))
print("Number of resolved tokens:", len(resolved_tokens))

print(test_results)
print("True labels:", len(all_true_labels))
print("Pred labels:", len(all_pred_labels))
assert len(all_true_labels) == len(all_pred_labels)

observation_counts = Counter(
    len(observations)
    for observations in token_observations.values()
)

print(observation_counts)

save_transformer_test_results(test_results, args, test_results_dir, current_time)
