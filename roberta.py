import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataset_preprocessing_transformer import *
from dataset_preprocessing import load_dataset
from constants import LABEL2ID
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from functools import partial
from data_split import data_split_on_khm, split_dataset_by_khm

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

transformer_collate_fn = partial(
    collate_fn_transformer,
    tokenizer=tokenizer
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=transformer_collate_fn
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=transformer_collate_fn
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    collate_fn=transformer_collate_fn
)
