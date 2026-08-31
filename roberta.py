import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from dataset_preprocessing_transformer import *
from dataset_preprocessing import load_dataset
from constants import LABEL2ID
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from functools import partial

# https://huggingface.co/docs/transformers/v4.27.2/en/model_doc/xlm-roberta#transformers.XLMRobertaForTokenClassification

# model: https://huggingface.co/FacebookAI/xlm-roberta-large-finetuned-conll03-german
TOKENIZER_NAME = "FacebookAI/xlm-roberta-large-finetuned-conll03-german"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

model = AutoModelForTokenClassification.from_pretrained("FacebookAI/xlm-roberta-large-finetuned-conll03-german", device_map="auto")



# check: https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ

# RoBERTa Tokenizer

DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_3.7.26.json"
dataset = load_dataset(DATA_PATH)

# Train-Test-Val-Split: 80-20-20
fairytales = dataset.fairytales

train_fairytales, temp_fairytales = train_test_split(
    fairytales,
    test_size=0.2,
    random_state=42
)

val_fairytales, test_fairytales = train_test_split(
    temp_fairytales,
    test_size=0.5,
    random_state=42
)

train_chunks = []
val_chunks = []
test_chunks = []

for fairytale in train_fairytales:
    train_chunks.extend(tokenize_fairytale(fairytale, tokenizer))

for fairytale in val_fairytales:
    val_chunks.extend(tokenize_fairytale(fairytale, tokenizer))

for fairytale in test_fairytales:
    test_chunks.extend(tokenize_fairytale(fairytale, tokenizer))

train_dataset = RoBERTaSequenceTaggingDataset(
    train_chunks,
    LABEL2ID
)

val_dataset = RoBERTaSequenceTaggingDataset(
    val_chunks,
    LABEL2ID
)

test_dataset = RoBERTaSequenceTaggingDataset(
    test_chunks,
    LABEL2ID
)

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


# Sorting fairytales
# fairytale_by_id = {fairytale.khm: fairytale for fairytale in dataset.fairytales}

