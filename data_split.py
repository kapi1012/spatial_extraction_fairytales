import numpy as np
import pandas as pd
import random
from dataset_preprocessing import load_dataset, FairytaleDataset, SentenceDocument

"""
Split data based on sentence count. -> Bi-LSTM is Sentence-based

Algorithm:
- count all sentences in dataset
- define 80-10-10 split
- counter
- repeat till counter >= 80 %:
    - pick a random fairytale
        - add sentence count to counter
    -> train
    - repeat till counter >= 90 %:
        - pick a random fairytale
        -> val
        - remaining data
        -> test

- repeat X times
    - define seeds (or save splits)

- compare label distribution in X data splits
    - 
    - choose most even distribution


random.choice() https://www.geeksforgeeks.org/python/python-select-random-value-from-a-list/
"""
DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_21-8-26_final.json"
ANALYSIS_PATH = "data/analysis_dataset/analysis_dataset.csv"

df = pd.read_csv(ANALYSIS_PATH)
df_numeric = df.apply(pd.to_numeric, errors='coerce')

def data_split_on_khm(analysis_path:str, seed:int) -> tuple[list[int], list[int], list[int]]:
    """
    Split data based on sentence count.
    Returns lists with khm-numbers based on split.
    """
    df = pd.read_csv(analysis_path)
    df = df.apply(pd.to_numeric, errors='coerce')
    sum_sentences = df["count sentences"].sum()

    counter = 0
    split_80 = int(round(((80 * sum_sentences) / 100), 0))
    split_90 = int(round(((90 * sum_sentences) / 100), 0))
    random.seed(seed)

    all_khm = list(df["khm"])
    train_khm = []
    val_khm = []

    while counter <= split_80:
        rnd_row = random.choice(all_khm)
        train_khm.append(rnd_row)
        sentences = df.loc[df["khm"] == rnd_row, "count sentences"].iloc[0]
        counter += sentences
        all_khm.remove(rnd_row)

    while counter <= split_90:
        rnd_row = random.choice(all_khm)
        val_khm.append(rnd_row)
        sentences = df.loc[df["khm"] == rnd_row, "count sentences"].iloc[0]
        counter += sentences
        all_khm.remove(rnd_row)

    test_khm = all_khm

    return train_khm, val_khm, test_khm

train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, seed=1)
print(train_khm)
print(val_khm)
print(test_khm)


dataset = load_dataset(DATA_PATH)
seed_list = [0, 5, 10, 42]

def compare_splits(seeds:list, analysis_path:str) -> int:
    """
    Takes Dataset and list of seeds to perform dataset splitting.
    Compares label distribution and returns seed with most even distribution.
    """
    ...


def split_documents_by_khm(
    documents: list[SentenceDocument],
    train_khm: list[int],
    val_khm: list[int],
    test_khm: list[int],
):
    train_documents = [
        doc for doc in documents
        if doc.khm in train_khm
    ]

    val_documents = [
        doc for doc in documents
        if doc.khm in val_khm
    ]

    test_documents = [
        doc for doc in documents
        if doc.khm in test_khm
    ]

    return train_documents, val_documents, test_documents





