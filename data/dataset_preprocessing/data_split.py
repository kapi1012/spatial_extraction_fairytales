import pandas as pd
import random
from datast_preprocessing.dataset_preprocessing import FairytaleDataset, SentenceDocument

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


random.choice() https://www.geeksforgeeks.org/python/python-select-random-value-from-a-list/
"""

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
    print(f"Train: {train_khm}")
    print(f"Val: {val_khm}")
    print(f"Test: {test_khm}")

    return train_khm, val_khm, test_khm


# used for BiLSTM-Split
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

def split_dataset_by_khm(
        dataset: FairytaleDataset,
        train_khm: list[int],
        val_khm: list[int],
        test_khm: list[int],
):
    train_fairytales = [
        fairytale
        for fairytale in dataset.fairytales
        if fairytale.khm in train_khm
    ]

    val_fairytales = [
        fairytale
        for fairytale in dataset.fairytales
        if fairytale.khm in val_khm
    ]

    test_fairytales = [
        fairytale
        for fairytale in dataset.fairytales
        if fairytale.khm in test_khm
    ]

    return train_fairytales, val_fairytales, test_fairytales
