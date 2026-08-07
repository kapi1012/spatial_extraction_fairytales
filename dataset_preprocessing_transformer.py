from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from constants import LABEL2ID
from dataset_preprocessing import Span, Fairytale, FairytaleDataset, load_dataset

from huggingface_hub import DataCollatorForTokenClassification


"""
- Tokenizer:
    - input_ids
    - attention_mask
    - offset_mapping
    - max_length = 512 (-> chunking longer fairytales)
        - stride = 128 (-> overlapping chunks)

- Class Span are offset-based -> Ground Truth can be checked against Subtoken

- HF DataCollatorForTokenClassification for padding: https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/data_collator#transformers.DataCollatorForTokenClassification
"""


@dataclass
class TransformerDocument:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]




# Source - https://stackoverflow.com/a/77066945
# Posted by nbeuchat
# Retrieved 2026-08-07, License - CC BY-SA 4.0

def collate_fn_transformer(data: List[Tuple[torch.Tensor, torch.Tensor]]):
    tensors, targets = zip(*data)
    features = pad_sequence(tensors, batch_first=True)
    targets = torch.stack(targets)
    return features, targets
