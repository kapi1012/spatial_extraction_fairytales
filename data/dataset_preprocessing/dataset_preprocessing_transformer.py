import torch
from torch.utils.data import Dataset

from dataset_preprocessing import Fairytale
from constants import *


"""
- Tokenizer:
    - input_ids
    - attention_mask
    - offset_mapping
    - max_length = 512 (-> chunking longer fairytales)
        - stride = 128 (-> overlapping chunks)

- Label Alignment:
    - Label to tokenized Subtoken

- Class Span are offset-based -> Ground Truth can be checked against Subtoken

- HF DataCollatorForTokenClassification for padding: https://huggingface.co/docs/transformers/v5.14.0/en/main_classes/data_collator#transformers.DataCollatorForTokenClassification
"""


class RoBERTaSequenceTaggingDataset(Dataset):

    def __init__(self, chunks, label2id):
        self.chunks = chunks
        self.label2id = label2id

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        chunk = self.chunks[index]

        return {
            "input_ids": torch.tensor(
                chunk["input_ids"],
                dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                chunk["attention_mask"],
                dtype=torch.long
            ),

            "labels": torch.tensor(
                chunk["labels"],
                dtype=torch.long
            ),

            # Evaluation
            "offset_mapping": chunk["offset_mapping"],
            "fairytale_id": chunk["fairytale_id"],
            "chunk_idx": chunk["chunk_idx"],
        }

def tokenize_fairytale(fairytale: Fairytale, tokenizer, max_length: int = 512, stride: int = 128,):
    spans = fairytale.resolve_nested_spans()

    encoding = tokenizer(
        fairytale.text,
        max_length=max_length,
        stride=stride,
        truncation=True,
        return_overflowing_tokens=True, # can be addresses with .overflow_to_sample_mapping -> to which text belongs which chunk
        return_offsets_mapping=True,
        padding=False,
        add_special_tokens=True,
    )

    chunks = []

    for chunk_idx in range(len(encoding["input_ids"])):

        input_ids = encoding["input_ids"][chunk_idx]
        attention_mask = encoding["attention_mask"][chunk_idx]
        offsets = encoding["offset_mapping"][chunk_idx]


        labels = align_labels(
            offset_mapping=offsets,
            spans=spans,
            label2id=LABEL2ID,
        )


        chunks.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offsets,
            "labels": labels,
            "fairytale_id": fairytale.khm, # to find original fairytale
            "chunk_idx": chunk_idx, # sequence in one fairytale

        })

    return chunks


# Label Alignment https://huggingface.co/docs/transformers/tasks/token_classification?utm_.com
# Span to BIO-Label
def align_labels(offset_mapping, spans, label2id):
    """"
    Assign labels to subword tokens.
    """
    labels = []

    for token_start, token_end in offset_mapping:
        # Special tokens <s>, </s>
        if token_start == token_end:
            labels.append(-100) # for Training -> Pytorch Standard -> will be ignored for train loss
            continue

        label = "O"

        for span in spans:
            # No overlap between token and span
            if token_end <= span.start or token_start >= span.end:
                continue

            # Token overlaps span
            if token_start == span.start:
                label = f"B-{span.label}"
            else:
                label = f"I-{span.label}"

            break
        labels.append(label2id[label])

    return labels
