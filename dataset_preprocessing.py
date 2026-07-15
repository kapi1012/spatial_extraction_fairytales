import json

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from spacy.lang.de import German
from spacy.tokens import Doc, Span

from constants import LABEL2ID

# https://www.geeksforgeeks.org/deep-learning/how-do-you-use-pytorchs-dataset-and-dataloader-classes-for-custom-data/

"""
Label Studio Export / JSON Format:

Fairytale
│
├── text
├── title
├── khm
└── spans (called "label" in JSON; multiple spans per text)
      ├── start
      ├── end
      ├── text
      └── labels 
      
Pipeline:

JSON
 │
 ▼
FairytaleDataset
- Python Dataclass
 │
 ├──────────────────────────────────┐
 ▼                                  ▼
SpaCyPreprocessor           HuggingFace Tokenizer
- Tokenizer                   - offset_mapping
- Character Offsets                 │ 
 │                                  │
 ▼                                  ▼
BiLSTMBuilder              TransformerBuilder
- IO(B)-Labels
- Token Documents
- Vocabulary
- FairytaleSequenceTaggingDataset
- Collate_fn for Sequence Padding + Masking
 │                                 │
 ▼                                 ▼
TorchDataset                   TorchDataset
      
"""

# Dataclass
@dataclass
class Span:
    start: int
    end: int
    text: str
    labels: list[str]

    @property
    def label(self) -> str | None:
        return self.labels[0] if self.labels else None # There is always only one Label per Span

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Fairytale:
    khm: int
    title: str
    text: str
    spans: list[Span] = field(default_factory=list) # rename from JSON "label" for clarity

    def resolve_nested_spans(self) -> list[Span]:
        """
        Resolves nested and overlapping spans and keeps the longest overlapping span.
        Nested Spans can not be used for IOB-Format Bi-LSTMs and Transformers.
        Returns a list of Spans without modifying the original Spans.

        """
        # Sort spans by start offset; for equal start offsets keep the longest span first.
        sorted_spans = sorted(self.spans, key=lambda span: (span.start, -span.length))
        resolved_spans = []

        for span in sorted_spans:
            overlap = False
            for kept_span in resolved_spans:
                # check if span lies in another span and discard it
                if span.start < kept_span.end and kept_span.start < span.end:
                    overlap = True
                    break
            if not overlap:
                resolved_spans.append(span)

        return resolved_spans


@dataclass
class FairytaleDataset:
    """"
    Class with basic text & spans
    Labels are offset characters
    """
    fairytales: list[Fairytale] = field(default_factory=list)

    @classmethod
    def from_json(cls, data):
        fairytales = []
        for item in data:
            fairytale = Fairytale(
                khm=item['khm'],
                title=item['title'],
                text=item['text'],
                spans=[
                    Span(start=span["start"],
                        end=span["end"],
                        text=span["text"],
                        labels=span["labels"])
                    for span in item.get("label", [])
                ]
            )
            fairytales.append(fairytale)
        # sort fairytales by khm-number for debugging
        fairytales.sort(key=lambda x: x.khm)
        return cls(fairytales=fairytales)

@dataclass
class TokenDocument:
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int]]
    labels: list[str]

class FairytaleSequenceTaggingDataset(Dataset):
    """
    PyTorch Dataset for sequence tagging tasks.
    """

    def __init__(self, token_documents: list[TokenDocument], word2id, label2id):
        self.documents = token_documents
        self.word2id = word2id
        self.label2id = label2id

    def __len__(self):
        return len(self.documents)

    # Documents to pytorch Tensors
    def __getitem__(self, idx: int):
        token_document = self.documents[idx]
        input_ids = []
        tag_ids = []

        for token in token_document.tokens:
            input_ids.append(self.word2id.get(token, self.word2id["<UNK>"]))

        for label in token_document.labels:
            tag_ids.append(self.label2id[label])

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        tag_ids = torch.tensor(tag_ids, dtype=torch.long)

        return {"input_ids": input_ids, "tags": tag_ids,}


def load_dataset(datapath: str) -> FairytaleDataset:
    with open(datapath, "r", encoding="utf-8") as f:
        fairytale_annotated = json.load(f)
    return FairytaleDataset.from_json(fairytale_annotated)

def build_spacy_spans(fairytale: Fairytale, doc: Doc) -> tuple[list[Span], list[str]]:
    resolved_spans = fairytale.resolve_nested_spans()
    spacy_spans = []
    errors = []

    for annotation in resolved_spans:  # Label-Studio Annotations
        spacy_span = doc.char_span(annotation.start, annotation.end, label=annotation.label)  # Maps to SpaCy-Spans

        if spacy_span is None:
            errors.append(
                f"KHM {fairytale.khm} ({fairytale.title}): "
                f"Could not map span '{annotation.text}' "
                f"[{annotation.start}, {annotation.end}]"
            )
            continue

        if spacy_span.text != annotation.text:
            errors.append(
                f"KHM {fairytale.khm} ({fairytale.title}): "
                f"Span mismatch\n"
                f"JSON : {annotation.text!r}\n"
                f"spaCy: {spacy_span.text!r}"
            )
            continue

        # check if spans are correctly assigned:
        assert spacy_span.text == annotation.text

        spacy_spans.append(spacy_span)

    return spacy_spans, errors

    #for token in doc[:20]:
    #    print(token.text, token.ent_iob_, token.ent_type_)

def annotate_dataset(dataset: FairytaleDataset,) -> tuple[list[Doc], list[str]]:

    # SpaCy Tokenizer for text in every fairytale
    nlp = German()  # contains tokenizer, no other SpaCy Pipeline Components

    texts = (fairytale.text for fairytale in dataset.fairytales)

    docs = []
    all_errors = []


    for fairytale, doc in zip(dataset.fairytales, nlp.pipe(texts)): # pipe() using all docs
        spacy_spans, errors = build_spacy_spans(fairytale, doc)
        doc.ents = spacy_spans

        docs.append(doc)
        all_errors.extend(errors)

    return docs, all_errors


DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_3.7.26.json"
dataset = load_dataset(DATA_PATH)

docs, errors = annotate_dataset(dataset)

if errors:
    print(f"\nFound {len(errors)} errors:\n")
    for error in errors:
        print(error)
else:
    print("No errors found.")

def doc_to_token_document(doc: Doc) -> TokenDocument:
    tokens = []
    offsets = []
    labels = []
    for token in doc:
        tokens.append(token.text)
        offsets.append((token.idx, token.idx + len(token)))
        if token.ent_iob_ == "O":
            labels.append("O")
        else:
            labels.append(f"{token.ent_iob_}-{token.ent_type_}")

    return TokenDocument(
        text=doc.text,
        tokens=tokens,
        offsets=offsets,
        labels=labels,
    )

def create_token_documents(docs: list[Doc]) -> list[TokenDocument]:
    return [doc_to_token_document(doc) for doc in docs]

def build_vocabulary(token_documents):
    word2id = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for doc in token_documents:
        for token in doc.tokens:
            if token not in word2id:
                word2id[token] = len(word2id)

    id2word = {v: k for k, v in word2id.items()}
    return word2id, id2word


token_documents = create_token_documents(docs)
word2id, id2word = build_vocabulary(token_documents)

dataset = FairytaleSequenceTaggingDataset(token_documents, word2id, label2id=LABEL2ID)