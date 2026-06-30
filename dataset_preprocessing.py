import json

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from spacy.lang.de import German

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

@dataclass
class Fairytale:
    khm: int
    title: str
    text: str
    spans: list[Span] = field(default_factory=list) # rename from JSON "label" for clarity


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

#class FairytaleSequenceTaggingDataset(Dataset):
#    """
#    PyTorch Dataset for sequence tagging tasks.
#    Builds Tokens & BIO Labels
#    """
#
#    def __init__(self, dataset: FairytaleDataset):
#        self.dataset = dataset
#
#    def __len__(self):
#        return len(self.dataset.fairytales)
#
#    def __getitem__(self, idx):
#        pass



DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_25.06.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    fairytale_annotated = json.load(f)

data = FairytaleDataset.from_json(fairytale_annotated)

# Bi-LSTM Preparation
# SpaCy Tokenizer for text in every fairytale
nlp = German() # contains tokenizer, no other SpaCy Pipeline Components
texts = (fairytale.text for fairytale in data.fairytales)


for fairytale, doc in zip(data.fairytales, nlp.pipe(texts)): # pipe() using all docs
    spacy_spans = []
    for annotation in fairytale.spans: # Label-Studio Annotations
        spacy_span = doc.char_span(annotation.start, annotation.end, label=annotation.label) # Maps to SpaCy-Spans
        if spacy_span is None:
            continue

        # check if spans are correctly assigned:
        assert spacy_span is not None
        assert spacy_span.text == annotation.text

        spacy_spans.append(spacy_span)
        # IOB-Tag
    doc.ents = spacy_spans

    for token in doc[:20]:
        print(token.text, token.ent_iob_, token.ent_type_)


# Bi-LSTM Dataclass

