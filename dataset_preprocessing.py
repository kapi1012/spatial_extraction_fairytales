import json

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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
- BIO-Labels
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
        return self.labels[0] if self.labels else None

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

dataset = FairytaleDataset.from_json(fairytale_annotated)

