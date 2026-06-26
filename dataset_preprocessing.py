import json

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# https://www.geeksforgeeks.org/deep-learning/how-do-you-use-pytorchs-dataset-and-dataloader-classes-for-custom-data/

""""
Label Studio Export / JSON Format:

Fairytale
│
├── text
├── title
├── khm
└── spans (called "label" in JSON)
      ├── start
      ├── end
      ├── text
      └── labels
"""

# Dataclass
@dataclass
class Span:
    start: int
    end: int
    text: str
    labels: list[str]

    @property
    def label(self) -> str:
        return self.labels[0]

@dataclass
class Fairytale:
    khm: int
    title: str
    fairytale_text: str
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
                fairytale_text=item['text'],
                spans=[
                    Span(start=span["start"],
                        end=span["end"],
                        text=span["text"],
                        labels=span["labels"])
                    for span in item.get("label", [])
                ]
            )
            fairytales.append(fairytale)
        return cls(fairytales)

class FairytaleSequenceTaggingDataset(Dataset):
    """
    PyTorch Dataset for sequence tagging tasks.
    Builds Tokens & BIO Labels
    """

    def __init__(self, dataset: FairytaleDataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset.fairytales)

    def __getitem__(self, idx):
        fairytale = self.dataset.fairytales[idx]
        return {
            'khm': fairytale.khm,
            'title': fairytale.title,
            'fairytale_text': fairytale.fairytale_text,
            'labels': fairytale.labels,
        }



DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_25.06.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    fairytale_annotated = json.load(f)

dataset = FairytaleDataset.from_json(fairytale_annotated)


# Train-test-split
#todo

# Dataloader
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)