import json

from dataclasses import dataclass, field
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# nach: https://www.geeksforgeeks.org/deep-learning/how-do-you-use-pytorchs-dataset-and-dataloader-classes-for-custom-data/


# Dataclass
@dataclass
class Span(Dataset):
    start: int
    end: int
    labels: list[str]
    text: str

@dataclass
class Fairytale(Dataset):
    khm: int
    title: str
    fairytale_text: str
    labels: list[Span] = field(default_factory=list)


@dataclass
class FairytaleDataset(Dataset):
    fairytales: list[Fairytale] = field(default_factory=list)

    def __len__(self):
        return len(self.fairytales)

    def __getitem__(self, idx):
        fairytale = self.fairytales[idx]
        return {
            'khm': fairytale.khm,
            'title': fairytale.title,
            'fairytale_text': fairytale.fairytale_text,
            'labels': fairytale.labels,
        }

    @classmethod
    def from_json(cls, data):
        fairytales = []
        for item in data:
            spans = [Span(**span) for span in item['labels']]
            fairytale = Fairytale(
                khm=item['khm'],
                title=item['title'],
                fairytale_text=item['text'],
                labels=spans,
            )
            fairytales.append(fairytale)
        return cls(fairytales)

DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_25.06.json"

with open(DATA_PATH, "r", encoding="utf-8") as f:
    fairytale_annotated = json.load(f)

dataset = FairytaleDataset.from_json(fairytale_annotated)


# Train-test-split
#todo

# Dataloader
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)