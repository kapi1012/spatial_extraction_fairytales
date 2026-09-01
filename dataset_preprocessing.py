import json

from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from spacy.lang.de import German
from spacy.tokens import Doc, Span

from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.data import Sentence


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
 ├───────────────────────────────────────────────────┐
 ▼                                                   ▼
SpaCyPreprocessor                           HuggingFace RoBERTa Tokenizer
- Tokenizer                                     - input_ids
- Character Offsets                             - attention_mask
- Sentence Boundaries                           - offset_mapping 
 │                                                   │
 ▼                                                   ▼
BiLSTMBuilder                               TransformerBuilder
- IO(B)-Labels                              - Character to Subword
- Token Documents                                    │   
- Flair Embeddings                                   ▼
- FairytaleSequenceTaggingDataset                 IOB-Labels 
- Collate_fn for Sequence Padding + Masking          │
 │                                                   ▼
 │                                            HuggingFace Dataset  
 │                                                   │
 │                                                   │
 ▼                                                   ▼
TorchDataset                                   DataCollatorForTokenClassification
      
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
    khm: int
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int]]
    labels: list[str]
    sentence_boundaries: list[tuple[int, int]] # Boundaries for SpaCy Sentenizer
    embeddings: torch.Tensor | None = None # FLAIR Embeddings

@dataclass
class SentenceDocument:
    khm: int
    tokens: list[str]
    labels: list[str]
    embeddings: torch.Tensor | None = None

class FairytaleSequenceTaggingDataset(Dataset):
    """
    PyTorch Dataset for sequence tagging tasks.
    """

    def __init__(self, documents: list[SentenceDocument], label2id):
        self.documents = documents
        self.label2id = label2id

    def __len__(self):
        return len(self.documents)

    # Documents to pytorch Tensors
    def __getitem__(self, idx: int):
        document = self.documents[idx]
        tag_ids = [self.label2id[label] for label in document.labels]

        return {
            "embeddings": document.embeddings,
            "tags": torch.tensor(tag_ids),
            "khm": document.khm,
        }


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
    nlp.add_pipe("sentencizer") # add segmentation by sentence for FLAIR Embeddings

    texts = (fairytale.text for fairytale in dataset.fairytales)

    docs = []
    all_errors = []

    for fairytale, doc in zip(dataset.fairytales, nlp.pipe(texts)): # pipe() using all docs
        doc.user_data["khm"] = fairytale.khm

        spacy_spans, errors = build_spacy_spans(fairytale, doc)
        doc.ents = spacy_spans

        docs.append(doc)
        all_errors.extend(errors)

    return docs, all_errors


DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_3.7.26.json"
#dataset = load_dataset(DATA_PATH)

#docs, errors = annotate_dataset(dataset)

#if errors:
#    print(f"\nFound {len(errors)} errors:\n")
#    for error in errors:
#        print(error)
#else:
#    print("No errors found.")

def doc_to_token_document(doc: Doc) -> TokenDocument:

    sentence_boundaries = []

    # Save sentence boundaries for FLAIR
    for sent in doc.sents:
        sentence_boundaries.append((sent.start, sent.end))


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
        khm=doc.user_data["khm"],
        text=doc.text,
        tokens=tokens,
        offsets=offsets,
        labels=labels,
        sentence_boundaries=sentence_boundaries,
    )

def create_token_documents(docs: list[Doc]) -> list[TokenDocument]:
    return [doc_to_token_document(doc) for doc in docs]


def flair_embedding(token_documents: list[TokenDocument]) -> list[SentenceDocument]:
    """
    Embeds the Fairytale-Texts on sentence-level for BiLSTM.
    """

    stacked_embeddings = StackedEmbeddings([
        WordEmbeddings('de-crawl'), # FastText for Subword Information
        FlairEmbeddings('de-historic-rw-forward'),
        FlairEmbeddings('de-historic-rw-backward'),
    ])

    sentence_documents = []

    for doc in token_documents:

        for start, end in doc.sentence_boundaries:
            sentence = Sentence(doc.tokens[start:end])
            with torch.no_grad():
                stacked_embeddings.embed(sentence)
            sentence_embedding = torch.stack([token.embedding.cpu() for token in sentence])

            sentence_documents.append(
                SentenceDocument(
                    khm=doc.khm,
                    tokens=doc.tokens[start:end],
                    labels=doc.labels[start:end],
                    embeddings=sentence_embedding
                )
            )

    torch.save(sentence_documents, "data/flair_embeddings/embeddings_21.8.26.pt")
    return sentence_documents


#token_documents = create_token_documents(docs)
#sentence_documents = flair_embedding(token_documents)
#dataset = FairytaleSequenceTaggingDataset(sentence_documents, label2id=LABEL2ID)

def collate_fn_bilstm(batch: list[dict[str, Tensor]]):
    """"
    For padding & masking all Tensors from Datasets.
    Padding -> same size in each batch
    Masking -> defines which tokens are real, which are padding
    Used by Pytorch DataLoader.
    """
    embeddings = [sample["embeddings"] for sample in batch]
    tags = [sample["tags"] for sample in batch]
    khms = [sample["khm"] for sample in batch]

    # Padding for a batch
    lengths = [e.size(0) for e in embeddings]

    embeddings = pad_sequence(embeddings, batch_first=True)

    tags = pad_sequence(tags, batch_first=True)

    # Bool mask for FLAIR Embedding
    mask = torch.zeros(len(batch), embeddings.size(1), dtype=torch.bool)

    for i, length in enumerate(lengths):
        mask[i, :length] = True

    return {
        "embeddings": embeddings,
        "tags": tags,
        "mask": mask,
        "khm": khms,
    }