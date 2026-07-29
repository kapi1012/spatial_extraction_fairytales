import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from torchcrf import CRF

from constants import LABEL2ID, ID2LABEL
from dataset_preprocessing import collate_fn, FairytaleSequenceTaggingDataset

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
"""
Number of possible label tags = 5

0
B-FUZZY_LOC
I-FUZZY_LOC
B-METAPHORIC_LOC
I-METAPHORIC_LOC

"""

# Bi-LSTM
# after: https://www.codegenes.net/blog/pytorch-bilstm-crf-tutorial/
# But using pytorch CRF for forward algorithm, log-likelihood, viterbi-decoding and score
class BiLSTM_CRF(nn.Module):
    def __init__(self, num_tags, embedding_dim, hidden_dim, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Dropout for Regularization
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)


    def forward(self, embeddings):
        """
        Calculate emission scores from Flair embeddings.
        """
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def decode(self, embeddings, mask):
        """"
        Gives prediction.
        Emissionsscore contains batch, Sequence Length and Number of Tags
        """
        emissions = self.forward(embeddings)
        return self.crf.decode(emissions, mask=mask)

    def compute_loss(self, embeddings, tags, mask):

        emissions = self.forward(embeddings)
        loss = -self.crf(
            emissions,
            tags,
            mask=mask,
            reduction="mean"
        )
        return loss


# load preprocessed embeddings
token_documents = torch.load("data/flair_embeddings/embeddings.pt", weights_only=False,)
dataset = FairytaleSequenceTaggingDataset(token_documents, label2id=LABEL2ID)

lengths = [doc.embeddings.size(0) for doc in token_documents]

print(f"max length fairytale:{max(lengths)}")
print(f"mean length fairytale:{np.mean(lengths)}")
print(f"Embedding Shape:{token_documents[0].embeddings.shape}")
print(f"max Embedding Shape:{max(doc.embeddings.size(0) for doc in token_documents)}")

# Dataloader
train_dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn) # pin_memory only for training on GPU
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)


# Training
EMBEDDING_DIM = token_documents[0].embeddings.size(1)
# Grid Search or random search for best value; test 64, 128, 264
HIDDEN_DIM = 64

model = BiLSTM_CRF(num_tags=len(LABEL2ID), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)

# Enable GPU Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# For Debugging
batch = next(iter(train_dataloader))

for batch in train_dataloader:

    optimizer.zero_grad()

    embeddings = batch["embeddings"].to(device)
    tags = batch["tags"].to(device)
    mask = batch["mask"].to(device)

    loss = model.compute_loss(embeddings, tags, mask)
    print(loss.item())

    loss.backward()

    optimizer.step()

print(batch["embeddings"].shape)
print(batch["tags"].shape)
print(batch["mask"].shape)

emissions = model(batch["embeddings"].to(device))
print(emissions.shape)



# Evaluation
#predictions = model(input_ids, mask)
#dataset.ID2LABEL[tag]
#predicted_labels = [ID2LABEL[i] for i in prediction]
model.eval()

all_predictions = []
all_true_tags = []

with torch.no_grad():

    for batch in train_dataloader:

        embeddings = batch["embeddings"].to(device)
        tags = batch["tags"].to(device)
        mask = batch["mask"].to(device)

        predictions = model.decode(embeddings, mask)

        for pred, gold, valid in zip(predictions, tags, mask):

            length = valid.sum().item()

            all_predictions.extend(pred)
            all_true_tags.extend(gold[:length].cpu().tolist())

accuracy = accuracy_score(all_true_tags, all_predictions)

precision, recall, f1, _ = precision_recall_fscore_support(all_true_tags, all_predictions, average="macro")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1       : {f1:.4f}")
