import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
import time

from torch.utils.data import DataLoader, random_split
from torchcrf import CRF

from constants import LABEL2ID
from dataset_preprocessing import collate_fn, FairytaleSequenceTaggingDataset

from sklearn.metrics import precision_recall_fscore_support, classification_report
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
sentence_documents = torch.load("data/flair_embeddings/embeddings.pt", weights_only=False, )
dataset = FairytaleSequenceTaggingDataset(sentence_documents, label2id=LABEL2ID)

lengths = [doc.embeddings.size(0) for doc in sentence_documents]

# Checking length after sentence-wise usage
print(f"max length fairytale:{max(lengths)}")
print(f"mean length fairytale:{np.mean(lengths)}")
print(f"Embedding Shape:{sentence_documents[0].embeddings.shape}")
print(f"max Embedding Shape:{max(doc.embeddings.size(0) for doc in sentence_documents)}")

# Train-Val-Test-Split
# Small Dataset, so 80 % train
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

# Dataloader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn) # pin_memory only for training on GPU
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

# Training
EMBEDDING_DIM = sentence_documents[0].embeddings.size(1)
EPOCHS = 50

# Hyperparamtersearch
# Grid Search or random search for best value; test 64, 128, 264
HIDDEN_DIM = 64
DROPOUT = 0.5
LEARNING_RATE = 0.001

hyperparameters = {
    "embedding_dim": EMBEDDING_DIM,
    "hidden_dim": HIDDEN_DIM,
    "dropout": 0.5,
    "learning_rate": 0.001,
    "batch_size": 8,
    "optimizer": "Adam",
    "epochs": EPOCHS,
}

# Model Initialization
model = BiLSTM_CRF(num_tags=len(LABEL2ID), embedding_dim=hyperparameters["embedding_dim"], hidden_dim=hyperparameters["hidden_dim"], dropout=hyperparameters["dropout"],)

# Enable GPU Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

def train_model(model, optimizer, train_data, val_data, hyperparameters):
    start_time_tracking = time.time()

    best_f1 = 0.0

    for epoch in range(hyperparameters["epochs"]):
        model.train()
        epoch_loss = 0

        for batch in train_data:

            optimizer.zero_grad()

            embeddings = batch["embeddings"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            # compute loss and backpropagate
            loss = model.compute_loss(embeddings, tags, mask)
            #print(loss.item())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Validation
        true_tags, predictions = test_model(model=model, test_data=val_data, hyperparameters=hyperparameters)
        _, _, val_f1, _ = precision_recall_fscore_support(true_tags, predictions, average='micro')

        print(
            f"Epoch {epoch+1}/{hyperparameters["epochs"]} "
            f"Loss={epoch_loss:.3f} "
            f"Val F1={val_f1:.4f}"
        )

        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "bilstm_models/bilstm_best.pt")

    model.load_state_dict(torch.load("bilstm_models/bilstm_best.pt"))

    end_time_tracking = time.time()
    print(f"Total time training & testing: {(end_time_tracking - start_time_tracking) / 60} minutes or {(end_time_tracking - start_time_tracking) / 3600} hours.")

    return model


# Evaluation
def test_model(model, test_data, hyperparameters):
    model.eval()

    all_predictions = []
    all_true_tags = []

    with torch.no_grad():

        for batch in test_data:
            embeddings = batch["embeddings"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            predictions = model.decode(embeddings, mask)

            for pred, gold, valid in zip(predictions, tags, mask):

                length = valid.sum().item()

                all_predictions.extend(pred)
                all_true_tags.extend(gold[:length].cpu().tolist())

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    report = classification_report(all_true_tags, all_predictions, target_names=LABEL2ID.keys(), output_dict=True)
    df_report = pd.DataFrame(report).T
    for key, value in hyperparameters.items():
        df_report[key] = value
    df_report = df_report.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    pd.DataFrame(df_report).to_csv("data/test_reports/" + current_time + ".csv")

    return all_true_tags, all_predictions


fuzzy_bilstm = train_model(model=model, optimizer=optimizer, train_data=train_dataloader, val_data=val_dataloader, hyperparameters=hyperparameters)
true_tags, predictions = test_model(model=fuzzy_bilstm, test_data=test_dataloader, hyperparameters=hyperparameters)

#todo f1-score on Span-level, not token-level?
print(classification_report(true_tags, predictions, target_names=LABEL2ID.keys()))