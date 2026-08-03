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

# Hyperparamter

hyperparameters = {
    "embedding_dim": sentence_documents[0].embeddings.size(1),
    "hidden_dim": 256,
    "dropout": 0.3,
    "learning_rate": 0.01,
    "batch_size": 8,
    "optimizer": "Adam",
    "epochs": 50,
    "patience": 10,
    "early_stopping": True,
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

    history = {
        "epoch": [],
        "train_loss": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "best_val_f1": [],
        "epoch_train_time": [],
    }

    training_info = {
        "hidden_dim": hyperparameters["hidden_dim"],
        "dropout": hyperparameters["dropout"],
        "learning_rate": hyperparameters["learning_rate"],
        "optimizer": hyperparameters["optimizer"],
        "epochs": hyperparameters["epochs"],
        "batch_size": hyperparameters["batch_size"],
        "patience": hyperparameters["patience"],
    }

    # Early stopping
    best_val_f1 = 0.0
    patience = hyperparameters["patience"]
    counter = 0
    stopped_epoch = hyperparameters["epochs"]

    for epoch in range(hyperparameters["epochs"]):
        model.train()
        epoch_loss = 0
        start_train_time = time.time()

        for batch in train_data:

            optimizer.zero_grad()

            embeddings = batch["embeddings"].to(device)
            tags = batch["tags"].to(device)
            mask = batch["mask"].to(device)

            # compute loss and backpropagate
            loss = model.compute_loss(embeddings, tags, mask)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # mean batch loss
            epoch_loss /= len(train_data)

        # Validation
        true_tags, predictions = test_model(model=model, test_data=val_data)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(true_tags, predictions, average='micro')

        end_train_time = time.time()
        epoch_train_time = end_train_time - start_train_time

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["epoch_train_time"].append(epoch_train_time)

        print(
            f"Epoch {epoch+1}/{hyperparameters["epochs"]} "
            f"Loss={epoch_loss:.3f} "
            f"Val F1={val_f1:.2f}"
        )

        # Early Stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            counter = 0
            torch.save(model.state_dict(), "bilstm_models/bilstm_best.pt")
        else:
            counter += 1
        if counter >= patience:
            stopped_epoch = epoch + 1
            print("Early stopping.")
            break


    model.load_state_dict(torch.load("bilstm_models/bilstm_best.pt", map_location=device))

    history["early_stopped"] = stopped_epoch < hyperparameters["epochs"]
    history["patience"] = patience
    history["best_val_f1"].append(best_val_f1)

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    df_history = pd.DataFrame(history)

    for key, value in training_info.items():
        df_history[key] = value

    df_history.to_csv(f"data/train-reports/training_history_{current_time}.csv", index=False)

    end_time_tracking = time.time()
    total_time = end_time_tracking - start_time_tracking

    print(
        f"Training finished in "
        f"{total_time / 60:.2f} min "
        f"({total_time / 3600:.2f} h)."
    )

    return model, history


# Evaluation
def test_model(model, test_data):
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

    return all_true_tags, all_predictions


def save_test_report(true_tags, predictions, hyperparameters):

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    filename = f"data/test_reports/{current_time}.csv"

    report = classification_report(true_tags, predictions, target_names=LABEL2ID.keys(), output_dict=True)
    df_report = pd.DataFrame(report).T

    for key, value in hyperparameters.items():
        df_report[key] = value
    df_report = df_report.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    df_report.to_csv(filename)


fuzzy_bilstm, history = train_model(model=model, optimizer=optimizer, train_data=train_dataloader, val_data=val_dataloader, hyperparameters=hyperparameters)
true_tags, predictions = test_model(model=fuzzy_bilstm, test_data=test_dataloader)
save_test_report(true_tags, predictions, hyperparameters)

#todo f1-score on Span-level, not token-level?
print(classification_report(true_tags, predictions, target_names=LABEL2ID.keys()))