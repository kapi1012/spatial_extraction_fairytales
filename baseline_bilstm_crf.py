import time
from pathlib import Path
from torch.utils.data import DataLoader
from torchcrf import CRF

from dataset_preprocessing import *
from evaluation.evaluation import *
from data_split import *

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


DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_21-8-26_final.json"
dataset = load_dataset(DATA_PATH)
docs, errors = annotate_dataset(dataset)

token_documents = create_token_documents(docs)

# load preprocessed embeddings
EMBEDDINGS_PATH = Path("data/flair_embeddings/embeddings_21.8.26.pt")
if EMBEDDINGS_PATH.exists():
    sentence_documents = torch.load("data/flair_embeddings/embeddings_21.8.26.pt", weights_only=False)
else:
    sentence_documents = flair_embedding(token_documents)

#dataset = FairytaleSequenceTaggingDataset(sentence_documents, label2id=LABEL2ID)

lengths = [doc.embeddings.size(0) for doc in sentence_documents]

# Checking length after sentence-wise usage
print(f"max length fairytale:{max(lengths)}")
print(f"mean length fairytale:{np.mean(lengths)}")
print(f"Embedding Shape:{sentence_documents[0].embeddings.shape}")
print(f"max Embedding Shape:{max(doc.embeddings.size(0) for doc in sentence_documents)}")

# Train-Val-Test-Split
# Small Dataset, so 80 % train
#train_dataset, val_dataset, test_dataset = random_split(dataset, [0.8, 0.1, 0.1])

# choose own train-val-test-split
ANALYSIS_PATH = "data/analysis_dataset/analysis_dataset.csv"
train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, 42)

train_documents, val_documents, test_documents = split_documents_by_khm(sentence_documents,train_khm, val_khm, test_khm)

train_dataset = FairytaleSequenceTaggingDataset(train_documents, LABEL2ID)
val_dataset = FairytaleSequenceTaggingDataset(val_documents, LABEL2ID)
test_dataset = FairytaleSequenceTaggingDataset(test_documents, LABEL2ID)

# Dataloader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn_bilstm) # pin_memory only for training on GPU
val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn_bilstm)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn_bilstm)

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
        "val_span_precision": [],
        "val_span_recall": [],
        "val_span_f1": [],
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
    best_val_span_f1 = 0.0
    patience = hyperparameters["patience"]
    min_delta = 0.001

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
        true_tags, predictions, true_spans, pred_spans = test_model(model=model, test_data=val_data)
        # val_f1 for training metric
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(true_tags, predictions, average='macro')
        # val_span_f1 for early stopping
        span_precision, span_recall, val_span_f1 = span_metrics(true_spans, pred_spans)

        end_train_time = time.time()
        epoch_train_time = end_train_time - start_train_time

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(epoch_loss)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_f1"].append(val_f1)
        history["val_span_precision"].append(span_precision)
        history["val_span_recall"].append(span_recall)
        history["val_span_f1"].append(val_span_f1)
        history["epoch_train_time"].append(epoch_train_time)

        print(
            f"Epoch {epoch+1}/{hyperparameters["epochs"]} "
            f"Loss={epoch_loss:.4f} "
            f"Token-F1={val_f1:.5f} "
            f"Span-F1={val_span_f1:.5f}"
        )

        # Early Stopping with span-level f1
        if val_span_f1 > best_val_span_f1 + min_delta:
            best_val_span_f1 = val_span_f1
            best_epoch = epoch + 1
            counter = 0
            torch.save(model.state_dict(), "bilstm_models/bilstm_best_2.9.26.pt")
        else:
            counter += 1
        if counter >= patience:
            stopped_epoch = epoch + 1
            print("Early stopping.")
            break


    model.load_state_dict(torch.load("bilstm_models/bilstm_best_2.9.26.pt", map_location=device))

    history["early_stopped"] = stopped_epoch < hyperparameters["epochs"]
    history["patience"] = patience

    training_info["best_epoch"] = best_epoch
    training_info["best_val_span_f1"] = best_val_span_f1

    current_time = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    df_history = pd.DataFrame(history)

    for key, value in training_info.items():
        df_history[key] = value
    df_history = df_history.map(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

    df_history.to_csv(f"data/train-reports/training_history_{current_time}.csv", index=False)

    end_time_tracking = time.time()
    total_time = end_time_tracking - start_time_tracking

    training_summary = {
        "best_epoch": best_epoch,
        "best_val_span_f1": best_val_span_f1,
        "stopped_epoch": stopped_epoch,
        **hyperparameters
    }

    pd.DataFrame([training_summary]).to_csv(f"data/train-reports/summary_{current_time}.csv", index=False)

    print(
        f"Training finished in "
        f"{total_time / 60:.2f} min "
        f"({total_time / 3600:.2f} h)."
    )

    return model, history


fuzzy_bilstm, history = train_model(model=model, optimizer=optimizer, train_data=train_dataloader, val_data=val_dataloader, hyperparameters=hyperparameters)
true_tags, predictions, true_spans, pred_spans = test_model(model=fuzzy_bilstm, test_data=test_dataloader)
span_precision, span_recall, span_f1 = span_metrics(true_spans, pred_spans)
save_test_report(true_tags, predictions, true_spans, pred_spans, hyperparameters)

# Token Level
print(classification_report(true_tags, predictions, target_names=LABEL2ID.keys()))
# Span-level
print("\nSpan-Level (Exact Match)")
print(f"Precision: {span_precision:.2f}")
print(f"Recall:    {span_recall:.2f}")
print(f"F1:        {span_f1:.2f}")
