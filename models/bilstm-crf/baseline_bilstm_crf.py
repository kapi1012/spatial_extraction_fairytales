import time
from torch.utils.data import DataLoader
from torchcrf import CRF

from evaluation.evaluation import *
from dataset_preprocessing.data_split import *
from dataset_preprocessing.dataset_preprocessing import *

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


def prepare_data():
    DATA_PATH = "../../data/final_dataset_vague_spatial_expressions_29-9-26.json"
    dataset = load_dataset(DATA_PATH)
    docs, errors = annotate_dataset(dataset)

    token_documents = create_token_documents(docs)

    # load preprocessed embeddings
    EMBEDDINGS_PATH = Path("flair_embeddings/embeddings_hamburger_anzeiger_12.10.26.pt")
    if EMBEDDINGS_PATH.exists():
        sentence_documents = torch.load("flair_embeddings/embeddings_hamburger_anzeiger_12.10.26.pt", weights_only=False)
    else:
        sentence_documents = flair_embedding(token_documents)

    lengths = [doc.embeddings.size(0) for doc in sentence_documents]

    # Checking length after sentence-wise usage
    print(f"max length fairytale:{max(lengths)}")
    print(f"mean length fairytale:{np.mean(lengths)}")
    print(f"Embedding Shape:{sentence_documents[0].embeddings.shape}")
    print(f"max Embedding Shape:{max(doc.embeddings.size(0) for doc in sentence_documents)}")

    # Train-Val-Test-Split
    # choose own train-val-test-split
    ANALYSIS_PATH = "../../data/analysis_dataset/analysis_dataset.csv"
    train_khm, val_khm, test_khm = data_split_on_khm(ANALYSIS_PATH, 42)

    train_documents, val_documents, test_documents = split_documents_by_khm(sentence_documents, train_khm, val_khm,
                                                                            test_khm)

    train_dataset = FairytaleSequenceTaggingDataset(train_documents, LABEL2ID)
    val_dataset = FairytaleSequenceTaggingDataset(val_documents, LABEL2ID)
    test_dataset = FairytaleSequenceTaggingDataset(test_documents, LABEL2ID)

    # Dataloader
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True,
                                  collate_fn=collate_fn_bilstm)  # pin_memory only for training on GPU
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                                collate_fn=collate_fn_bilstm)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=0, pin_memory=True,
                                 collate_fn=collate_fn_bilstm)

    return sentence_documents, train_dataloader, val_dataloader, test_dataloader


def train_model(model, optimizer, train_data, val_data, hyperparameters, device):
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

    # Early stopping
    best_val_span_f1 = -1.0
    best_early_stopping_span_f1 = -1.0
    patience = hyperparameters["patience"]
    min_delta = hyperparameters["min_delta"]

    counter = 0
    best_epoch = 0
    best_state = None
    stopped_epoch = hyperparameters["epochs"]

    training_info = {
        "best_epoch": best_epoch,
        "best_val_span_f1": best_val_span_f1,
        "stopped_epoch": stopped_epoch,
        "early_stopped": stopped_epoch < hyperparameters["epochs"],
        **hyperparameters,
    }

    for epoch in range(hyperparameters["epochs"]):
        model.train()
        epoch_loss = 0.0
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
        true_tags, predictions, true_spans, pred_spans, prediction_documents = test_model(model=model, test_data=val_data)
        # val_f1 for training metric
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(true_tags, predictions, average='macro', zero_division=0)
        # val_span_f1 for early stopping
        # exclude O-Label for span_metrics
        entity_types = sorted({
            label[2:]
            for label in ID2LABEL.values()
            if label.startswith(("B-", "I-"))
        })
        span_precision, span_recall, val_span_f1 = span_metrics(true_spans, pred_spans, entity_types)

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
            f"Epoch {epoch + 1}/{hyperparameters['epochs']} "
            f"Loss={epoch_loss:.4f} "
            f"Token-F1={val_f1:.5f} "
            f"Span-F1={val_span_f1:.5f}"
        )

        # Always save the model with the highest observed validation Span-F1
        if val_span_f1 > best_val_span_f1:
            best_val_span_f1 = val_span_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch + 1

        # Early stopping uses min_delta separately
        if val_span_f1 > best_early_stopping_span_f1 + min_delta:
            best_early_stopping_span_f1 = val_span_f1
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            stopped_epoch = epoch + 1
            print("Early stopping.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    # save best model for later use
    torch.save(model.state_dict(), "bilstm_models/bilstm_best_1.10.26.pt")

    history["early_stopped"] = stopped_epoch < hyperparameters["epochs"]
    history["patience"] = patience
    history["min_delta"] = min_delta

    training_info["stopped_epoch"] = stopped_epoch
    training_info["early_stopped"] = stopped_epoch < hyperparameters["epochs"]
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


if __name__ == "__main__":
    sentence_documents, train_dataloader, val_dataloader, test_dataloader = prepare_data()

    # Hyperparamter
    hyperparameters = {
        "embedding_dim": sentence_documents[0].embeddings.size(1),
        "hidden_dim": 256,
        "dropout": 0.1,
        "learning_rate": 0.001,
        "batch_size": 16,
        "optimizer": "Adam",
        "epochs": 50,
        "patience": 15,
        "min_delta": 0.0005,
        "early_stopping": True,
    }

    # Model Initialization
    model = BiLSTM_CRF(num_tags=len(LABEL2ID), embedding_dim=hyperparameters["embedding_dim"],
                       hidden_dim=hyperparameters["hidden_dim"], dropout=hyperparameters["dropout"], )

    # Enable GPU Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])

    fuzzy_bilstm, history = train_model(model=model, optimizer=optimizer, train_data=train_dataloader, val_data=val_dataloader, hyperparameters=hyperparameters, device=device)
    true_tags, predictions, true_spans, pred_spans, prediction_documents = test_model(model=fuzzy_bilstm, test_data=test_dataloader)
    # exclude O-Label for span metrics
    entity_types = sorted({label[2:]for label in ID2LABEL.values() if label.startswith(("B-", "I-"))})
    span_precision, span_recall, span_f1 = span_metrics(true_spans, pred_spans, entity_types)

    save_test_report(true_tags, predictions, true_spans, pred_spans, hyperparameters)
    save_bilstm_predictions_by_khm(prediction_documents)

    # Token Level
    print(classification_report(
        true_tags,
        predictions,
        labels=list(LABEL2ID.keys()),
        target_names=list(LABEL2ID.keys()),
        zero_division=0,
    ))
    # Span-level
    print("\nSpan-Level (Exact Match)")
    print(f"Precision: {span_precision:.2f}")
    print(f"Recall:    {span_recall:.2f}")
    print(f"F1:        {span_f1:.2f}")
