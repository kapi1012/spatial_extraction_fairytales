import baseline_bilstm_crf as base
import torch
from constants import LABEL2ID
from sklearn.metrics import precision_recall_fscore_support

def run_grid_search(sentence_documents, train_dataloader, val_dataloader, device):
    param_grid = {
        "dropout" : [0.1, 0.3, 0.5],
        "learning_rate" : [1e-2, 1e-3, 1e-4],
        "hidden_dim" : [64, 128, 256],
    }

    best_f1 = 0.0
    best_params = None

    for hidden_dim in param_grid["hidden_dim"]:
        for dropout in param_grid["dropout"]:
            for lr in param_grid["learning_rate"]:
                print(f"\nTesting: hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}")

                hyperparameters = {
                    "embedding_dim": sentence_documents[0].embeddings.size(1),
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "learning_rate": lr,
                    "batch_size": 8,
                    "optimizer": "Adam",
                    "epochs": 50,
                    "patience": 10,
                    "early_stopping": True,
                }

                model = base.BiLSTM_CRF(
                    num_tags=len(LABEL2ID),
                    embedding_dim=hyperparameters["embedding_dim"],
                    hidden_dim=hyperparameters["hidden_dim"],
                    dropout=hyperparameters["dropout"],
                )

                model.to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=hyperparameters["learning_rate"],
                )

                model, history = base.train_model(
                    model,
                    optimizer,
                    train_dataloader,
                    val_dataloader,
                    hyperparameters,
                    device=device,
                )

                # Train and evaluate
                val_true, val_pred, _, _ = base.test_model(model, val_dataloader)

                _, _, val_f1, _ = precision_recall_fscore_support(val_true, val_pred, average="macro", zero_division=0)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_params = {"hidden_dim": hidden_dim, "dropout": dropout, "learning_rate": lr}
                    torch.save(model.state_dict(), f"bilstm_models/{hidden_dim}-{dropout}-{lr}.pt")
    return best_params, best_f1


if __name__ == "__main__":
    sentence_documents, train_dataloader, val_dataloader, test_dataloader = base.prepare_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_params, best_f1 = run_grid_search(sentence_documents, train_dataloader, val_dataloader, device)

    print(f"\nBest hyperparameters: {best_params}")
    print(f"Best F1 score: {best_f1:.4f}")
# Best hyperparameters: {'hidden_dim': 256, 'dropout': 0.3, 'learning_rate': 0.01}
# Best F1 score: 0.8213

# with new split & finished dataset
#Best hyperparameters: {'hidden_dim': 128, 'dropout': 0.2, 'learning_rate': 0.0001}
#Best F1 score: 0.8333

#final dataset after iaa
#Best hyperparameters: {'hidden_dim': 256, 'dropout': 0.1, 'learning_rate': 0.001}
#Best F1 score: 0.8207