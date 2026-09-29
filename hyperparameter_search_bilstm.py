from baseline_bilstm_crf import *
from constants import LABEL2ID
from sklearn.metrics import precision_recall_fscore_support


param_grid = {
    "dropout" : [0.1, 0.2, 0.3, 0.5],
    "learning_rate" : [1e-2, 1e-3, 1e-4, 1e-5],
    "hidden_dim" : [64, 128, 256, 512],
}

best_f1 = 0.0
best_params = None

for hidden_dim in param_grid["hidden_dim"]:
    for dropout in param_grid["dropout"]:
        for lr in param_grid["learning_rate"]:
            print(f"\nTesting: hidden_dim={hidden_dim}, dropout={dropout}, lr={lr}")

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

            model = BiLSTM_CRF(
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

            model, history = train_model(
                model,
                optimizer,
                train_dataloader,
                val_dataloader,
                hyperparameters,
            )

            # Train and evaluate
            val_true, val_pred, _, _ = test_model(model, val_dataloader)

            _, _, val_f1, _ = precision_recall_fscore_support(val_true, val_pred, average="macro")

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_params = {"hidden_dim": hidden_dim, "dropout": dropout, "learning_rate": lr}
                torch.save(model.state_dict(), f"bilstm_models/{hidden_dim}-{dropout}-{lr}.pt")

print(f"\nBest hyperparameters: {best_params}")
print(f"Best F1 score: {best_f1:.4f}")
# Best hyperparameters: {'hidden_dim': 256, 'dropout': 0.3, 'learning_rate': 0.01}
# Best F1 score: 0.8213

# with new split & finished dataset
#Best hyperparameters: {'hidden_dim': 128, 'dropout': 0.2, 'learning_rate': 0.0001}
#Best F1 score: 0.8333