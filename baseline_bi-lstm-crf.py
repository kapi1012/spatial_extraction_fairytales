import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torchcrf import CRF

from constants import LABEL2ID, ID2LABEL
from dataset_preprocessing import collate_fn, load_dataset, annotate_dataset, create_token_documents, build_vocabulary, FairytaleSequenceTaggingDataset

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
    def __init__(self, vocab_size, num_tags, embedding_dim, hidden_dim, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # Dropout for Regularization
        self.hidden2tag = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)


    def forward(self, input_ids):
        """
        Calculate Modeloutput
        """
        embeddings = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeddings)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)

        return emissions

    def decode(self, input_ids, mask):
        """"
        Gives prediction.
        Emissionsscore contains batch, Sequence Length and Number of Tags
        """
        emissions = self.forward(input_ids)
        return self.crf.decode(emissions, mask=mask)

    def compute_loss(self, input_ids, tags, mask):

        emissions = self.forward(input_ids)
        loss = -self.crf(
            emissions,
            tags,
            mask=mask,
            reduction="mean"
        )
        return loss



# Dataset
DATA_PATH = "data/grimm_fairytales_annotated/fuzzy_loc_3.7.26.json"
dataset = load_dataset(DATA_PATH)
docs, errors = annotate_dataset(dataset)

token_documents = create_token_documents(docs)
word2id, id2word = build_vocabulary(token_documents)

dataset = FairytaleSequenceTaggingDataset(token_documents, word2id, label2id=LABEL2ID)


# Dataloader
train_dataloader = DataLoader(dataset=dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn) # pin_memory only for training on GPU
#test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)


# Training
# Test 50, 100, 300
EMBEDDING_DIM = 50
# Grid Search or random search for best value; test 64, 128, 264
HIDDEN_DIM = 64

model = BiLSTM_CRF(vocab_size=len(word2id), num_tags=len(LABEL2ID), embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM)

# Enable GPU Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# For Debugging
batch = next(iter(train_dataloader))

for batch in train_dataloader:

    optimizer.zero_grad()

    input_ids = batch["input_ids"].to(device)
    tags = batch["tags"].to(device)
    mask = batch["mask"].to(device)

    loss = model.compute_loss(input_ids, tags, mask)
    print(loss.item())

    loss.backward()

    optimizer.step()

print(batch["input_ids"].shape)
print(batch["tags"].shape)
print(batch["mask"].shape)

emissions = model(batch["input_ids"].to(device))
print(emissions.shape)



# Evaluation
#predictions = model(input_ids, mask)
#dataset.ID2LABEL[tag]
#predicted_labels = [ID2LABEL[i] for i in prediction]