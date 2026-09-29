import itertools
import numpy as np

from dataset_preprocessing import *
from constants import *
from collections import defaultdict


""""
Calculate Inter-Annotator Agreement with Krippendorff's Alpha.

- add label-studio ID to Dataclass
    -> khm is not unique anymore
- Extract 25 fairytales with annotators
- over all: 7 different annotators
- Most have 1 annotators, some up to 3
- Add my annotations
- each fairytale has min 1 annotator, max 4 annotators
    -> Krippendorff can handle Null/NaN values

- Calculate for all 25 fairytales
- Calculate on token-level
- 1 item = 1 token
- labels (see constants):
    "O": 0,
    "B-FUZZY_LOC": 1,
    "I-FUZZY_LOC": 2,
    "B-METAPHORIC_LOC": 3,
    "I-METAPHORIC_LOC": 4,
- -> nominal calculation

- Form (per fairytale, number of annotators vary):
  Token (=item)     Annotator1      Annotator2          Annotator3
├───────────────├─────────────────├─────────────────├─────────────────
    1                   0               0               0
    2                   0               1               1
    3                   1               2               0   

for I.
- input as np.array: [ [0 0 0], [0 1 1], [1 2 0] ]

for III.
- input as sequence of 3-tuple (coder,item,label)
    - [(annotator1, 1, 0), (annotator2, 1, 0), (annotator3, 1, 0),
     (annotator1, 2, 0), (annotator2, 2, 1), (annotator3, 2, 1),
     (annotator1, 3, 1), (annotator2, 3, 2), (annotator3, 3, 0)]

- Krippendorff implementations:
    I. by hand: https://mbrenndoerfer.com/writing/inter-annotator-agreement-kappa-alpha-reliability
    II. python lib: https://pypi.org/project/krippendorff/
    III. nlkt: https://www.nltk.org/api/nltk.metrics.agreement.html (see: https://stackoverflow.com/questions/45353866/nltk-inter-annotator-agreement-using-krippendorff-alpha)

- Implementation not on fairytale-level, but for all items (tokens) in same matrix
- -> up to 7 annotator columns (-> Krippendorff ignores NaNs)
"""""

# khm numbers of annotated fairytales
khm_annotators = [1, 10, 11, 12, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121]

# load all data as dataset
main_annotator_dataset = load_dataset("../data/grimm_fairytales_annotated/fuzzy_loc_21-8-26_final.json")
inter_annotator_dataset = load_dataset("../data/annotators_annotations/fuzzy_loc_IAA_8-9-26_mini.json")

# filter for only annotated fairytales by khm number
main_annotator_list = [fairytale for fairytale in main_annotator_dataset.fairytales if fairytale.khm in khm_annotators]
inter_annotator_list = [fairytale for fairytale in inter_annotator_dataset.fairytales if fairytale.khm in khm_annotators]

# create 2 datasets: labels have different names, but refer to the same
# due to updated annotator guidelines
main_docs, main_errors = annotate_dataset(FairytaleDataset(main_annotator_list))
iaa_docs, iaa_errors = annotate_dataset(FairytaleDataset(inter_annotator_list))

main_token_documents = create_token_documents(main_docs)
iaa_token_documents = create_token_documents(iaa_docs)

# convert labels to numbers for Krippendorff

main_token_data = []

for doc in main_token_documents:
    for token_idx, (token, label) in enumerate(
        zip(doc.tokens, doc.labels)
    ):
        main_token_data.append({
            "khm": doc.khm,
            "id": doc.id, # not unique per annotator
            "annotator_id": doc.annotator_id, # unique annotator ID
            "token_idx": token_idx,
            "token": token,
            "label": LABEL2ID[label],
        })

iaa_token_data = []

for doc in iaa_token_documents:
    for token_idx, (token, label) in enumerate(
        zip(doc.tokens, doc.labels)
    ):
        iaa_token_data.append({
            "khm": doc.khm,
            "id": doc.id,
            "annotator_id": doc.annotator_id,
            "token_idx": token_idx,
            "token": token,
            "label": LABEL2ID_IAA[label],
        })

#combine both lists
all_token_data = main_token_data + iaa_token_data
#print(all_token_data)

# count all token_idx in all data (unique per khm, every count starts again at 0 for each fairytale)
unique_tokens = {
    (item["khm"], item["token_idx"])
    for item in all_token_data
}

#print(len(unique_tokens))

# for every missing entry -> NaN
# up to max 7 annotators
MAX_ANNOTATORS = 7

# count token length per fairytale to use as row index
fairytale_length = defaultdict(lambda: 0 )
for khm, token_idx in unique_tokens:
    fairytale_length[khm] = max(fairytale_length[khm], token_idx + 1)  # token_idx is 0 based

# to collect each fairytale matrix
matrizes = {}

for entry in all_token_data: # -> rows
    if entry["khm"] not in matrizes:
        # initialize matrix with NaN values
        matrizes[entry["khm"]] = np.full((fairytale_length[entry["khm"]], MAX_ANNOTATORS), np.nan)
    # fill in annotations
    matrizes[entry["khm"]][entry["token_idx"], ANNOTATOR2IDX[entry["annotator_id"]]] = entry["label"]

# combine all matrices (one per fairytale) to one
annotations_iaa_matrix = np.vstack(tuple(matrizes.values()))


# Krippendorff's alpha
# implementation after: https://mbrenndoerfer.com/writing/inter-annotator-agreement-kappa-alpha-reliability

# expected: 5 labels: 0, 1, 2, 3, 4
categories = np.unique(list(LABEL2ID.values()))

def krippendorff_alpha(annotations, categories):
    """Alpha with missing data handling."""
    n_items, n_raters = annotations.shape
    k = len(categories)

    coincidence = np.zeros((k, k))

    for i in range(n_items):
        # Filter out NaN values
        valid = annotations[i, ~np.isnan(annotations[i, :])]
        n_valid = len(valid)

        if n_valid < 2:
            continue

        for r1, r2 in itertools.product(range(n_valid), range(n_valid)):
            if r1 != r2:
                c1 = np.where(categories == valid[r1])[0].item()
                c2 = np.where(categories == valid[r2])[0].item()
                coincidence[c1, c2] += 1

    total = np.sum(coincidence)
    # observed disagreement
    D_o = (total - np.trace(coincidence)) / total

    probs = np.sum(coincidence, axis=1) / total
    # expected disagreement
    D_e = 1 - np.sum(probs**2)

    return 1 - (D_o / D_e)


alpha = krippendorff_alpha(annotations_iaa_matrix, categories)
print(f"{alpha:.2f}")



























