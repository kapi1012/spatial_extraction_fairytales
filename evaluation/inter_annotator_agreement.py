import itertools
import numpy as np

from dataset_preprocessing import *


""""
Calculate Inter-Annotator Agreement with Krippendorff's Alpha.

- add label-studio ID to Dataclass
    -> khm is not unique anymore
- Extract 25 fairytales with annotators
- Most have 1 annotators, some up to 3
- Add my annotations
- each fairytale has min 1 annotator, max 4 annotators
    -> Krippendorff can handle Null/NaN values

- Calculate per Fairytale? or for all 25 fairytales?
- Calculate on (sub-) word-level or token-level?
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

"""""

# khm numbers of annotated fairytales
khm_annotators = [1, 10, 11, 12, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 114, 115, 116, 117, 118, 119, 120, 121]

# load all data as dataset
main_annotator_dataset = load_dataset("../data/grimm_fairytales_annotated/fuzzy_loc_21-8-26_final.json")
inter_annotator_dataset = load_dataset("../data/annotators_annotations/fuzzy_loc_IAA_8-9-26_mini.json")

# filter for only annotated fairytales by khm number
main_annotator_list = [fairytale for fairytale in main_annotator_dataset.fairytales if fairytale.khm in khm_annotators]
inter_annotator_list = [fairytale for fairytale in inter_annotator_dataset.fairytales if fairytale.khm in khm_annotators]

# combine both lists and convert back to dataset
inter_annotator = main_annotator_list + inter_annotator_list
inter_annotator_ds = FairytaleDataset(inter_annotator)

# dataset annotieren
docs, errors = annotate_dataset(inter_annotator_ds)
# token docs erstellen
token_documents = create_token_documents(docs)

# testing
for token, label in zip(
    token_documents[0].tokens,
    token_documents[0].labels,
):
    print(f"{token:20} {label:10}")

# add khm and id
# convert label to number


# build np array:
# for each KHM iterate over all tokens, take each label from every unique ID
# for every missing entry -> NaN
# should have: a lot of token items, up to 4 annotator entries: [ [0 1 1 2 NaN], [1 2 2 0 NaN], ... ]


# Krippendorff's alpha
# implementation after: https://mbrenndoerfer.com/writing/inter-annotator-agreement-kappa-alpha-reliability

# Define our annotation data: 10 items, 3 annotators (A, B, C)
# Categories: 0=Negative(G), 1=Neutral(N), 2=Positive(P)
annotations = np.array(
    [
        [2, 2, 2],  # Item 1: P,P,P
        [2, 2, 1],  # Item 2: P,P,N
        [1, 1, 1],  # Item 3: N,N,N
        [2, 2, 2],  # Item 4: P,P,P
        [1, 2, 1],  # Item 5: N,P,N
        [0, 0, 0],  # Item 6: G,G,G
        [2, 1, 2],  # Item 7: P,N,P
        [1, 1, 1],  # Item 8: N,N,N
        [2, 2, 2],  # Item 9: P,P,P
        [0, 2, 0],  # Item 10: G,P,G
    ]
)

n_items, n_raters = annotations.shape
categories = np.unique(annotations)
# expected: 5 labels: 0, 1, 2, 3, 4
k = len(categories)

def krippendorff_alpha(annotations, categories, metric="nominal"):
    """
    Calculate Krippendorff's alpha.
    Handles varying numbers of raters per item (missing data via NaN).
    """
    n_items, n_raters = annotations.shape
    k = len(categories)

    # Build coincidence matrix
    coincidence = np.zeros((k, k))

    for i in range(n_items):
        # Get valid annotations for this item (non-NaN if we had missing data)
        valid = annotations[i, :]
        n_valid = len(valid)

        # All ordered pairs of raters
        for r1, r2 in itertools.product(range(n_valid), range(n_valid)):
            if r1 != r2:
                c1 = np.where(categories == valid[r1])[0][0]
                c2 = np.where(categories == valid[r2])[0][0]
                coincidence[c1, c2] += 1

    total_coincidences = np.sum(coincidence)

    # Observed disagreement D_o (for nominal: proportion of off-diagonal)
    if metric == "nominal":
        D_o = (total_coincidences - np.trace(coincidence)) / total_coincidences
    else:
        raise NotImplementedError("Only nominal metric implemented here")

    # Expected disagreement D_e based on marginals
    category_counts = np.sum(coincidence, axis=1)
    n_total = np.sum(category_counts)
    probs = category_counts / n_total

    D_e = 1 - np.sum(probs**2)

    alpha = 1 - (D_o / D_e) if D_e != 0 else 0

    return alpha, D_o, D_e, coincidence


alpha, d_o, d_e, coin_mat = krippendorff_alpha(annotations, categories)
print(alpha)



























