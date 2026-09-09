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
main_annotator_dataset = load_dataset("data/grimm_fairytales_annotated/fuzzy_loc_21-8-26_final.json")
inter_annotator_dataset = load_dataset("data/annotators_annotations/fuzzy_loc_IAA_8-9-26_mini.json")

# filter for only annotated fairytales by khm number
main_annotator_list = [fairytale for fairytale in main_annotator_dataset.fairytales if fairytale.khm in khm_annotators]
inter_annotator_list = [fairytale for fairytale in inter_annotator_dataset.fairytales if fairytale.khm in khm_annotators]

# combine both lists and convert back to dataset
inter_annotator = main_annotator_list + inter_annotator_list
inter_annotator_ds = FairytaleDataset(inter_annotator)
































