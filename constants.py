LABEL2ID = {
    "O": 0,
    "B-FUZZY_LOC": 1,
    "I-FUZZY_LOC": 2,
    "B-METAPHORIC_LOC": 3,
    "I-METAPHORIC_LOC": 4,
}

ID2LABEL = {v: k for k, v in LABEL2ID.items()}

LABEL2ID_IAA = {
    "O": 0,
    "B-VAGUE_LOC": 1,
    "I-VAGUE_LOC": 2,
    "B-CONCEPTUAL_LOC": 3,
    "I-CONCEPTUAL_LOC": 4,
}

ID2LABEL_IAA = {v: k for k, v in LABEL2ID_IAA.items()}