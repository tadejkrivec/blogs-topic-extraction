import json
from itertools import islice

import numpy as np
import pandas as pd

from .nlp import calculate_embeddings


def batcher(iterable, batch_size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, batch_size)):
        yield batch

def import_data(file, precalculated_file=None):
    try:
        data = pd.read_csv(precalculated_file)
        embeddings = [eval(x) for x in data.embeddings]
    except FileNotFoundError:
        with open(file, "r") as f:
            data = json.load(f)

        pars = [(x["paragraphs"], x["title"]) for x in data["data"]]
        qas = [(z["qas"], y) for x, y in pars for z in x]
        questions = [(z["question"], y) for x, y in qas for z in x]

        res = []
        for batch in batcher(questions, 100):
            embeddings = calculate_embeddings([x[0] for x in batch])
            res.extend([(*b, e) for b, e in zip(batch, embeddings)])

        questions = np.array([x[0] for x in res])
        titles = np.array([x[1] for x in res])
        embeddings = np.array([x[2] for x in res])
                
        data = pd.DataFrame()
        data["embeddings"] = [str(list(x)) for x in embeddings]
        data["questions"] = questions
        data["title"] = titles
        data.to_csv(precalculated_file)
    return data, embeddings
