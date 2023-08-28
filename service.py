from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from datetime import datetime

import bentoml
from bentoml.io import JSON, NumpyNdarray

#model = SentenceTransformer(model_name)
model = SentenceTransformer('moka-ai/m3e-base', device='cuda:0')

def find_top_k_indices(similarity_matrix_row, k):
    return np.argsort(similarity_matrix_row)[-k:][::-1]

def get_m3e_embeddings(sentences: list | str):
    if not isinstance(sentences, list):
        sentences = [sentences]

    print(sentences)
    embeddings = model.encode(sentences, normalize_embeddings=True)

    print(type(embeddings))
    print(embeddings)
    return embeddings

svc = bentoml.Service("m3e-embedding-svc", runners=[])

@svc.api(input=JSON(), output=NumpyNdarray())
def embeddings(json_dict):
    sentences = json_dict["data"]
    embeddings = get_m3e_embeddings(sentences)
    return embeddings

