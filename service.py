from __future__ import annotations

import asyncio
import os
import json
from datetime import datetime

from pydantic import BaseModel

import numpy as np
from sentence_transformers import SentenceTransformer

import bentoml
from bentoml.io import JSON, NumpyNdarray


def find_top_k_indices(similarity_matrix_row, k):
    return np.argsort(similarity_matrix_row)[-k:][::-1]


class M3eRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        import torch
        self.model = SentenceTransformer('moka-ai/m3e-base')

        if torch.cuda.is_available():
            self.model.cuda()
        else:
            self.model.cpu()

    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def embeddings(self, sentences):
        embeddings = self.model.encode(sentences, normalize_embeddings=True)
        return embeddings

m3e_runner = bentoml.Runner(M3eRunnable, embedded=False)

svc = bentoml.Service("m3e-embedding-svc", runners=[m3e_runner])

@svc.api(input=JSON(), output=NumpyNdarray())
def embeddings(json_dict):
    sentences = json_dict["data"]
    if not isinstance(sentences, list):
        sentences = [sentences]
    embeddings = m3e_runner.embeddings.run(sentences)
    return embeddings


class RankArgs(BaseModel):
    queries: str | list[str]
    passages: str | list[str]
    topk: int

sample_rank_input = RankArgs(
    queries=['你今年几岁了啊？', '小明来自哪里？'],
    passages=['小明如今23岁了', '小明的年龄没人知道', '小明还是一个小孩子', '小红如今23岁了', '小明可不笨！', '小明很聪明的！', '小明是可靠的伙伴！', '小明喜欢的食物是甜甜圈！', '小红多大了', '小红出生在一个人鱼国家', '小红的家乡在哪里', '小红16岁的时候，离家出走了', "小小明", "小明你好"],
    topk=3,
)

@svc.api(input=JSON.from_sample(sample=sample_rank_input), output=JSON())
async def ranks(q: RankArgs):
    queries = q.queries
    if not isinstance(queries, list):
        queries = [queries]

    passages = q.passages
    if not isinstance(passages, list):
        queries = [passages]

    topk = q.topk

    result = await asyncio.gather(
        m3e_runner.embeddings.async_run(queries),
        m3e_runner.embeddings.async_run(passages),
    )

    q_embeddings, p_embeddings = result

    similarity_matrix = q_embeddings @ p_embeddings.T

    topk_results = {}

    for i, query in enumerate(queries):
        topk_indices = find_top_k_indices(similarity_matrix[i], topk)
        topk_passages = [(index, similarity_matrix[i, index]) for index in topk_indices]
        topk_results[query] = topk_passages

    final_res = []
    for query, passage_scores in topk_results.items():
        res = []
        for index, score in passage_scores:
            d = dict(
                query=query,
                passage=passages[index],
                score=score,
            )
            res.append(d)

        final_res.append(res)

    return final_res
