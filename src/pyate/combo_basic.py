# combo basic
import math
from typing import List, Mapping, Sequence

import pandas as pd
import numpy as np
from tqdm import tqdm

from .term_extraction import TermExtraction, add_term_extraction_method, Corpus


def helper_get_subsequences(s: str) -> List[str]:
    sequence = s.split()
    if len(sequence) <= 2:
        return []
    answer = []
    for left in range(len(sequence) + 1):
        for right in range(left + 1, len(sequence) + 1):
            if left == 0 and right == len(sequence):
                continue
            answer.append(" ".join(sequence[left:right]))
    return answer


@add_term_extraction_method
def combo_basic(
    technical_corpus: Corpus,
    smoothing: float=0.01,
    verbose: bool=False,
    have_single_word: bool=False,
    technical_counts: Mapping[str, int]=None,
    weights: Sequence[float]=None,
):

    if technical_counts is None:
        technical_counts = (
            TermExtraction(technical_corpus)
            .count_terms_from_documents(verbose=verbose)
            .reindex()
        )

    if len(technical_counts) == 0:
        return pd.Series()

    order = sorted(
        list(technical_counts.keys()), key=TermExtraction.word_length, reverse=True
    )

    if not have_single_word:
        order = list(filter(lambda s: TermExtraction.word_length(s) > 1, order))

    technical_counts = technical_counts[order]

    df = pd.DataFrame(
        {
            "xlogx_score": technical_counts.reset_index()
            .apply(
                lambda s: math.log(TermExtraction.word_length(s["index"])) * s[0],
                axis=1,
            )
            .values,
            "times_subset": 0,
            "times_superset": 0,
        },
        index=technical_counts.index,
    )

    indices = set(technical_counts.index)
    iterator = tqdm(technical_counts.index) if verbose else technical_counts.index

    for index in iterator:
        for substring in helper_get_subsequences(index):
            if substring in indices:
                df.at[substring, "times_subset"] += 1
                df.at[index, "times_superset"] += 1

    if weights is None:
        weights = np.array([1, 0.75, 0.1])
    return df.apply(lambda s: s.values.dot(weights), axis=1)


if __name__ == "__main__":
    corpus = "Hello I am a term extractor."
    print(TermExtraction(corpus).combo_basic().sort_values(ascending=False).head(50))
