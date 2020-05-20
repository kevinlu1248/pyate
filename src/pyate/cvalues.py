# c_value

import time
import math
from typing import List, Mapping

from tqdm import tqdm
import pandas as pd

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
def cvalues(
    technical_corpus: Corpus,
    smoothing: float = 0.01,
    verbose: bool = False,
    have_single_word: bool = False,
    technical_counts: Mapping[str, int] = None,
    threshold: float = 0,
):

    if technical_counts is None:
        technical_counts = (
            TermExtraction(technical_corpus)
            .count_terms_from_documents(verbose=verbose)
            .reindex()
        )

    order = sorted(
        list(technical_counts.keys()), key=TermExtraction.word_length, reverse=True
    )

    if not have_single_word:
        order = list(filter(lambda s: TermExtraction.word_length(s) > 1, order))

    technical_counts = technical_counts[order]

    df = pd.DataFrame(
        {
            "frequency": technical_counts.values,
            "times_nested": technical_counts.values,
            "number_of_nested": 1,
            "has_been_evaluated": False,
        },
        index=technical_counts.index,
    )

    # print(df)
    output = []
    indices = set(df.index)

    iterator = tqdm(df.iterrows()) if verbose else df.iterrows()

    for candidate, row in iterator:
        f, t, n, h = row
        length = TermExtraction.word_length(candidate)
        if length == TermExtraction.MAX_WORD_LENGTH:
            c_val = math.log(length + smoothing) * f
        else:
            c_val = math.log(length + smoothing) * f
            if h:
                c_val -= t / n
        if c_val >= threshold:
            output.append((candidate, c_val))
            for substring in helper_get_subsequences(candidate):
                if substring in indices:
                    df.loc[substring, "times_nested"] += 1
                    df.loc[substring, "number_of_nested"] += f
                    df.loc[substring, "has_been_evaluated"] = True

    srs = pd.Series(map(lambda s: s[1], output), index=map(lambda s: s[0], output))
    return srs.sort_values(ascending=False)


if __name__ == "__main__":
    corpus = "Hello, I am a term extractor."
    print(TermExtraction(corpus).cvalues(verbose=True))
