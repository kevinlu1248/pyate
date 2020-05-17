# c_value

import time
import math
from tqdm import tqdm
import pandas as pd
from term_extraction import TermExtraction, add_term_extraction_method

start_ = 0
tmp = 0
# TOTAL_WORK = 27768
# success = 27768
# pbar = tqdm(total=27768)


def start():
    global start_
    start_ = time.time()


def end():
    global start_
    print(time.time() - start_)


MAX_WORD_LENGTH = 6
THRESHOLD = 0


def helper_get_subsequences(s):
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
def c_values(
    technical_corpus,
    smoothing=0.01,
    verbose=False,
    have_single_word=False,
    technical_counts=None,
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
        if length == MAX_WORD_LENGTH:
            c_val = math.log(length + smoothing) * f
        else:
            c_val = math.log(length + smoothing) * f
            if h:
                c_val -= t / n
        if c_val >= THRESHOLD:
            output.append((candidate, c_val))
            nstart = time.time()  # TODO: optimize
            for substring in helper_get_subsequences(candidate):
                if substring in indices:
                    df.loc[substring, "times_nested"] += 1
                    df.loc[substring, "number_of_nested"] += f
                    df.loc[substring, "has_been_evaluated"] = True
            global tmp
            tmp += time.time() - nstart

    srs = pd.Series(map(lambda s: s[1], output), index=map(lambda s: s[0], output))
    return srs.sort_values(ascending=False)


if __name__ == "__main__":
    import pickle

    pkl = pickle.load(open("../data/pmc_testing.pkl", "rb"))
    corpus = pkl
    print(list(TermExtraction(pkl[0]).c_values(verbose=True).index))
    print(pkl[0])
