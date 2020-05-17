# c_value

import spacy
import pickle
import time
import math
import json
import pandas as pd
from .term_extraction import TermExtraction, add_term_extraction_method


@add_term_extraction_method
def weirdness(
    technical_corpus,
    general_corpus=None,
    general_corpus_size=TermExtraction.DEFAULT_GENERAL_DOMAIN_SIZE,
    normalized=False,
    technical_counts=None,
    verbose=False,
):
    # http://ceur-ws.org/Vol-1031/paper3.pdf

    if general_corpus is None:
        general_corpus = TermExtraction.DEFAULT_GENERAL_DOMAIN[:general_corpus_size]

    if technical_counts is None:
        technical_counts = TermExtraction(technical_corpus).count_terms_from_documents(
            verbose=verbose
        )
    general_counts = TermExtraction(
        general_corpus, technical_counts.index
    ).count_terms_from_documents()
    technical_word_count = TermExtraction.word_length("\n".join(technical_corpus))
    general_word_count = TermExtraction.word_length("\n".join(general_corpus))

    zero_division_preventer = pd.Series(index=technical_counts.index, data=1)
    general_counts = zero_division_preventer.add(general_counts, fill_value=0)
    if normalized:
        return (
            technical_counts
            * general_word_count
            / general_counts
            / technical_word_count
        )
    else:
        return technical_counts / general_counts


if __name__ == "__main__":
    PATH_TO_GENERAL_DOMAIN = "../data/wiki_testing.pkl"
    PATH_TO_TECHNICAL_DOMAIN = "../data/pmc_testing.pkl"
    wiki = pd.read_pickle(PATH_TO_GENERAL_DOMAIN)
    pmc = pd.read_pickle(PATH_TO_TECHNICAL_DOMAIN)
    # print(pmc, '\n', wiki)
    pairdf = weirdness(pmc[:200], wiki[:500])
    print(pairdf.sort_values(ascending=False).head(50))
