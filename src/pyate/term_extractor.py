# c_value

import time
import math
from typing import Mapping, Sequence

import spacy
import pickle
import pandas as pd
import numpy as np

from .term_extraction import TermExtraction, add_term_extraction_method, Corpus


@add_term_extraction_method
def term_extractor(
    technical_corpus: Corpus,
    general_corpus: Corpus,
    # general_corpus_size=TermExtraction.DEFAULT_GENERAL_DOMAIN_SIZE,
    weights: Sequence[float] = None,
    verbose: bool = False,
    technical_counts: Mapping[str, int] = None,
):

    # if general_corpus is None:
    # general_corpus = TermExtraction.DEFAULT_GENERAL_DOMAIN[:100]

    # reused initializations
    technical_counts_seperate = TermExtraction(
        technical_corpus
    ).count_terms_from_documents(True, verbose=verbose)
    if type(technical_counts_seperate) is pd.DataFrame:
        technical_counts = technical_counts_seperate.sum(axis=1)
    else:
        technical_counts = technical_counts_seperate
    XLOGX = lambda x: x * math.log(x) if x else 0

    # domain pertinence
    general_counts = TermExtraction(
        general_corpus, vocab=technical_counts.index
    ).count_terms_from_documents(verbose=verbose)
    general_counts /= general_counts.max()
    domain_pertinence = pd.DataFrame(
        data={
            "technical": technical_counts / technical_counts.sum(),
            "general": general_counts,
        }
    ).fillna(0)

    def domain_pertinence_function(s):
        tech, gen = s.iloc
        return tech / max(tech, gen)

    domain_pertinence = domain_pertinence.apply(domain_pertinence_function, axis=1)

    # domain consensus
    domain_consensus = technical_counts_seperate
    if type(technical_counts_seperate) is pd.DataFrame:
        domain_consensus = domain_consensus.div(domain_consensus.sum(axis=0), axis=1)
        domain_consensus = domain_consensus.applymap(lambda x: -XLOGX(x))
        domain_consensus = domain_consensus.sum(axis=1)
    else:
        domain_consensus = domain_consensus.apply(lambda x: -XLOGX(x))

    # Lexical cohesion
    term_words = set(
        word for term in technical_counts_seperate.index for word in term.split()
    )
    term_counts = TermExtraction(
        technical_corpus, term_words
    ).count_terms_from_documents()
    lexical_cohesion = technical_counts

    def lexical_cohesion_function(row):
        word, freq = row.iloc
        return (
            TermExtraction.word_length(word)
            * XLOGX(freq)  # remove plus 1 later
            / sum(map(lambda s: term_counts.loc[s], word.split()))
        )

    lexical_cohesion = pd.Series(
        lexical_cohesion.reset_index().apply(lexical_cohesion_function, axis=1).values,
        index=lexical_cohesion.index,
    )

    # print(
    #         domain_pertinence.sort_values(ascending=False),
    #         "\n",
    #         domain_consensus.sort_values(ascending=False),
    #         "\n",
    #         lexical_cohesion.sort_values(ascending=False),
    #     )
    # print('lexical cohesion:', lexical_cohesion)

    domain_pertinence /= domain_pertinence.max()
    domain_consensus /= domain_consensus.max()
    lexical_cohesion /= lexical_cohesion.max()

    df = pd.DataFrame(
        data={
            "domain_pertinence": domain_pertinence,
            "domain_consensus": domain_consensus,
            "lexical_cohesion": lexical_cohesion,
        }
    ).fillna(0)

    if verbose:
        print(
            domain_pertinence.sort_values(ascending=False),
            "\n",
            domain_consensus.sort_values(ascending=False),
            "\n",
            lexical_cohesion.sort_values(ascending=False),
        )
    if weights is None:
        weights = np.array([1, 1, 1]) / 3
    return df.dot(weights)


if __name__ == "__main__":
    PATH_TO_GENERAL_DOMAIN = "../data/wiki_testing.pkl"
    PATH_TO_TECHNICAL_DOMAIN = "../data/pmc_testing.pkl"
    wiki = pd.read_pickle(PATH_TO_GENERAL_DOMAIN)
    pmc = pd.read_pickle(PATH_TO_TECHNICAL_DOMAIN)
    # term_extractor(pmc[:50], wiki[:250])

    # print(term_extractor(pmc, wiki).sort_values(ascending=False).head(50))
    print(
        term_extractor(pmc[:50], wiki[:1000], verbose=True)
        .sort_values(ascending=False)
        .head(50)
    )
    # print(pmc[0])
    # Syntactic Analysis: (PP NP PP CNP RVP NP PP)
    # POS: (PAJNNN AJN PNCNNWVANPJN)
