# c_value

import spacy
import pickle
import time
import math
import pandas as pd
import numpy as np
from term_extraction import TermExtraction

# from sklearn import preprocessing

start_ = 0
tmp = 0


def start():
    global start_
    start_ = time.time()


def end():
    global start_
    print(time.time() - start_)


def domain_pertinence(technical_corpus, general_corpus):
    # http://ceur-ws.org/Vol-1031/paper3.pdf
    return 1


def domain_consensus(technical_corpus, general_corpus):
    return 1


def lexical_cohesion(technical_corpus, general_corpus):
    return 1

@add_term_extraction_method
def term_extractor(technical_corpus, general_corpus, weights=None, verbose=False, technical_counts=None):

    # reused initializations
    start()
    technical_counts_seperate = TermExtraction(
        technical_corpus
    ).count_terms_from_documents(True, verbose=verbose)
    end()

    start()
    technical_counts = technical_counts_seperate.sum(axis=1)
    XLOGX = lambda x: x * math.log(x) if x else 0

    # domain pertinence
    general_counts = TermExtraction(
        general_corpus, technical_counts.index
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
    domain_consensus = domain_consensus.div(domain_consensus.sum(axis=0), axis=1)
    domain_consensus = domain_consensus.applymap(lambda x: -XLOGX(x))
    domain_consensus = domain_consensus.sum(axis=1)

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
        # print(word, freq)
        return (
            TermExtraction.word_length(word)
            * XLOGX(freq)  # remove plus 1 later
            / sum(map(lambda s: term_counts.loc[s], word.split()))
        )

    lexical_cohesion = pd.Series(
        lexical_cohesion.reset_index().apply(lexical_cohesion_function, axis=1).values,
        index=lexical_cohesion.index,
    )

    if True:
        domain_pertinence /= domain_pertinence.max()
        domain_consensus /= domain_consensus.max()
        lexical_cohesion /= lexical_cohesion.max()

    df = pd.DataFrame(
        data={
            "domain_pertinence": domain_pertinence,
            "domain_consensus": domain_consensus,
            "lexical_cohesion": lexical_cohesion,
        }
    )

    if verbose:
        print(
            domain_pertinence.sort_values(ascending=False).head(10),
            "\n",
            domain_consensus.sort_values(ascending=False).head(10),
            "\n",
            lexical_cohesion.sort_values(ascending=False).head(10),
        )
    if weights is None:
        weights = np.array([1, 1, 1]) / 3
    end()
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
