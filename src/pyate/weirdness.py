# weirdness.py
from typing import Mapping

import pandas as pd

from .term_extraction import add_term_extraction_method
from .term_extraction import Corpus
from .term_extraction import TermExtraction


@add_term_extraction_method
def weirdness(
    technical_corpus: Corpus,
    general_corpus: Corpus = None,
    general_corpus_size=TermExtraction.config["DEFAULT_GENERAL_DOMAIN_SIZE"],
    normalized: bool = False,
    technical_counts: Mapping[str, int] = None,
    verbose: bool = False,
) -> pd.Series:
    """The weirdness algorithm (Fedorenko, Astrakhantsev and Turdakov, 2013 from http://ceur-ws.org/Vol-1031/paper3.pdf)."""

    if general_corpus is None:
        general_corpus = TermExtraction.get_general_domain(
            size=general_corpus_size)

    if technical_counts is None:
        # this is the bulk of the calculations
        technical_counts = TermExtraction(
            technical_corpus).count_terms_from_documents(verbose=verbose)
    general_counts = TermExtraction(
        general_corpus, technical_counts.index).count_terms_from_documents()
    technical_word_count = TermExtraction.word_length(
        "\n".join(technical_corpus))
    general_word_count = TermExtraction.word_length("\n".join(general_corpus))

    zero_division_preventer = pd.Series(index=technical_counts.index, data=1)
    general_counts = zero_division_preventer.add(general_counts, fill_value=0)
    if normalized:
        return (technical_counts * general_word_count / general_counts /
                technical_word_count)
    else:
        return technical_counts / general_counts


if __name__ == "__main__":
    PATH_TO_GENERAL_DOMAIN = "../data/wiki_testing.pkl"
    PATH_TO_TECHNICAL_DOMAIN = "../data/pmc_testing.pkl"
    wiki = pd.read_pickle(PATH_TO_GENERAL_DOMAIN)
    pmc = pd.read_pickle(PATH_TO_TECHNICAL_DOMAIN)
    pairdf = weirdness(pmc[:200], wiki[:500])
    print(pairdf.sort_values(ascending=False).head(50))
