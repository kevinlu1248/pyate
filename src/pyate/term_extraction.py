# term_extraction.py

import collections.abc
from collections import defaultdict
from multiprocessing import Pool
from typing import Callable, Iterable, Sequence, Union

import ahocorasick
import numpy as np
import pandas as pd
import pkg_resources
from tqdm import tqdm

import spacy
from spacy.matcher import Matcher

start_ = 0
tmp = 0
doctime, matchertime = 0, 0
Corpus = Union[str, Sequence[str]]


class TermExtraction:
    # TODO: find some way to prevent redundant loading of csv files
    nlp = spacy.load("en_core_web_sm", parser=False, entity=False)
    language = "en"
    MAX_WORD_LENGTH = 6
    DEFAULT_GENERAL_DOMAIN_SIZE = 300
    DEFAULT_GENERAL_DOMAIN = pd.read_csv(
        pkg_resources.resource_stream(__name__, "default_general_domain.en.csv"),
        nrows=DEFAULT_GENERAL_DOMAIN_SIZE,
    )

    noun, adj, prep = (
        {"POS": "NOUN", "IS_PUNCT": False},
        {"POS": "ADJ", "IS_PUNCT": False},
        {"POS": "DET", "IS_PUNCT": False},
    )

    patterns = [
        [adj],
        [{"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "*", "IS_PUNCT": False}, noun],
        [
            {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "*", "IS_PUNCT": False},
            noun,
            prep,
            {"POS": {"IN": ["ADJ", "NOUN"]}, "OP": "*", "IS_PUNCT": False},
            noun,
        ],
    ]

    def __init__(
        self,
        corpus: Union[str, Iterable[str]],
        vocab: Sequence[str] = None,
        patterns=patterns,
        do_parallelize: bool = True,
        language="en",
    ):
        """
        If corpus is a string, then find vocab sequentially, but if the corpus is an iterator,
        compute in parallel. If there is a vocab list, only search for frequencies from the vocab list,
        otherwise search using the patterns.

        TODO: do_parallelize and do_lower
        """
        self.corpus = corpus
        self.vocab = vocab
        self.patterns = patterns
        self.do_parallelize = do_parallelize
        TermExtraction.language = language

    @staticmethod
    def set_language(language: str, model_name: str = ""):
        """
        For changing the language. Currently, the DEFAULT_GENERAL_DOMAIN is still in English only.
        If you have a good dataset in another language please put it in an issue on Github.
        """
        if model_name == "":
            model_name = language
        TermExtraction.language = language
        TermExtraction.nlp = spacy.load(model_name)
        TermExtraction.matcher = Matcher(TermExtraction.nlp.vocab)
        TermExtraction.DEFAULT_GENERAL_DOMAIN = pd.read_csv(
            pkg_resources.resource_stream(
                __name__, f"default_general_domain.{language}.csv"
            ),
            nrows=TermExtraction.DEFAULT_GENERAL_DOMAIN_SIZE,
        )

    @staticmethod
    def word_length(string: str):
        return string.count(" ") + 1

    @property
    def trie(self):
        if not hasattr(self, "_TermExtraction__trie"):
            self.__trie = ahocorasick.Automaton()
            for idx, key in enumerate(self.vocab):
                self.__trie.add_word(key, (idx, key))
            self.__trie.make_automaton()
        return self.__trie

    def count_terms_from_document(self, document: str):
        # for single documents
        term_counter = defaultdict(int)
        if self.vocab is None:
            # initialize a Matcher here - not at the class level
            new_matcher = Matcher(TermExtraction.nlp.vocab)

            def add_to_counter(matcher, doc, i, matches):
                match_id, start, end = matches[i]
                candidate = str(doc[start:end])
                if (
                    TermExtraction.word_length(candidate)
                    <= TermExtraction.MAX_WORD_LENGTH
                ):
                    term_counter[candidate] += 1

            for i, pattern in enumerate(self.patterns):
                new_matcher.add("term{}".format(i), add_to_counter, pattern)

            doc = TermExtraction.nlp(document.lower(), disable=["parser", "ner"])
            matches = new_matcher(doc)
        else:
            for end_index, (insert_order, original_value) in self.trie.iter(
                document.lower()
            ):
                term_counter[original_value] += 1
        return term_counter

    def count_terms_from_documents(self, seperate: bool = False, verbose: bool = False):
        if hasattr(self, "_TermExtraction__term_counts"):
            return self.__term_counts

        if type(self.corpus) is str:
            self.__term_counts = pd.Series(self.count_terms_from_document(self.corpus))
            return self.__term_counts
        elif isinstance(self.corpus, collections.abc.Iterable):
            if seperate:
                term_counters = []
            else:
                term_counter = pd.Series(dtype="int64")
            if verbose:
                pbar = tqdm(total=len(self.corpus))

            def callback(counter_list):
                if verbose:
                    pbar.update(1)
                if seperate:
                    term_counters.append(
                        (tuple(counter_list.keys()), tuple(counter_list.values()))
                    )
                else:
                    nonlocal term_counter
                    # print(tuple(counter_list.values()))
                    term_counter = term_counter.add(
                        pd.Series(
                            index=tuple(counter_list.keys()),
                            data=tuple(counter_list.values()),
                            dtype=np.int64,
                        ),
                        fill_value=0,
                    ).astype(np.int64)

            def error_callback(e):
                print(e)

            P = Pool()

            for document in self.corpus:
                P.apply_async(
                    self.count_terms_from_document,
                    [document],
                    callback=callback,
                    error_callback=error_callback,
                )
            P.close()
            P.join()

            P.terminate()
            if verbose:
                pbar.close()
        else:
            raise TypeError()

        if seperate:

            def counter_to_series(counter):
                return pd.Series(data=counter[1], index=counter[0], dtype="int8")

            self.__term_counter = (
                pd.DataFrame(data=map(counter_to_series, term_counters))
                .fillna(0)
                .astype("int8")
                .T
            )
            return self.__term_counter
        else:
            self.__term_counter = term_counter
            return self.__term_counter


def add_term_extraction_method(extractor: Callable[..., pd.Series]):
    def term_extraction_decoration(self, *args, **kwargs):
        return extractor(
            self.corpus,
            technical_counts=self.count_terms_from_documents(),
            *args,
            **kwargs,
        )

    setattr(TermExtraction, extractor.__name__, term_extraction_decoration)
    return extractor


if __name__ == "__main__":
    PATH_TO_GENERAL_DOMAIN = "../data/wiki_testing.pkl"
    PATH_TO_TECHNICAL_DOMAIN = "../data/pmc_testing.pkl"
    wiki = pd.read_pickle(PATH_TO_GENERAL_DOMAIN)
    pmc = pd.read_pickle(PATH_TO_TECHNICAL_DOMAIN)
    vocab = ["Cutaneous melanoma", "cancer", "secondary clusters", "bio"]
    #     start()
    print(
        TermExtraction(pmc[:100]).count_terms_from_documents(
            seperate=True, verbose=True
        )
    )
