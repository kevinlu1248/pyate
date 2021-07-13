import os
import time

import pytest as pytest

from pyate import TermExtraction
from pyate import basic
from pyate import combo_basic
from pyate import cvalues
from pyate import term_extractor
from pyate import weirdness

MODELS = {
    "en": "en_core_web_sm",
    "it": "it_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "es": "es_core_news_sm",
    "ru": "ru_core_news_sm",
    "nl": "nl_core_news_sm",
    "pt": "pt_core_news_sm"
}
ALGORITHMS = (basic, combo_basic, cvalues, weirdness, term_extractor)


@pytest.mark.parametrize("lang", MODELS.keys())
def test_lang(lang):
    """
    Tests language change.
    """
    print(f"Testing algorithms for language {lang}:")
    testfile = os.path.join(os.path.dirname(__file__),
                            os.path.join('data', lang + '.txt'))
    with open(testfile, "r") as fin:
        CORPUS = fin.read()
        start = time.time()
        TermExtraction.configure(
            {"language": lang, "model_name": MODELS[lang], "MAX_WORD_LENGTH": 8})
        try:
            TermExtraction.get_nlp(lang)
            for func in ALGORITHMS:
                result = func(CORPUS)
                print(
                    f"\nTime elapsed for algorithm {func.__name__}: {time.time() - start:.5f}s"
                )
                print(result.sort_values(ascending=False).head(10))
            print(f"Total time elapsed: {time.time() - start:.5f}s\n")
        except IOError:
            print(
                f"You need to install the missing model with: python -m spacy download {MODELS[lang]}")
