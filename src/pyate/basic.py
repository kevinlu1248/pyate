# basic.py

import numpy as np

from .term_extraction import TermExtraction, add_term_extraction_method
from .combo_basic import combo_basic


@add_term_extraction_method
def basic(technical_corpus: str, *args, **kwargs):
    weights = np.array([1, 3.5, 0])
    return combo_basic(technical_corpus, weights=weights, *args, **kwargs)


if __name__ == "__main__":
    corpus = "Hello I am a term extractor."
    print(TermExtraction(corpus).basic().sort_values(ascending=False).head(50))
