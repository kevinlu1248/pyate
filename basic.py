# combo basic
import math
from tqdm import tqdm
import pandas as pd
import numpy as np
from term_extraction import TermExtraction, add_term_extraction_method
from combo_basic import combo_basic

@add_term_extraction_method
def basic(technical_corpus, *args, **kwargs):
    weights = np.array([1, 3.5, 0])
    return combo_basic(technical_corpus, weights=weights, *args, **kwargs)

if __name__ == "__main__":
    import pickle
    pkl = pickle.load(open("../data/pmc_testing.pkl", "rb"))
    print(len(pkl))
    corpus = pkl
    print(TermExtraction(pkl[0]).basic().sort_values(ascending=False).head(50))
