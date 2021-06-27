import time

from pyate import basic
from pyate import combo_basic
from pyate import cvalues
from pyate import term_extractor
from pyate import TermExtraction
from pyate import weirdness

import spacy

# CORPUS = "Hello world! I am a term extractor"
# source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1994795/
CORPUS = """Central to the development of cancer are genetic changes that endow these “cancer cells” with many of the hallmarks of cancer, such as self-sufficient growth and resistance to anti-growth and pro-death signals. However, while the genetic changes that occur within cancer cells themselves, such as activated oncogenes or dysfunctional tumor suppressors, are responsible for many aspects of cancer development, they are not sufficient. Tumor promotion and progression are dependent on ancillary processes provided by cells of the tumor environment but that are not necessarily cancerous themselves. Inflammation has long been associated with the development of cancer. This review will discuss the reflexive relationship between cancer and inflammation with particular focus on how considering the role of inflammation in physiologic processes such as the maintenance of tissue homeostasis and repair may provide a logical framework for understanding the U
connection between the inflammatory response and cancer."""
ALGORITHMS = (basic, combo_basic, cvalues, weirdness, term_extractor)
nlp = spacy.load("en_core_web_sm")


def test_algs():
    """
    Tests the algorithms and benchmarks them.
    """
    print("Testing algorithms:")
    start_of_test = time.time()
    TermExtraction(CORPUS)
    for func in ALGORITHMS:
        start = time.time()
        result = func(CORPUS)
        # assert "dysfuncitonal tumor" in result.index
        print(
            f"\nTime elapsed for algorithm {func.__name__}: {time.time() - start:.5f}s"
        )
        print(result.sort_values(ascending=False).head(5))
    print(f"\nTotal time elapsed: {time.time() - start_of_test:.5f}s\n")


def test_algs_cached():
    """
    Tests the algorithms executed from a TermExtraction object.
    """
    print("Testing algorithms from a TermExtraction object:")
    # initiating te:
    start = time.time()
    te = TermExtraction(CORPUS)
    print(
        f"Time to generate TermExtraction object: {time.time() - start:.5f}s")
    start_of_counting = time.time()
    te.count_terms_from_documents()
    print(f"Time to count terms: {time.time() - start_of_counting:.5f}s")
    for func in ALGORITHMS:
        start_of_alg = time.time()
        getattr(te, func.__name__)()
        print(
            f"Time to run algorithm {func.__name__}: {time.time() - start_of_alg:.5f}s"
        )
    print("Total time elapsed: {time.time() - start:.5f}\n")


def test_pipelines():
    """
    Tests the algorithms in pipelines.
    """
    # TODO: find time elapsed per function
    print("Testing algorithms in pipelines:")
    start = time.time()
    for func in ALGORITHMS:

        def decorated_algorithm(*args, **kwargs):
            start_of_alg = time.time()
            result = func(*args, **kwargs)
            print(
                f"Time to run algorithm {func.__name__}: {time.time() - start_of_alg:.2f}s"
            )
            return result

        # nlp.add_pipe(TermExtractionPipeline(nlp, func))
        nlp.add_pipe(func.__name__)
    doc = nlp(CORPUS)
    for func in ALGORITHMS:
        print(getattr(doc._, func.__name__))
    print(f"Total time elapsed: {time.time() - start:.5f}s\n")


def test_lang_change():
    """
    Tests language change.
    """
    print("Testing algorithms after a language change to Italian:")
    start = time.time()
    TermExtraction.set_language("it", "it_core_news_sm")
    for func in ALGORITHMS:
        func(CORPUS)
    print(f"Total time elapsed: {time.time() - start:.5f}s\n")
