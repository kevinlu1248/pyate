from pyate import *
import spacy

CORPUS = "Hello world! I am a term extractor"
ALGORITHMS = (basic, combo_basic, cvalues, weirdness, term_extractor)
nlp = spacy.load("en_core_web_sm")


def test_algs():
    for func in ALGORITHMS:
        assert "term extractor" in func(CORPUS).index


def test_pipelines():
    for func in ALGORITHMS:
        nlp.add_pipe(TermExtractionPipeline(nlp, func))
    doc = nlp(CORPUS)
    for func in ALGORITHMS:
        assert "term extractor" in getattr(doc._, func.__name__).index


def test_lang_change():
    TermExtraction.set_language("it", "it_core_news_sm")  # italian
    for func in ALGORITHMS:
        func(CORPUS)
