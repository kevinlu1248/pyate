import spacy
from src import *

if __name__ == "__main__":
    corpus = "Hello world! I am a term extractor"
    nlp = spacy.load("en_core_web_sm")

    functions = (basic, combo_basic, cvalues)
    for func in functions:
        nlp.add_pipe(TermExtractionPipeline(nlp, func), func.__name__)
    doc = nlp(corpus)
    for func in functions:
        print(func.__name__, "\n", getattr(doc._, func.__name__), "\n")
