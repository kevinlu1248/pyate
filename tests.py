from src import *

if __name__ == "__main__":
    corpus = "Hello world! I am a term extractor"
    term_extraction = TermExtraction(corpus)
    functions = (basic, combo_basic, cvalues, term_extractor, weirdness)
    for f in functions:
        print(f.__name__, "\n", f(corpus), "\n")
