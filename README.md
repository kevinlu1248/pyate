# PYthon Automated Term Extraction
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyate.svg)](https://pypi.python.org/pypi/pyate/)
[![PyPI version fury.io](https://badge.fury.io/py/pyate.svg)](https://pypi.python.org/pypi/pyate/)
[![PyPI download month](https://img.shields.io/pypi/dm/pyate.svg)](https://pypi.python.org/pypi/pyate/)
[![HitCount](http://hits.dwyl.com/kevinlu1248/pyate.svg)](http://hits.dwyl.com/kevinlu1248/pyate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with spaCy](https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg)](https://spacy.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of term extraction algorithms such as C-Value, Basic, Combo Basic, Weirdness and Term Extractor using spaCy POS tagging.

If you have a suggestion for another ATE algorithm you would like implemented in this package feel free to file it as an issue with the paper the algorithm is based on.

For ATE packages implemented in Scala and Java, see [ATR4S](https://github.com/ispras/atr4s) and [JATE](https://github.com/ziqizhang/jate), respectively.

## :tada: Installation
Using pip:
```bash
pip install pyate https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz
```

## :rocket: Quickstart
To get started, simply call one of the implemented algorithms. According to Astrakhantsev 2016, `combo_basic` is the most precise of the five algorithms, though `basic` and `cvalue` is not too far behind (see Precision). The same study shows that PU-ATR and KeyConceptRel have higher precision than `combo_basic` but are not implemented and PU-ATR take significantly more time since it uses machine learning.
```python3
from pyate import combo_basic

# source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1994795/
string = """Central to the development of cancer are genetic changes that endow these “cancer cells” with many of the
hallmarks of cancer, such as self-sufficient growth and resistance to anti-growth and pro-death signals. However, while the
genetic changes that occur within cancer cells themselves, such as activated oncogenes or dysfunctional tumor suppressors,
are responsible for many aspects of cancer development, they are not sufficient. Tumor promotion and progression are
dependent on ancillary processes provided by cells of the tumor environment but that are not necessarily cancerous 
themselves. Inflammation has long been associated with the development of cancer. This review will discuss the reflexive 
relationship between cancer and inflammation with particular focus on how considering the role of inflammation in physiologic 
processes such as the maintenance of tissue homeostasis and repair may provide a logical framework for understanding the 
connection between the inflammatory response and cancer."""

print(combo_basic(string).sort_values(ascending=False))
""" (Output)
dysfunctional tumor                1.443147
tumor suppressors                  1.443147
genetic changes                    1.386294
cancer cells                       1.386294
dysfunctional tumor suppressors    1.298612
logical framework                  0.693147
sufficient growth                  0.693147
death signals                      0.693147
many aspects                       0.693147
inflammatory response              0.693147
tumor promotion                    0.693147
ancillary processes                0.693147
tumor environment                  0.693147
reflexive relationship             0.693147
particular focus                   0.693147
physiologic processes              0.693147
tissue homeostasis                 0.693147
cancer development                 0.693147
dtype: float64
"""
```
If you would like to add this to a spacy pipeline, simply use add Spacy's `add_pipe` method.
```python3
import spacy
from pyate.term_extraction_pipeline import TermExtractionPipeline

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(TermExtractionPipeline())
doc = nlp(string)
print(doc._.combo_basic.sort_values(ascending=False).head(5))
""" (Output)
dysfunctional tumor                1.443147
tumor suppressors                  1.443147
genetic changes                    1.386294
cancer cells                       1.386294
dysfunctional tumor suppressors    1.298612
dtype: float64
"""
```
Also, `TermExtractionPipeline.__init__` is defined as follows
```
__init__(
  self,
  func: Callable[..., pd.Series] = combo_basic,
  *args,
  **kwargs
)
```
where `func` is essentially your term extracting algorithm that takes in a corpus (either a string or iterator of strings) and outputs a Pandas Series of term-value pairs of terms and their respective termhoods. `func` is by default `combo_basic`. `args` and `kwargs` are for you to overide default values for the function, which you can find by running `help` (might document later on).

### Summary of functions 
Each of `cvalue, basic, combo_basic, weirdness` and `term_extractor` take in a string or an iterator of strings and outputs a Pandas Series of term-value pairs, where higher values indicate higher chance of being a domain specific term. Furthermore, `weirdness` and `term_extractor` take a `general_corpus` key word argument which must be an iterator of strings which defaults to the General Corpus described below. 

All functions only take the string of which you would like to extract terms from as the mandatory input (the `technical_corpus`), as well as other tweakable settings, including `general_corpus` (contrasting corpus for `weirdness` and `term_extractor`), `general_corpus_size`, `verbose` (whether to print a progress bar), `weights`, `smoothing`, `have_single_word` (whether to have a single word count as a phrase) and `threshold`. If you have not read the papers and are unfamiliar with the algorithms, I recommend just using the default settings. Again, use `help` to find the details regarding each algorithm since they are all different.

### General Corpus
Under `path/to/site-packages/pyate/default_general_domain.csv`, there is a general CSV file of a general corpus, specifically, 3000 random sentences from Wikipedia. The source of it can be found at https://www.kaggle.com/mikeortman/wikipedia-sentences. Access it using it using the following after installing `pyate`.

```python3
import pandas as pd
from distutils.sysconfig import get_python_lib  
df = pd.read_csv(get_python_lib() + "/pyate/default_general_domain.csv")["SECTION_TEXT"]
print(df.head())
""" (Output)
0    '''Anarchism''' is a political philosophy that...
1    The term ''anarchism'' is a compound word comp...
2    ===Origins===\nWoodcut from a Diggers document...
3    Portrait of philosopher Pierre-Joseph Proudhon...
4    consistent with anarchist values is a controve...
Name: SECTION_TEXT, dtype: object
"""
```

## :dart: Precision
Here is the average precision of some of the implemented algorithms using the Average Precision (AvP) metric on seven distinct databases, as tested in Astrakhantsev 2016.
![Evaluation](evaluation.png)

## :stars: Motivation
This project was planned to be a tool to be connected to a Google Chrome Extension that highlights and defines key terms that the reader probably does not know of. Furthermore, term extraction is an area where there is not a lot of focused research on in comparison to other areas of NLP and especially recently is not viewed to be very practical due to the more general tool of NER tagging. However, modern NER tagging usually incorporates some combination of memorized words and deep learning which are spatially and computationally heavy. Furthermore, to generalize an algorithm to recognize terms to the ever growing areas of medical and AI research, a list of memorized words will not do.

Of the five implemented algorithms, none are expensive, in fact, the bottleneck of the space allocation and computation expense is from the spaCy model and spaCy POS tagging. This is because they most rely simply on POS patterns, word frequencies, and the existence of embedded term candidates. For example, the term candidate "breast cancer" implies that "malignant breast cancer" is probably not a term and simply a form of "breast cancer" that is "malignant" (implemented in C-Value).

## :pushpin: Todo
* Add PU-ATR algorithm since its precision is a lot higher, though more computationally expensive
* Page Rank algorithm
* Add sources
* Add voting algorithm and capabilities
* Optimize perhaps using Cython, however, the bottleneck is POS tagging by Spacy so this will not help much
* Clearer documentation

## :bookmark_tabs: Sources
I cannot seem to find the original Basic and Combo Basic papers but I found papers that referenced them. "ATR4S: Toolkit with State-of-the-art Automatic Terms Recognition Methods in Scala" more or less summarizes everything and incorporates several algorithms not incorporated in this package.
* [Automatic Recognition of Multi-word Terms: The C-value/ NC-value Method](https://www.researchgate.net/publication/220387502_Automatic_Recognition_of_Multi-word_Terms_The_C-value_NC-value_Method)
* [Domain-independent term extraction through domain modelling](https://aran.library.nuigalway.ie/handle/10379/4130)
* [ATR4S: Toolkit with State-of-the-art Automatic Terms Recognition Methods in Scala](https://arxiv.org/abs/1611.07804)
* [TermExtractor: a Web Application to Learn the Shared Terminology of Emergent Web Communities](https://link.springer.com/chapter/10.1007/978-1-84628-858-6_32)
* [Learning Domain Ontologies from
Document Warehouses and Dedicated
Web Sites](https://www.aclweb.org/anthology/J04-2002.pdf)
* [A Comparative Evaluation of Term Recognition Algorithms](https://www.aclweb.org/anthology/L08-1281/)
* [SemRe-Rank: Improving Automatic Term Extraction By Incorporating
Semantic Relatedness With Personalised PageRank](https://arxiv.org/pdf/1711.03373.pdf)
* [Term extraction: A Review
Draft Version 091221](https://www.ida.liu.se/~larah03/Publications/tereview_v2.pdf)
