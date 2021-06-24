# PYthon Automated Term Extraction

[![Build Status](https://travis-ci.com/kevinlu1248/pyate.svg?branch=master)](https://travis-ci.com/kevinlu1248/pyate)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/pyate.svg)](https://pypi.python.org/pypi/pyate/)
[![PyPI version fury.io](https://badge.fury.io/py/pyate.svg)](https://pypi.python.org/pypi/pyate/)
[![Downloads](https://pepy.tech/badge/pyate)](https://pepy.tech/project/pyate)
[![Downloads](https://pepy.tech/badge/pyate/month)](https://pepy.tech/project/pyate/month)
[![Downloads](https://pepy.tech/badge/pyate/week)](https://pepy.tech/project/pyate/week)
[![HitCount](http://hits.dwyl.com/kevinlu1248/pyate.svg)](http://hits.dwyl.com/kevinlu1248/pyate)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with spaCy](https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg)](https://spacy.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of term extraction algorithms such as C-Value, Basic,
Combo Basic, Weirdness and Term Extractor using spaCy POS tagging.

NEW: Documentation can be found at https://kevinlu1248.github.io/pyate/. The
documentation so far is still missing two algorithms and details about the
`TermExtraction` class but I will have it done soon.

NEW: spaCy V3 is supported! For spaCy V2, use `pyate==0.4.3` and view the
[spaCy V2 README.md file](README-spacy-v2.md)

If you have a suggestion for another ATE algorithm you would like implemented in
this package feel free to file it as an issue with the paper the algorithm is
based on.

For ATE packages implemented in Scala and Java, see
[ATR4S](https://github.com/ispras/atr4s) and
[JATE](https://github.com/ziqizhang/jate), respectively.

## :tada: Installation

Using pip:

```bash
pip install pyate https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz
```

## :rocket: Quickstart

To get started, simply call one of the implemented algorithms. According to
Astrakhantsev 2016, `combo_basic` is the most precise of the five algorithms,
though `basic` and `cvalues` is not too far behind (see Precision). The same
study shows that PU-ATR and KeyConceptRel have higher precision than
`combo_basic` but are not implemented and PU-ATR take significantly more time
since it uses machine learning.

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
processes such as the maintenance of tissue homeostasis and repair may provide a logical framework for understanding the U
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

If you would like to add this to a spacy pipeline, simply use add Spacy's
`add_pipe` method.

```python3
import spacy
from pyate.term_extraction_pipeline import TermExtractionPipeline

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("combo_basic")
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

where `func` is essentially your term extracting algorithm that takes in a
corpus (either a string or iterator of strings) and outputs a Pandas Series of
term-value pairs of terms and their respective termhoods. `func` is by default
`combo_basic`. `args` and `kwargs` are for you to overide default values for the
function, which you can find by running `help` (might document later on).

### Summary of functions

Each of `cvalues, basic, combo_basic, weirdness` and `term_extractor` take in a
string or an iterator of strings and outputs a Pandas Series of term-value
pairs, where higher values indicate higher chance of being a domain specific
term. Furthermore, `weirdness` and `term_extractor` take a `general_corpus` key
word argument which must be an iterator of strings which defaults to the General
Corpus described below.

All functions only take the string of which you would like to extract terms from
as the mandatory input (the `technical_corpus`), as well as other tweakable
settings, including `general_corpus` (contrasting corpus for `weirdness` and
`term_extractor`), `general_corpus_size`, `verbose` (whether to print a progress
bar), `weights`, `smoothing`, `have_single_word` (whether to have a single word
count as a phrase) and `threshold`. If you have not read the papers and are
unfamiliar with the algorithms, I recommend just using the default settings.
Again, use `help` to find the details regarding each algorithm since they are
all different.

### General Corpus

Under `path/to/site-packages/pyate/default_general_domain.en.csv`, there is a
general CSV file of a general corpus, specifically, 3000 random sentences from
Wikipedia. The source of it can be found at
https://www.kaggle.com/mikeortman/wikipedia-sentences. Access it using it using
the following after installing `pyate`.

```python3
import pandas as pd
from distutils.sysconfig import get_python_lib
df = pd.read_csv(get_python_lib() + "/pyate/default_general_domain.en.csv")["SECTION_TEXT"]
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

### Other Languages

For switching languages, simply run
`Term_Extraction.set_language({language}, {model_name})`, where `model_name`
defaults to `language`. For example,
`Term_Extraction.set_language("it", "it_core_news_sm"})` for Italian. By
default, the language is English. So far, only _English_ (en) and _Italian_ (it)
are supported.

To add more languages, file an issue with a corpus of at least 3000 paragraphs
of a general domain in the desired language (preferably wikipedia) named
`default_general_domain.{lang}.csv` replacing lang with the ISO-639-1 code of
the language, or the ISO-639-2 if the language does not have a ISO-639-1 code
(can be found at https://www.loc.gov/standards/iso639-2/php/code_list.php). The
file format should be of the following form to be parsable by Pandas.

```
,SECTION_TEXT
0,"{paragraph_0}"
1,"{paragraph_1}"
...
```

Alternatively, place the file in `src/pyate` and file a pull request.

### Models

Though this model was originally intended for symbolic AI algorithms
(non-machine learning), I realized a spaCy model on term extraction can reach
significantly higher performance, and thus decided to include the model here.

For a comparison with the symbolic AI algorithms, see
[Precision](https://github.com/kevinlu1248/pyate#dart-precision). Note that only
the F-Score, accuracy and precision was taken here yet for the model, but for
the algorithms the AvP was taken so directly comparing the metrics would not
really make sense.

| URL                                                                                        | F-Score (%) | Precision (%) | Recall (%) |
| ------------------------------------------------------------------------------------------ | ----------- | ------------- | ---------- |
| https://github.com/kevinlu1248/pyate/releases/download/v0.4.2/en_acl_terms_sm-2.0.4.tar.gz | 94.71       | 95.41         | 94.03      |

The model was trained and evaluated on the
[ACL dataset](http://pars.ie/lr/acl-rd-tec-terminology/_acl_arc_comp), which is
a computer science oriented dataset where the terms are manually picked. This
has not yet been tested on other fields yet, however.

This model does not come with PyATE. To install, run

```bash
pip install https://github.com/kevinlu1248/pyate/releases/download/v0.4.2/en_acl_terms_sm-2.0.4.tar.gz
```

To extract terms,

```python3
import spacy

nlp = spacy.load("en_acl_terms_sm")
doc = nlp("Hello world, I am a term extraction algorithm.")
print(doc.ents)
"""
(term extraction, algorithm)
"""
```

## :dart: Precision

Here is the average precision of some of the implemented algorithms using the
Average Precision (AvP) metric on seven distinct databases, as tested in
Astrakhantsev 2016. ![Evaluation](img/evaluation.png)

## :stars: Motivation

This project was planned to be a tool to be connected to a Google Chrome
Extension that highlights and defines key terms that the reader probably does
not know of. Furthermore, term extraction is an area where there is not a lot of
focused research on in comparison to other areas of NLP and especially recently
is not viewed to be very practical due to the more general tool of NER tagging.
However, modern NER tagging usually incorporates some combination of memorized
words and deep learning which are spatially and computationally heavy.
Furthermore, to generalize an algorithm to recognize terms to the ever growing
areas of medical and AI research, a list of memorized words will not do.

Of the five implemented algorithms, none are expensive, in fact, the bottleneck
of the space allocation and computation expense is from the spaCy model and
spaCy POS tagging. This is because they mostly rely simply on POS patterns, word
frequencies, and the existence of embedded term candidates. For example, the
term candidate "breast cancer" implies that "malignant breast cancer" is
probably not a term and simply a form of "breast cancer" that is "malignant"
(implemented in C-Value).

## :pushpin: Todo

- Add other languages and data encapsulation for set language
- Add automated tests and CI/CD
- Add a brief CLI
- Make NER version of this using the datasets from the sources
- Add PU-ATR algorithm since its precision is a lot higher, though more
  computationally expensive
- Page Rank algorithm
- Add sources
- Add voting algorithm and capabilities
- Optimize perhaps using Cython, however, the bottleneck is POS tagging by Spacy
  and word counting with Pandas and Numpy, which are already at C-level so this
  will not help much
- Clearer documentation
- Allow GPU acceleration with Cupy

## :bookmark_tabs: Sources

I cannot seem to find the original Basic and Combo Basic papers but I found
papers that referenced them. "ATR4S: Toolkit with State-of-the-art Automatic
Terms Recognition Methods in Scala" more or less summarizes everything and
incorporates several algorithms not in this package.

- [Automatic Recognition of Multi-word Terms: The C-value/ NC-value Method](https://www.researchgate.net/publication/220387502_Automatic_Recognition_of_Multi-word_Terms_The_C-value_NC-value_Method)
- [Domain-independent term extraction through domain modelling](https://aran.library.nuigalway.ie/handle/10379/4130)
- [ATR4S: Toolkit with State-of-the-art Automatic Terms Recognition Methods in Scala](https://arxiv.org/abs/1611.07804)
- [TermExtractor: a Web Application to Learn the Shared Terminology of Emergent Web Communities](https://link.springer.com/chapter/10.1007/978-1-84628-858-6_32)
- [Learning Domain Ontologies from Document Warehouses and Dedicated Web Sites](https://www.aclweb.org/anthology/J04-2002.pdf)
- [A Comparative Evaluation of Term Recognition Algorithms](https://www.aclweb.org/anthology/L08-1281/)
- [SemRe-Rank: Improving Automatic Term Extraction By Incorporating Semantic Relatedness With Personalised PageRank](https://arxiv.org/pdf/1711.03373.pdf)
- [Term extraction: A Review Draft Version 091221](https://www.ida.liu.se/~larah03/Publications/tereview_v2.pdf)

## :closed_book: Influences on Academia

This package was used in the paper
([Unsupervised Technical Domain Terms Extraction using Term Extractor (Dowlagar and Mamidi, 2021)](https://arxiv.org/pdf/2101.09015.pdf).

## :coffee: Buy Me a Coffee

If my work helped you, please consider buying me a coffee at
https://www.buymeacoffee.com/kevinlu1248.
