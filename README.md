# PYthon Automated Term Extraction
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Built with spaCy](https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg)](https://spacy.io)

Python implementation of term extraction algorithms such as C-Value, Basic, Combo Basic, Weirdness and Term Extractor using Spacy POS tagging.

Warning: Weirdness and Term Extractor doesn't work through pip at the moment due to errors with upload CSV files.

## Installation
Using pip:
```bash
pip install pyate https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz
```

## Quickstart
To get started, simply call one of the implemented algorithms. According to studies, `combo_basic` is the most precise, though `basic` and `cvalue` is not too far behind.
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
nlp.add_pipeline(TermExtractionPipeline())
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
where `func` is essentially your term extracting algorithm that takes in a corpus (either a string or iterator of strings) and outputs a Pandas Series of term-value pairs of terms and their respective termhoods. `func` is by default `combo_basic`.

## Summary of functions 
Each of `cvalue, basic, combo_basic, weirdness` and `term_extractor` take in a string or an iterator of strings and outputs a Pandas Series of term-value pairs, where higher values indicate higher chance of being a domain specific term. Furthermore, `weirdness` and `term_extractor` take a `general_corpus` key word argument which must be an iterator of strings.

## Todo
* Add PU-ATR algorithm since its precision is a lot higher, though more computationally expensive
* Add sources

## Sources
* TODO
