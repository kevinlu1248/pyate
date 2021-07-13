*****
General Usage
*****

Algorithm Usage
########
Each of ``cvalues``,  ``basic``,  ``combo_basic``, ``weirdness`` and ``term_extractor`` take in a string or an iterator of strings and outputs a Pandas Series of term-value pairs, where higher values indicate higher chance of being a domain specific term. Furthermore, ``weirdness`` and ``term_extractor`` take a general_corpus key word argument which must be an iterator of strings which defaults to the General Corpus described below.

All functions only take the string of which you would like to extract terms from as the mandatory input (the ``technical_corpus``), as well as other tweakable settings, including ``general_corpus`` (contrasting corpus for ``weirdness`` and ``term_extractor``), ``general_corpus_size``, ``verbose`` (whether to print a progress bar), ``weights``, ``smoothing``, ``have_single_word`` (whether to have a single word count as a phrase) and ``threshold``. If you have not read the papers and are unfamiliar with the algorithms, I recommend just using the default settings. Again, use help to find the details regarding each algorithm since they are all different. More information can be found under "Algorithms".

General Corpus
########
Under path/to/site-packages/pyate/default_general_domain.en.zip, there is a general CSV file of a general corpus, specifically, 3000 random sentences from Wikipedia. The source of it can be found at https://www.kaggle.com/mikeortman/wikipedia-sentences. Access it using it using the following after installing pyate.

::

    from pyate import TermExtraction
    df = TermExtraction.get_general_domain()
    print(df.head())
    """
       Unnamed: 0                                       SECTION_TEXT
    0           0  '''Anarchism''' is a political philosophy that...
    1           1  The term ''anarchism'' is a compound word comp...
    2           2  ===Origins===\nWoodcut from a Diggers document...
    3           3  Portrait of philosopher Pierre-Joseph Proudhon...
    4           4  consistent with anarchist values is a controve...
    """

Other Languages
########
For switching languages, simply run TermExtraction.set_language({language}, {model_name}), where model_name defaults to language. For example, Term_Extraction.set_language("it", "it_core_news_sm"}) for Italian. By default, the language is English. So far, only English (en) and Italian (it) are supported.

To add more languages, file an issue with a corpus of at least 3000 paragraphs of a general domain in the desired language (preferably wikipedia) named default_general_domain.{lang}.zip replacing lang with the ISO-639-1 code of the language, or the ISO-639-2 if the language does not have a ISO-639-1 code (can be found at https://www.loc.gov/standards/iso639-2/php/code_list.php). The file format should be of the following form to be parsable by Pandas.

::

    ,SECTION_TEXT
    0,"{paragraph_0}"
    1,"{paragraph_1}"
    ...

Alternatively, place the file in src/pyate and file a pull request.

Other Configurations
#########

There are five default settings configurable by the user:

* ``spacy_model:str = "en_core_web_sm"`` is the name of the spaCy model to be used by ``TermExtraction`` for POS tagging to filter for possible term candidates.
* ``language:str = "en"`` indicates the language to be used, which must be the ISO-639-1 or  ISO-639-2 of the language.
* ``MAX_WORD_LENGTH:int = 6`` is the maximum number of words in a phrase to be considered a candidate for a term. Any phrases longer will not be considered as a candidate. Since the time complexity is roughly O(MAX_WORD_LENGTH^2), it is a generally good idea to keep this value as low as possible.
* ``DEFAULT_GENERAL_DOMAIN:int = 300`` is the number of sentences to be used in the general domain. A larger value increases accuracy but also computation time and memory usage.
* ``dtype:np.dtype = np.uint16`` is the dtype to be used by the counters, which are Pandas series. By default, it could only store a frequency of at most 65535, so increase this setting if term frequencies exceed this value, which consequently also increases the memory usage. The default is sufficient for most use cases, however.

To configure them, simply run ``TermExtraction.configure(new_settings)`` where ``new_settings`` is a dict of values to update. For example,

::

    import numpy as np
    TermExtraction.configure({"dtype": np.uint32})

would increase the ``dtype`` setting to ``np.uint32``.
