*****
Algorithms
*****

Five algorithms are implemented in this package: 

* C-Value
* Basic
* Combo Basic
* Weirdness
* Term Extractor.

As indicated in "General Usage", the following are parameters for the algorithms:

* ``technical_corpus: Union[str, Sequence[str]]`` is the corpus which technical terms are to be searched from if it is a ``str``. If it is a sequence of ``str``'s, then the algorithm will run on every element of the sequence.
* ``general_corpus:pd.Series = TermExtraction.get_general_domain()`` is the corpus which will be compared with.
* ``general_corpus_size:int = 300`` is the number of sentences desired by the ``general_corpus``. 
* ``verbose:bool = False``, whether a progress bar will be loaded.
* ``weights``, meaning varies between algorithms but essentially how much weight is placed on every factor for determining how likely a candidate is a term.
* ``smoothing: double = 0.01`` is the factor added to a number before dividing the number, to prevent division by zero errors. The default value works fairly well.
* ``have_single_word:bool = False`` determines whether single words are to be considered as terms.
* ``technical_counts: Mapping[str, int]`` is the frequency counter from each candidate to the frequency. Helpful for running multiple algorithms since this will allow the POS tagging and counting to only be done once. If it is not set, it will be computed.
* ``threshold: float = 0`` the threshold for the C-Value algorithms.

And the whether each parameters exists for each algorithm can be summarized by the following table:

+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+
|                | ``technical_corpus`` | ``general_corpus`` | ``general_corpus_size`` | ``normalized`` | ``verbose`` | ``weights`` | ``smoothing`` | ``have_single_word`` | ``technical_counts`` | ``threshold`` |
+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+
| C-Value        | ✓                        |                        |                             |                    | ✓               |                 | ✓                 | ✓                        |                          | ✓                 |
+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+
| Basic          | ✓                        |                        |                             |                    | ✓               | ✓               | ✓                 | ✓                        | ✓                        |                   |
+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+
| Combo Basic    | ✓                        |                        |                             |                    | ✓               | ✓               | ✓                 | ✓                        | ✓                        |                   |
+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+
| Weirdness      | ✓                        | ✓                      | ✓                           | ✓                  | ✓               |                 |                   |                          | ✓                        |                   |
+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+
| Term Extractor | ✓                        | ✓                      | ✓                           |                    | ✓               | ✓               |                   |                          | ✓                        |                   |
+----------------+--------------------------+------------------------+-----------------------------+--------------------+-----------------+-----------------+-------------------+--------------------------+--------------------------+-------------------+

.. note::
    Remember that ``technical_corpus`` is **required** for each algorithm.

Basic
########

.. autofunction:: pyate.basic

This algorithm is equivalent to ``combo_basic(technical_corpus, weights=np.array([1, 3.5, 0]), *args, **kwargs)``. In other words, it is ``combo_basic`` without the third factor. Specifically, it is [#f1]_

.. math::
    
    Basic(t)=\alpha|t|\log(f(t))+\beta e_t

For every term candidate *t* in the domain, where :math:`\alpha` and :math:`\beta` are configurable weights (1 and 3.5 by default), :math:`|t|` is the length in characters of the candidate *t*, *f(t)* is the frequency of the term in the domain, and :math:`e_t` is the number of term candidates containing *t*.

Example:

::
    
    from pyate import basic
    basic("I am a term extractor!")

ComboBasic
########

.. autofunction:: pyate.combo_basic

The ComboBasic algorithm is a generalization of Basic by adding a third parameter :math:`e_t'`, which is the number of candidate terms contained in the candidate `t`. Specifically,

.. math::

    ComboBasic(t)=\alpha |t|\log f(t) + \beta e_t + \gamma e_t'

where :math:`\gamma` is a third weight. The ``weights`` paramater is a list of floats elements such that :math:`weights=[\alpha,\beta,\gamma]`.

Example:

::
    
    from pyate import combo_basic
    combo_basic("I am a term extractor!")
    
C-Value
########

.. autofunction:: pyate.cvalues

TODO

Example:

::
    
    from pyate import cvalues
    cvalues("I am a term extractor!")

Term Extractor
########

.. autofunction:: pyate.term_extractor

TODO

Example:

::
    
    from pyate import term_extractor 
    term_extractor("I am a term extractor!")

Weirdness
########

.. autofunction:: pyate.weirdness

The Weirdness algorithm compares the term's frequency in a technical domain with its frequency in a general domain. Although it is one of the simplest algorithms to implement and understand, it is the least precise [#f1]_. The formula is as follows: [#f4]_

.. math::
    
    W(t)=\frac{f_{technical}(t)/|TechnicalCorpus|}{f_{general}(t)/|TargetCorpus|}

Where :math:`f_{technical}(t)` and :math:`f_{general}(t)` denote the frequency of candidate *t* in the technical and general domains, respectively, and *|TechnicalCorpus|* and *|GeneralCorpus|* denote the number of words in the two corpora, respectively.

Example:

::
    
    from pyate import weirdness 
    weirdness("I am a term extractor!")

References
########

.. [#f1] Astrakhantsev, N. (2018). ATR4S: toolkit with state-of-the-art automatic terms recognition methods in Scala. Language Resources and Evaluation, 52(3), 853-872, retrieved from https://arxiv.org/abs/1611.07804.

.. [#f2] Frantzi K.T., Ananiadou S., Tsujii J. (1998) The C-value/NC-value Method of Automatic Recognition for Multi-word Terms. In: Nikolaou C., Stephanidis C. (eds) Research and Advanced Technology for Digital Libraries. ECDL 1998. Lecture Notes in Computer Science, vol 1513. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-49653-X_35

.. [#f3] Sclano F., Velardi P. (2007) TermExtractor: a Web Application to Learn the Shared Terminology of Emergent Web Communities. In: Gonçalves R.J., Müller J.P., Mertins K., Zelm M. (eds) Enterprise Interoperability II. Springer, London. https://doi.org/10.1007/978-1-84628-858-6_32

.. [#f4] Astrakhantsev, N.A., Fedorenko, D.G. & Turdakov, D.Y. Methods for automatic term recognition in domain-specific text collections: A survey. Program Comput Soft 41, 336–349 (2015). https://doi.org/10.1134/S036176881506002X

.. |br| raw:: html

      <br>
