*****
Models
*****

Though this model was originally intended for symbolic AI algorithms (non-machine learning), I realized a spaCy model on term extraction can reach significantly higher performance, and thus decided to include the model here.

For a comparison with the symbolic AI algorithms, see Precision. Note that only the F-Score, accuracy and precision was taken here yet for the model, but for the algorithms the AvP was taken so directly comparing the metrics would not really make sense.

+--------------------------------------------------------------------------------------------+-------------+---------------+------------+
| URL                                                                                        | F-Score (%) | Precision (%) | Recall (%) |
+--------------------------------------------------------------------------------------------+-------------+---------------+------------+
| https://github.com/kevinlu1248/pyate/releases/download/v0.4.2/en_acl_terms_sm-2.0.4.tar.gz | 94.71       | 95.41         | 95.41      |
+--------------------------------------------------------------------------------------------+-------------+---------------+------------+

The model was trained and evaluated on the ACL dataset, which is a computer science oriented dataset where the terms are manually picked. This has not yet been tested on other fields yet, however.

This model does not come with PyATE. To install, run

::

    pip install https://github.com/kevinlu1248/pyate/releases/download/v0.4.2/en_acl_terms_sm-2.0.4.tar.gz

To extract terms with the model,

::

    import spacy

    nlp = spacy.load("en_acl_terms_sm")
    doc = nlp("Hello world, I am a term extraction algorithm.")
    print(doc.ents)
    """
    (term extraction, algorithm)
    """
