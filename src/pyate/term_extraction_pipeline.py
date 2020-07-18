from collections import defaultdict
from typing import Callable

import pandas as pd
from spacy.tokens import Doc

from .combo_basic import combo_basic
from .term_extraction import TermExtraction


class TermExtractionPipeline:
    def __init__(
        self,
        func: Callable[..., pd.Series] = combo_basic,
        force: bool = True,
        *args,
        **kwargs
    ) -> None:
        self.func = func
        self.args = args
        self.kwargs = kwargs
        Doc.set_extension(self.func.__name__, default=None, force=force)

    def __call__(self, doc: Doc):
        term_counter = defaultdict(int)

        def add_to_counter(matcher, doc, i, matches) -> Doc:
            match_id, start, end = matches[i]
            candidate = str(doc[start:end])
            if TermExtraction.word_length(candidate) <= TermExtraction.MAX_WORD_LENGTH:
                term_counter[candidate] += 1

        for i, pattern in enumerate(TermExtraction.patterns):
            TermExtraction.matcher.add("term{}".format(i), add_to_counter, pattern)
        matches = TermExtraction.matcher(doc)
        terms = self.func(
            str(doc),
            technical_counts=pd.Series(term_counter),
            *self.args,
            **self.kwargs
        )
        setattr(doc._, self.func.__name__, terms)
        return doc
