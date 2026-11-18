"""Routers. Each learned router outputs p(Haiku answers correctly) for a prompt.

Routing rule (applied in evaluate.py): send to Haiku if score >= tau, else Sonnet.
Fixed routers (always_haiku, always_sonnet, random, oracle) need no scores and
are handled directly in the evaluation, since their curves are known exactly.
"""

from .knn import KNNRouter
from .logreg import LogRegRouter

LEARNED = {"knn": KNNRouter, "logreg": LogRegRouter}

__all__ = ["KNNRouter", "LogRegRouter", "LEARNED"]


def get(name: str):
    if name == "distilbert":
        from .distilbert import DistilBertRouter

        return DistilBertRouter()
    return LEARNED[name]()
