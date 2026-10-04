"""Deterministic, model-free answering.

The package is the replacement for the LLM generation step:

* :mod:`app.services.answering.intents` -- the intent taxonomy and the scored
  router that maps a question to one or more corpus documents,
* :mod:`app.services.answering.corpus` -- which document backs which domain,
* :mod:`app.services.answering.domains` -- project-name resolution,
* :mod:`app.services.answering.composer` -- grounded extraction and assembly.
"""

from app.services.answering.composer import AnswerComposer, ComposedAnswer
from app.services.answering.corpus import CorpusProfile, build_profile
from app.services.answering.domains import detect_project
from app.services.answering.intents import (
    AnswerShape,
    Domain,
    Intent,
    Routing,
    detect_domain,
    route,
)

__all__ = [
    "AnswerComposer",
    "AnswerShape",
    "ComposedAnswer",
    "CorpusProfile",
    "Domain",
    "Intent",
    "Routing",
    "build_profile",
    "detect_domain",
    "detect_project",
    "route",
]
