"""Security layer: prompt-injection detection, prompt assembly, output vetting.

Retrieved documents are **untrusted data**. Nothing in this package ever treats
context text as an instruction; it is fenced, tagged and passed to the model as
reference material only.
"""

from __future__ import annotations

from app.security.classification import ClassificationResult, QueryClassifier
from app.security.context import ContextBlock, build_context, neutralise_untrusted_text
from app.security.injection import (
    HARD_TIER,
    KEYWORD_TIER,
    InjectionDetector,
    InjectionMatch,
    InjectionVerdict,
)
from app.security.output_validator import OutputValidationResult, OutputValidator
from app.security.prompt_builder import SYSTEM_RULES, PromptBuilder, build_prompt

__all__ = [
    "HARD_TIER",
    "KEYWORD_TIER",
    "SYSTEM_RULES",
    "ClassificationResult",
    "ContextBlock",
    "InjectionDetector",
    "InjectionMatch",
    "InjectionVerdict",
    "OutputValidationResult",
    "OutputValidator",
    "PromptBuilder",
    "QueryClassifier",
    "build_context",
    "build_prompt",
    "neutralise_untrusted_text",
]
