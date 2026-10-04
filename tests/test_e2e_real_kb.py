"""Pytest entry point for the real knowledge-base end-to-end evaluation.

The evaluation itself lives in ``scripts/e2e_evaluation.py`` because loading the
embedding model and rebuilding the FAISS index takes several seconds, and a
report is more useful than a wall of dots. This module makes it runnable from the
normal test command without making the default suite slow:

    AHMED_RAG_E2E=1 pytest tests/test_e2e_real_kb.py -v

Without that environment variable the whole module is skipped, so ``pytest`` on a
laptop stays fast and hermetic while the deeper check stays one command away.

Nothing here touches the network: the runner pins ``HF_HUB_OFFLINE`` and
``TRANSFORMERS_OFFLINE`` before the embedder is imported, and the model is the one
already cached locally. No language model is involved at any point.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "e2e_evaluation.py"

pytestmark = pytest.mark.skipif(
    os.environ.get("AHMED_RAG_E2E") != "1",
    reason="set AHMED_RAG_E2E=1 to run the real knowledge-base evaluation",
)


def test_the_runner_script_exists() -> None:
    """The documented command must not rot."""
    assert RUNNER.is_file()


def test_the_real_knowledge_base_evaluation_passes() -> None:
    """Run the whole evaluation in a subprocess and surface its report.

    A subprocess rather than an in-process call so the runner's environment
    pinning is exercised exactly as a user would experience it, and so a crash
    in the embedding stack cannot take the test session down with it.
    """
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--quiet", "--json", str(ROOT / ".e2e-report.json")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    report = completed.stdout
    print(report)
    assert completed.returncode == 0, (
        f"real-KB evaluation failed (exit {completed.returncode})\n"
        f"--- stdout ---\n{report}\n--- stderr ---\n{completed.stderr[-2000:]}"
    )

    # The headline numbers are asserted as well, so a future change cannot make
    # the evaluation pass by testing nothing.
    assert "passed                  : 56" in report
    assert "failed                  : 0" in report
    assert "grounding violations    : 0" in report
    assert "source-retrieval accuracy: 100.0%" in report
    assert "documents cited         : 12/12" in report
    assert "llm used anywhere       : False" in report


def test_the_evaluation_is_deterministic() -> None:
    """Two runs over an unchanged corpus must produce identical answers.

    Determinism is a property the whole design rests on: there is no model to
    sample and no tuning knob that varies between processes. If this fails, the
    pipeline has picked up something non-reproducible -- a hash seed, a set
    iteration order, an mtime dependency in the index manifest.
    """
    outputs = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, str(RUNNER), "--quiet"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        outputs.append(completed.stdout)

    # The first run may rebuild the index, which legitimately changes the
    # "reused"/"rebuilt" wording only if the header were printed; --quiet omits
    # it, so the reports must be byte-identical.
    assert outputs[0] == outputs[1]
