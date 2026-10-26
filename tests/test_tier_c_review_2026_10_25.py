"""Tier C review findings T84-1..T84-4 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "smolvlm_vision_language_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def test_t84_1_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "inferred to be unlikely, not" in md
    assert "it was never tuned on" not in md and "handwriting it was never trained on" not in md


def test_t84_2_no_cross_row_ranking_or_mechanism_claim():
    md = _markdown()
    assert "is the weakest of the three" not in md and "0.759" not in md and "0.797" not in md
    assert "learn the language of the corpus before they learn to read it" not in md
    assert "are not a ranking" in md


def test_t84_3_numbers_name_their_run():
    md = _markdown()
    assert "build record" not in md and "`5b2c5b22`" in md and "`3cded40b`" in md


def test_t84_4_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] A. Marafioti" in md and "`DIMER_ISOLATED_ENV`" in md and "NOTEBOOK_SPEC 2.0 §5" not in md
