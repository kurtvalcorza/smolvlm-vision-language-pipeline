"""Regression tests for the 2026-10-05 notebook review findings (SVL-M1..M4, SVL-m1..m3).

Every test needs only CI's dependencies and no model: the notebook's own cell sources are executed with stand-ins
where a model would be needed. Stand-in evidence is plumbing evidence, not model evidence.
"""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from smolvlm_vision_language_pipeline import SmolVLMPipeline
from smolvlm_vision_language_pipeline import pipeline as pl
from smolvlm_vision_language_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "smolvlm_vision_language_colab.ipynb"
LOCK = ROOT / "tutorials" / "requirements-colab.lock.txt"
PIPELINE = ROOT / "src" / "smolvlm_vision_language_pipeline" / "pipeline.py"
STEM = "smolvlm_vision_language"


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _markdown(notebook: dict) -> str:
    return "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")


# --- SVL-M1: no in-kernel install, no restart, idempotent Section 1 ------------------------------------------


def test_svl_m1_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "Restart the runtime" not in json.dumps(notebook)
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256", 'MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source


def test_svl_m1_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(notebook):
    source = _cell(notebook, "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    spec = importlib.util.spec_from_file_location("_review_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    build.check_lock(build._pins(ROOT), lock_text)


@pytest.mark.skipif(sys.platform != "linux", reason="the worker protocol uses Linux pass_fds (as in owlv2-detection-pipeline)")
def test_svl_m1_section_1_is_idempotent_and_keeps_the_live_worker(notebook, tmp_path, monkeypatch, capsys):
    """The real Section 1 cell, run twice with a stand-in interpreter: the matching environment is reused (no
    download) and the live worker — with every variable later cells created — is kept."""
    source = _cell(notebook, "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = types.SimpleNamespace(input_transformers_cleanup=[])
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]
    finally:
        runtime.close()


# --- SVL-M2: every adaptation starts from the pinned base -------------------------------------------------------


def test_svl_m2_adapt_and_load_artifact_restore_the_base_first():
    text = PIPELINE.read_text(encoding="utf-8")
    adapt = text[text.index("    def adapt(") : text.index("    def _remember_base(")]
    order = [adapt.index(m) for m in ("self._remember_base(model, names)", "restored = self.restore_base()", "frozen_state = {", '"note": "frozen model"', "for epoch in range(1, epochs + 1):")]
    assert order == sorted(order)
    assert "model.load_state_dict(previous_state, strict=False)" in adapt  # a failed call leaves the weights as before
    load = text[text.index("    def load_artifact(") : text.index("    def from_artifact(")]
    assert load.index("self.restore_base()") < load.index("model.load_state_dict({k: v.to(")


class _Tensor(np.ndarray):
    def detach(self):
        return self

    def clone(self):
        return self.copy().view(_Tensor)


class _FakeModel:
    """Stand-in model (numpy arrays, not torch) with the state-dict surface restore_base uses."""

    def __init__(self):
        self.params = {"model.text_model.layers.29.w": np.zeros(3).view(_Tensor), "model.text_model.norm.w": np.ones(3).view(_Tensor)}

    def state_dict(self):
        return dict(self.params)

    def load_state_dict(self, state, strict=True):
        for key, value in state.items():
            self.params[key][...] = value

    def eval(self):
        return self


def test_svl_m2_restore_base_undoes_every_earlier_change_stand_in():
    model = _FakeModel()
    pipe = SmolVLMPipeline(lambda *a: {}, _model=model, _processor=object())
    names = sorted(model.params)
    pipe._remember_base(model, names)
    model.params[names[0]] += 5  # an earlier adaptation
    pipe.adapter = {"best_epoch": 3}
    assert pipe.restore_base() == [names[0]] and pipe.adapter is None  # only the changed tensor is reported
    assert pipe.restore_base() == []  # nothing differs from the base any more (a fresh adapt says "pinned base")
    assert (model.params[names[0]] == 0).all() and (model.params[names[1]] == 1).all()
    pipe._remember_base(model, names)  # a second call keeps the first (base) values
    model.params[names[1]] += 2
    pipe.restore_base()
    assert (model.params[names[1]] == 1).all()


def test_svl_m2_byod_rerun_restores_the_base_and_the_experiment_has_its_own_pipeline(notebook):
    section_4 = _cell(notebook, "USE_BYOD = False")
    assert section_4.index("pipe.restore_base()") < section_4.index("if USE_BYOD:")
    experiment = _cell(notebook, "RUN_EXPERIMENT = False")
    assert "EXPERIMENT_PROMPT = " in experiment and "# @param" in experiment.split("EXPERIMENT_PROMPT = ", 1)[1].splitlines()[0]
    assert "experiment_pipe = SmolVLMPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)" in experiment
    assert f"Path('outputs/{STEM}_experiment')" in experiment
    assert "raise RuntimeError(f'the experiment changed a default export: {unchanged}')" in experiment
    assert not re.search(r"(?<!experiment_)pipe\.adapt\(", experiment)
    assert "**Predict → Change one thing → Run → Observe → Explain**" in _markdown(notebook)
    assert "they do not affect the default path" not in _markdown(notebook)
    assert "TRANSCRIBE_PROMPT`'s wording in the carried module" not in _markdown(notebook)  # no experiment edits a carried cell


# --- SVL-M3: quality outcomes are reported verdicts -------------------------------------------------------------


def test_svl_m3_no_quality_assert_remains(notebook):
    code = "\n".join(c["source"] for c in _code_cells(notebook) if not c["metadata"].get("dimer", {}).get("embedded_module"))
    assert re.findall(r"(?m)^\s*assert .*$", code) == ["assert parity['identical_lines'] == parity['of']"]


def _rates(cer, n=4):
    return {"cer": cer, "wer": cer, "cer_macro": cer, "exact_match": 0.0, "n": n, "ref_chars": 40, "hyp_chars": 30, "truncated": 0, "rows": [{"id": f"r{i}", "cer": cer} for i in range(n)], "hypotheses": ["x"] * n, "doctags": ["<doctag>"] * n}


def test_svl_m3_a_worse_adapted_model_is_recorded_and_does_not_stop_the_notebook(notebook, tmp_path, monkeypatch):
    """The notebook's Section 8 cell with stand-in rates: adapted CER above the frozen CER completes, writes the
    evaluation report and records the verdict (stand-in evidence, no model)."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    pipe = types.SimpleNamespace(evaluate=lambda *a, **k: _rates(1.2))
    ns = {
        "pipe": pipe, "test_records": [], "val_records": [], "TRANSCRIBE_PROMPT": "p", "LINE_MAX_NEW_TOKENS": 160, "EVAL_BATCH_SIZE": 8,
        "METRICS": ("cer", "wer", "cer_macro", "exact_match"), "baseline_empty": _rates(1.0), "baseline_constant": _rates(0.95), "frozen_test": _rates(0.9),
        "MODEL_ID": "stand-in", "MODEL_REVISION": "0" * 40, "MODEL_KEY": "stand-in", "data_source": "stand-in",
        "dataset_manifests": {"test": {"digest": "d"}}, "disjoint": {}, "adapt_result": {"history": []}, "adapt_seconds": 0.0, "json": json,
    }
    exec(_cell(notebook, "adapted_test = pipe.evaluate("), ns)
    report = json.loads((tmp_path / "outputs" / f"{STEM}_evaluation_report.json").read_text(encoding="utf-8"))
    assert report["comparison"]["verdicts"] == {"adapted_vs_frozen_cer": "worse", "adapted_below_empty_baseline": False}


# --- SVL-M4: guided layer and infrastructure labelling ----------------------------------------------------------


def test_svl_m4_guided_layer_is_present(notebook):
    markdown = _markdown(notebook)
    for heading in ("**Who this notebook is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "## 10. Change one thing", "**Learner:**"):
        assert heading in markdown, heading
    assert markdown.count("**Predict") >= 7
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= 7
    assert markdown.count("**What to notice:**") >= 6


def test_svl_m4_infrastructure_cells_are_labelled_and_collapsed(notebook):
    infra = [c for c in _code_cells(notebook) if c["metadata"].get("cellView") == "form"]
    assert len([c for c in infra if c["metadata"].get("dimer", {}).get("embedded_module")]) == 3
    titled = [c["source"].splitlines()[0] for c in infra if not c["metadata"].get("dimer")]
    assert len(titled) == 3 and all(t.startswith("# @title Infrastructure:") for t in titled), titled


def test_svl_m4_no_template_placeholders_leak(notebook):
    learner = "\n".join(c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module"))
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in learner, leftover
    assert "}}" not in _markdown(notebook)


# --- SVL-m1: BYOD contract --------------------------------------------------------------------------------------


def _png(i: int) -> bytes:
    image = Image.new("RGB", (320, 48), "white")
    ImageDraw.Draw(image).text((8, 16), f"ligne numero {i}", fill="black")
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _zip(path: Path, n: int, extra: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("transcripts.csv", "file,text\n" + "".join(f"img{i}.png,ligne numero {i}\n" for i in range(n)))
        for i in range(n):
            archive.writestr(f"img{i}.png", _png(i))
        for name, data in (extra or {}).items():
            archive.writestr(name, data)
    return path


def test_svl_m1_stated_minimum_is_what_the_split_accepts(tmp_path):
    assert sm.min_byod_records() == {"total": 12, "train": 8, "validation": 2, "test": 2}
    split = sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "ok.zip", 12)), seed=42)
    assert {k: len(v) for k, v in split.items()} == {"test": 2, "validation": 2, "train": 8}
    with pytest.raises(ValueError, match=r"training split with 7 image\(s\).*supply at least 12 distinct images"):
        sm.split_dataset(sm.load_byod_dataset(_zip(tmp_path / "small.zip", 11)), seed=42)


def test_svl_m1_same_named_images_are_refused_and_litter_is_skipped(tmp_path):
    with pytest.raises(ValueError, match=r"two images share the file name 'img0.png' \(img0.png and b/img0.png\)"):
        sm.load_byod_dataset(_zip(tmp_path / "dup.zip", 12, {"b/img0.png": _png(99)}))
    report: dict = {}
    records = sm.load_byod_dataset(_zip(tmp_path / "mac.zip", 12, {"__MACOSX/._img0.png": b"\0", "README.txt": b"notes", ".DS_Store": b""}), report=report)
    assert len(records) == 12 and report["skipped"] == [".DS_Store", "README.txt", "__MACOSX/._img0.png"]
    with zipfile.ZipFile(tmp_path / "missing.zip", "w") as archive:
        archive.writestr("transcripts.csv", "file,text\nimg0.png,a\nimg1.png,b\n")
        archive.writestr("img0.png", _png(0))
    with pytest.raises(ValueError, match=r"transcripts.csv line 3 \(file 'img1.png'\): names a missing image"):
        sm.load_byod_dataset(tmp_path / "missing.zip")


def _section_4(notebook: dict, path: str) -> str:
    source = _cell(notebook, "USE_BYOD = False")
    source = source.replace("USE_BYOD = False  # @param", "USE_BYOD = True  # @param", 1)
    return source.replace("BYOD_PATH = ''  # @param", f"BYOD_PATH = {path!r}  # @param", 1)


def _section_4_namespace(restored: list) -> dict:
    ns = {k: getattr(pl, k) for k in dir(pl) if not k.startswith("__")}
    ns.update({k: getattr(sm, k) for k in dir(sm) if not k.startswith("__")})
    pipe = types.SimpleNamespace(adapter={"best_epoch": 8}, restore_base=lambda: restored.append(True) or ["a", "b"])
    ns.update({"os": __import__("os"), "Path": Path, "pipe": pipe, "__name__": "__main__"})
    return ns


def test_svl_m1_byod_path_runs_section_4_outside_colab_from_the_base(notebook, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    _zip(tmp_path / "mine.zip", 12, {"__MACOSX/._img0.png": b"\0"})
    restored: list = []
    ns = _section_4_namespace(restored)
    exec(_section_4(notebook, "mine.zip"), ns)
    out = capsys.readouterr().out
    assert restored == [True], "a BYOD re-run must put the model back to the pinned base first"
    assert ns["raw_rows"]["byod"] == 12 and ns["raw_rows"]["skipped_files"] == ["__MACOSX/._img0.png"] and ns["raw_rows"]["effective_minimum"] == 12
    assert ns["disjoint"] == {"test": 2, "validation": 2, "train": 8}
    assert "only 2 held-out test lines" in out


def test_svl_m1_upload_outside_colab_cancelled_and_bad_path_are_explained(notebook, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "google", None)
    with pytest.raises(RuntimeError, match="upload dialog exists only in Google Colab"):
        exec(_section_4(notebook, ""), _section_4_namespace([]))
    with pytest.raises(FileNotFoundError, match="BYOD_PATH 'nowhere.zip' does not exist"):
        exec(_section_4(notebook, "nowhere.zip"), _section_4_namespace([]))
    for uploaded, message in (({}, "received 0"), ({"a.zip": b"", "b.zip": b""}, "received 2")):
        google, colab, files = (types.ModuleType(n) for n in ("google", "google.colab", "google.colab.files"))
        files.upload = lambda uploaded=uploaded: uploaded
        colab.files, google.colab = files, colab
        for name, module in (("google", google), ("google.colab", colab), ("google.colab.files", files)):
            monkeypatch.setitem(sys.modules, name, module)
        with pytest.raises(ValueError, match=message):
            exec(_section_4(notebook, ""), _section_4_namespace([]))


# --- SVL-m2 / SVL-m3: registry row and split disclosure ---------------------------------------------------------


def test_svl_m2_section_7_prose_matches_the_code_and_the_registry_names_the_drawing(notebook):
    text = json.dumps(notebook)
    assert "four trainable" not in text and "of the characters right" not in text
    section_7 = _markdown(notebook)[_markdown(notebook).index("## 7. ") : _markdown(notebook).index("## 8. ")]
    assert "only the eight trainable layers" in section_7 and "1 − CER is not the share of characters read correctly" in section_7
    assert "drawing = Image.new('RGB', (384, 384)" in _cell(notebook, "drawing = Image.new(")
    registry = (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    row = next(line for line in registry.splitlines() if line.startswith("| `smolvlm_vision_language_colab.ipynb`"))
    assert "384 × 384 drawing" in row and "512×512" not in row


def test_svl_m3_split_boundary_is_disclosed_in_sections_4_and_8(notebook):
    markdown = _markdown(notebook)
    section_4 = markdown[markdown.index("## 4. ") : markdown.index("## 5. ")]
    section_8 = markdown[markdown.index("## 8. ") : markdown.index("## 9. ")]
    for text in (section_4, section_8):
        assert "share pages and writers" in text and "optimistic" in text


def test_svl_s4_declares_notebook_spec_2_2():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.5"
    assert nb["metadata"]["dimer"]["notebook_kind"] == "TUTORIAL"
