"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E instruction-adaptation workflow: the pinned HuggingFaceTB/SmolVLM-500M-Instruct
snapshot is digest-verified and loaded, 800 MIT-licensed Belfort handwritten line images with their transcripts are
fetched as eight digest-pinned parquet row groups over HTTPS range requests, the records are validated and split by
line, two prompts are asked of a synthetic drawing through the inference contract, the frozen model's transcription
instruction is scored over the held-out lines (character and word error rates) beside an empty-string and a
constant-transcript baseline, a bounded fine-tuning of the last eight decoder layers runs on cached prefix hidden
states with validation-CER epoch selection, the held-out split is scored again, six held-out lines and the two
drawing prompts are re-run with the adapted model, and the adapter is exported and reloaded.

2026-10-05 review fix cycle (SVL-M1..M4, SVL-m1..m3): generator /2.2 keys ``isolated_runtime`` (nothing installed into
the kernel; no restart), ``infrastructure_labels`` and ``guided``; quality outcomes are reported verdicts;
``SmolVLMPipeline.adapt`` restarts from the pinned base and Section 4 restores it before a BYOD re-run; the prompt
experiment is a form field in its own frozen pipeline; BYOD takes a path or one uploaded zip; the Section 7 prose is
corrected and the split's shared-hand boundary disclosed.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "smolvlm_vision_language_pipeline",
    "repo_name": "smolvlm-vision-language-pipeline",
    "stem": "smolvlm_vision_language",
    "notebook_name": "smolvlm_vision_language_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (bioclip2-biodiversity-pipeline): managed CPython, a size- and
    # SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
    "pipeline_class": "SmolVLMPipeline",
    "weights_key": "smolvlm-500m-instruct",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "SmolVLM-500M-Instruct — DIMER E2E handwritten-line transcription adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/tutorials/smolvlm_vision_language_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HuggingFaceTB%2FSmolVLM--500M--Instruct-ffcc4d?style=flat",
            "https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-huggingface%2Fsmollm-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/huggingface/smollm",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2504.05299-b31b1b.svg", "https://arxiv.org/abs/2504.05299"),
    ],
    "capability": "image + text to text generation (captioning and visual question answering) and bounded supervised fine-tuning of one instruction — transcribing a text line — on transcribed lines, using the pinned SmolVLM-500M-Instruct weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (nothing is "
        "installed into the notebook's own Python, so no restart is needed and Run all completes in one pass), stages and digest-verifies the "
        "pinned `HuggingFaceTB/SmolVLM-500M-Instruct` snapshot (a 1.02 GB `model.safetensors`; no pickle is opened anywhere), "
        "fetches the first eight row groups of the Belfort-line test shard from the Hugging Face Hub at an immutable revision "
        "with HTTPS range requests (about 44 MB; each row group refused on any SHA-256 or byte-total mismatch), validates the "
        "800 line records and splits them by line into 600 / 60 / 140, asks two prompts of a synthetic drawing through the "
        "inference contract with a combined input manifest and a rejection probe, scores the frozen model's transcription "
        "instruction over the 140 held-out lines (character and word error rates) beside an empty-string and a "
        "constant-transcript baseline, runs a bounded fine-tuning of the last eight decoder layers on cached prefix hidden "
        "states with validation-CER epoch selection, scores the held-out lines again, re-runs six held-out lines and the two "
        "drawing prompts with the adapted model, exports the adapter as safetensors with a manifest, and reloads that "
        "artifact into a fresh pipeline to verify transcript parity. The default path needs no repository clone, no DIMER "
        "worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.5 §5). On a Tesla T4 "
        "the default path took about 32 minutes of cell time (Kaggle T4 run of blob `5b2c5b22`, 2026-09-21: eight epochs 1452 s, frozen scoring of "
        "140 lines 157 s; the Colab T4 run of blob `3cded40b`, 2026-10-10, took 1871.1 s of session wall); a CUDA runtime is used automatically when present, and **a CPU runtime is not "
        "practical for the default path** (greedy decoding of some 800 lines plus 600 cached forwards of a 507M-parameter "
        "model)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and either set `BYOD_PATH` to a zip or folder "
        "in the runtime (Colab, Kaggle or Jupyter) or leave it empty to upload one zip in Colab, then choose **Run after** from "
        "that cell. The zip or folder holds line images plus a `transcripts.csv` (`file`, `text`, optional `id`; one row per "
        "image). After the split the training split needs eight images and the validation split one, so the **effective "
        "minimum is 12 distinct images** (split 8 / 2 / 2); with so few, the held-out rates rest on two lines and the cell says "
        "so. Section 4 first puts the model back to the pinned base, so Sections 5 and 6 score the frozen model on your lines. "
        "The records pass through the same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact "
        "export and reload-parity cells as the Belfort sample. Uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "intro": (
        "SmolVLM-500M-Instruct is an Idefics3-architecture vision-language model: a SigLIP vision encoder turns each 512-px "
        "tile of the image (the longest edge resized to 2048 px, aspect ratio preserved, plus one global view) into 64 visual "
        "tokens through a pixel-shuffle connector, and a 32-layer SmolLM2 decoder reads them with the chat-templated user turn "
        "and generates the assistant turn (507,482,304 parameters in all, published under the **Apache-2.0** licence). "
        "Decoding is **greedy** by default (`do_sample=False`), deterministic on a fixed device and dtype. The output is "
        "**generated text with no score**: a fluent answer is not evidence that it is right.\n\n"
        "What this notebook adds to inference is **adaptation of one instruction on transcribed lines**. The instruction is "
        "`Transcribe the handwritten text in this image.`; the lines are nineteenth-century French council minutes in cursive "
        "— the Belfort-line dataset — far outside the model's instruction-tuning distribution, and on them the frozen model "
        "reads a word here and there and answers in English about the rest: a character error rate of **1.809** on "
        "the 140 held-out lines (the recorded runs' Tesla T4 figure). So the honest question is narrow: does a bounded "
        "fine-tuning of the last eight decoder layers on 600 transcribed lines move the held-out **CER** and **WER** on a "
        "line-disjoint test split past two **non-adapted baselines** and the frozen model — and what does it do to the other "
        "prompts the same decoder answers? The two sibling rows that adapted GOT-OCR 2.0 and Florence-2 on the same split "
        "squash a line into a square; this model's tiling keeps the line's aspect ratio, which is the comparison the fleet "
        "wanted. Nothing here is a claim about your documents or your script: it is one seeded split of one small labelled "
        "set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 13-file manifest with the tokenizer and chat "
        "template files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files "
        "before the processor or the model is constructed. The pipeline loads the checkpoint in **float32 on every device** "
        "(the weights are stored as bfloat16): the adapter is trained in float32 and overlays without a cast, and CPU, "
        "Tesla-class and consumer GPUs then run the same arithmetic."
    ),
    "guided": {
        "opening": [
            (
                "**Who this notebook is for.** A learner who knows basic Python, has used Colab or Jupyter, and wants to see what a small instruction-tuned vision-language model does with handwriting outside its documented training data, how to measure that honestly with character and word error rates, and how to adapt one instruction to a small transcribed set without fooling themselves — including what the adaptation costs on other prompts. No prior experience with SmolVLM, vision-language models or fine-tuning is assumed; each term is explained where it first matters and again in the **Glossary** at the end. A CUDA GPU runtime (T4) is required in practice: the default path takes about 32 minutes there and hours on CPU.\n\n**Input → Model → Output.**\n\n| | Image + prompt → text | Line transcription (scored) | Bounded fine-tuning |\n|---|---|---|---|\n| Input | one image and one chat-templated user prompt | one handwritten line image and `TRANSCRIBE_PROMPT` | transcribed line images (600 training and 60 validation in the sample) |\n| Model | SigLIP vision encoder, pixel-shuffle connector and a 32-layer decoder, greedy decoding | the same model | the last eight decoder layers and the final norm, trained with the causal language-model loss on the transcript; everything else frozen |\n| Output | the generated assistant turn, `new_tokens` and a `truncated` flag — no score | the transcript, scored by CER and WER against the reference | a 315 MB safetensors adapter, and held-out CER / WER / exact match |\n\n**How to use this notebook.** Choose a GPU runtime (**Runtime → Change runtime type → T4 GPU**), then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed. Sections 1–3 are **infrastructure** — the isolated environment, the carried package and the model snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it come **What to notice** and a collapsible **Check your reasoning** with a worked answer from the recorded run (the Kaggle T4 run of blob `5b2c5b22`, 21 September 2026, whose figures the Colab T4 run of blob `3cded40b` on 10 October 2026 repeated; neither re-run for this revision, and another GPU can move the last digits). Section 10 is a **change-one-thing experiment** on the instruction's wording, off by default. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Writing your predictions down is optional.\n\n**Roadmap:** 1–3 infrastructure → 4 the Belfort lines, validation and the split *(evaluation practice)* → 5 the inference contract on a drawing *(core concept: generated text has no score)* → 6 two baselines and the frozen model *(evaluation practice)* → 7 bounded fine-tuning *(core concept: what is trained)* → 8 held-out evaluation → 9 the lines by eye, the drawing again, export and reload *(engineering)* → 10 change one thing: the prompt (optional) → conclude."
            )
        ]
    },
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled line set with its transcripts, validate it and split it by line "
        "image-disjoint (and say what that split does not keep apart); ask two prompts of a synthetic drawing through the public API and read the output contract correctly "
        "(generated text, no score, a `truncated` flag, no intrinsic metric for open-ended answers); measure the frozen "
        "transcription instruction's corpus CER and WER beside two non-adapted baselines; run a bounded fine-tuning with the "
        "model's own instruction-tuning loss, explicit hyperparameters and validation-based epoch selection; evaluate on a "
        "line-disjoint test split; look at the adapted transcripts next to the frozen ones and the references, and at what the "
        "other prompts do after the shared decoder was tuned; and export a safetensors adapter that reloads against the pinned "
        "base with verified parity."
    ),
    "exclusions": (
        "multi-image and multi-turn conversations (`MAX_IMAGES = 1`, one user turn), video, sampling-based decoding in the "
        "corpus stages, fine-tuning of the vision encoder, the connector, the embeddings, the output head or the first 24 "
        "decoder layers, fine-tuning of any instruction but the transcription one, a metric for open-ended answers "
        "(captions and free-form VQA stay `not-measurable`), evaluation on an OCR or VQA benchmark proper (only one seeded "
        "800-line sample is scored here), and any claim that French cursive minutes stand in for your documents. The "
        "repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and Colab or Jupyter familiarity; no prior experience with SmolVLM, vision-language models or fine-tuning. The notebook explains the chat template and the assistant turn, greedy decoding, micro and macro CER/WER, the medoid baseline, the frozen-prefix cache and the adapter where they are first used; the Glossary repeats them.",
        "- **Overlap with the training data:** SmolVLM-500M-Instruct [1] was trained on The Cauldron and Docmatix (model card). "
        "Neither card names the Belfort lines [5] (Zenodo 8041668, public since 2023), and The Cauldron's subset list (checked "
        "2026-10-25) does not include them, so no overlap with the test lines is documented; it is inferred to be unlikely, not "
        "verified, because the PDF-derived Docmatix pages are not enumerated line by line. The Cauldron does include IAM English "
        "handwriting lines, so the model has seen some handwriting, though no French cursive is documented.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime with a CUDA GPU (Google Colab or Kaggle GPU). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the kernel's own Python version does not matter and nothing is installed into it. The default path uses CUDA automatically when present. Generation is batched for the corpus stages — the prompt length varies with the image's tile count, so batches are left-padded — and the Kaggle T4 run of an earlier blob (`5b2c5b22`, 2026-09-21) measured 157 s to score 140 lines and 1452 s for the eight epochs (caching the prefix hidden states for 600 lines took 427 s); the Colab T4 run of blob `3cded40b` (2026-10-10) took 1871.1 s of session wall in all, about 32 minutes of cell time for the whole path; a CPU runtime would take hours. The pinned `torch==2.14.0` install and the 1.02 GB checkpoint are the large downloads of the run; the row groups are about 44 MB.",
        "- **Knowledge:** basic Python and PIL; what a chat template, a user turn and greedy decoding are; what character and word error rate measure and why they are not capped at 1; why a self-drawn image is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, text}` — `image` a PIL image (or a file decodable by Pillow) with sides within 1..16,384 px and at most 4096² pixels, `text` its transcript (1..512 characters after whitespace runs are collapsed; case and punctuation kept). Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images plus a `transcripts.csv` in the layout named above; after the split the training split needs eight images and the validation split one, so the effective BYOD minimum is **12 distinct images** (`min_byod_records()` computes it). Images are matched by file name: two images with the same name in different folders are refused, and `__MACOSX` entries, hidden files and non-image files are skipped and listed. Every refusal names the `transcripts.csv` line, the file and the rule.",
        "- **Recorded runs:** numbers quoted from \"the recorded runs\" come from the Kaggle T4 run of blob `5b2c5b22` "
        "(2026-09-21) and the Colab T4 run of blob `3cded40b` (commit `fc5b53c`, 2026-10-10), which printed the same figures; neither was re-run "
        "for this revision, and your printed output is the result.",
        "- **Validation is structural, not semantic:** every image is decoded and every transcript checked for length, but nothing checks that a transcript says what its image shows — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads eight row groups of `default/test/0000.parquet` from `https://huggingface.co/datasets/Teklia/Belfort-line/resolve/<revision>/` at the immutable parquet-conversion revision `c4a74bbd…` with HTTPS range requests (the parquet footer plus about 44 MB of row-group bytes out of a 210 MB shard), each row group pinned by SHA-256 and byte total in the carried `samples.py` and refused on any mismatch. Belfort-line is published under the MIT licence (Teklia; Tarride et al. 2023); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Belfort lines, the transcripts and the split\n\n"
                "`fetch_corpus` returns the eight pinned row groups from the cache under `weights/belfort/` or the Hub at the "
                "pinned parquet-conversion revision — `pyarrow` reads the shard's footer and exactly those row groups over "
                "HTTPS range requests; every cached file is re-hashed and every fetched row group refused on any SHA-256 or "
                "byte-total mismatch — and `read_corpus` turns each row into a record: the line image (128 px tall, 145 to "
                "8,956 px wide) and its crowdsourced transcript with whitespace runs collapsed. `build_sample_dataset` draws a "
                "seeded line-level split (600 / 60 / 140). `validate_dataset` then checks every record against the contract, "
                "`check_split_disjoint` asserts no image (by decoded-pixel digest) is shared, and the training split's summary "
                "table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 800 lines and 33,117 reference characters, three digests, and four refusal probes — a duplicate id, "
                "an empty transcript, an image above the side ceiling, and a dataset too small to use — each rejected before "
                "the model does anything.\n\n"
                "*Evaluation practice.* **What the split preserves, and what it does not.** The 800 lines are consecutive rows "
                "of one council's minutes, and the split is made **by line**: no image is in two splits (the pixel-digest check "
                "proves that), but lines cut from the same page share a hand, so test lines can share pages and writers with "
                "training lines. The held-out rates in Section 8 are therefore a within-collection estimate and are optimistic "
                "for a hand the model has never seen.\n\n"
                "**Bring your own data (optional):** set `USE_BYOD = True` and either `BYOD_PATH` (a zip or a folder holding "
                "`transcripts.csv` and the line images, as a path in this runtime — this works on Colab, Kaggle and Jupyter) "
                "or leave `BYOD_PATH` empty to upload exactly one zip through the Colab dialog; then choose **Run after** from "
                "this cell. This cell first puts the model back to the pinned base, so Sections 5 and 6 score the frozen model "
                "on your lines, not the Belfort-adapted one. The effective minimum is 12 distinct images. A refused input stops the cell with one line, `Refused: <file>: <reason and fix>`, and "
                "removes the earlier run's results from the session and this notebook's exports from `outputs/`, so nothing from that "
                "run can pass as a result of your lines.\n\n"
                "**Predict before running:** the split draws 600 / 60 / 140 lines at random from 800 consecutive lines of one "
                "volume. Can a test line share a writer with a training line — and would that make the test score higher or "
                "lower than on a new writer?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if pipe.adapter is not None:\n"
                "    # A re-run after Section 7 (BYOD, or a new split): Sections 5 and 6 must score the frozen model, not the adapted one.\n"
                "    print({{'restored_pinned_base': len(pipe.restore_base()), 'note': 'the adapted weights were removed; Sections 5-6 score the frozen model again'}})\n"
                'import zipfile\n\n\n'
                'class BYODRefusal(ValueError):\n'
                '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
                '    dimer_refusal = True\n\n'
                '    def __init__(self, file_name, reason):\n'
                '        super().__init__(f"{{file_name}}: {{reason}}")\n'
                '        self.file_name = file_name\n'
                '        self.reason = reason\n\n'
                '    def _render_traceback_(self):\n'
                '        return [f"Refused: {{self}}"]\n\n\n'
                '# The results an earlier run of this cell and the sections after it left in this session. A refused input removes them,\n'
                "# and this notebook's exports in outputs/, so none of them can pass as a result of the refused input (DAT20).\n"
                'RUN_RESULT_NAMES = (\n'
                "    'byod_report', 'records', 'splits', 'data_source', 'raw_rows', 't0', 'corpus_groups', 'corpus', 'dataset_manifests', 'disjoint', 'train_records',\n"
                "    'val_records', 'test_records', 'manifest', 'example', 'probes', 'probe', 'drawing', 'draw', 'drawing_name', 'drawing_sha256', 'manifests',\n"
                "    'input_manifest', 'frozen_answers', 'frozen_timings', 'frozen_checks', 'frozen_report', 'baseline_empty', 'baseline_constant', 'frozen_test',\n"
                "    'record', 'hypothesis', 'settings', 'adapt_result', 'adapt_seconds', 'adapted_test', 'adapted_val', 'comparison', 'delta_cer',\n"
                "    'adaptation_verdict', 'key', 'row', 'evaluation_report_payload', 'f', 'examples_dir', 'caption_font', 'frozen_hyp', 'adapted_hyp', 'line',\n"
                "    'width', 'sheet', 'marker', 'i', 'tag', 'text', 'adapted_answers', 'adapted_timings', 'adapted_checks', 'adapted_report', 'artifact_dir',\n"
                "    'artifact_manifest', 'reloaded', 'before', 'after', 'parity', 'result_payload', 'canonical_files', 'canonical', 'experiment_dir',\n"
                "    'experiment_pipe', 'experiment_test', 'side_by_side', 'unchanged',\n"
                ')\n\n\n'
                'def refuse_byod(file_name, reason):\n'
                '    """Remove the earlier run\'s results, then stop the cell with one Refused: line naming the file and the fix."""\n'
                '    import shutil\n\n'
                '    for name in RUN_RESULT_NAMES:\n'
                '        globals().pop(name, None)\n'
                "    for path in Path('outputs').glob('smolvlm_vision_language_*'):\n"
                '        shutil.rmtree(path) if path.is_dir() else path.unlink()\n'
                "    reason = str(reason).removeprefix(f'{{file_name}}: ').rstrip('. ')\n"
                "    if 'again' not in reason:\n"
                "        reason += '; fix the input and run this cell again'\n"
                "    raise BYODRefusal(file_name, reason + ' (the earlier results of this session were removed).') from None\n\n\n"
                'if USE_BYOD:\n'
                '    # DAT20: every problem with your input (path, upload, archive, transcripts.csv, line images, split) stops here as one Refused: line.\n'
                "    file_name = (Path(BYOD_PATH.strip()).name or BYOD_PATH.strip()) if BYOD_PATH.strip() else '(no file)'\n"
                '    try:\n'
                '        if BYOD_PATH.strip():\n'
                '            byod_zip = Path(BYOD_PATH.strip()).expanduser()\n'
                '            if not byod_zip.exists():\n'
                "                raise BYODRefusal(file_name, f'BYOD_PATH {{BYOD_PATH!r}} does not exist (relative paths start at {{Path.cwd()}}): give a .zip or a folder holding transcripts.csv and the line images.')\n"
                '        else:\n'
                '            try:\n'
                '                from google.colab import files\n'
                '            except ImportError:\n'
                "                raise BYODRefusal(file_name, 'USE_BYOD is True but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: on Kaggle or Jupyter put the zip (or folder) in the runtime and set BYOD_PATH to its path.') from None\n"
                '            uploaded = files.upload() or {{}}\n'
                '            if len(uploaded) != 1:\n'
                "                raise BYODRefusal(file_name, f'Upload exactly one .zip file (received {{len(uploaded)}}; a cancelled dialog sends none): run this cell again.')\n"
                '            file_name, payload = next(iter(uploaded.items()))\n'
                "            if not file_name.lower().endswith('.zip'):\n"
                "                raise BYODRefusal(file_name, 'upload one .zip holding transcripts.csv and the line images.')\n"
                "            byod_zip = Path('work') / 'byod.zip'\n"
                '            byod_zip.parent.mkdir(parents=True, exist_ok=True)\n'
                '            byod_zip.write_bytes(payload)\n'
                '        byod_report = {{}}\n'
                '        records = load_byod_dataset(byod_zip, report=byod_report)\n'
                '        splits = split_dataset(records, seed=SPLIT_SEED)\n'
                '    except BYODRefusal as exc:\n'
                '        refuse_byod(exc.file_name, exc.reason)\n'
                '    except (TypeError, ValueError, KeyError, OSError, zipfile.BadZipFile) as exc:\n'
                '        refuse_byod(file_name, exc)\n'
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records), 'skipped_files': byod_report['skipped'], 'duplicate_images_dropped': len(records) - sum(len(part) for part in splits.values()), 'effective_minimum': min_byod_records()['total']}}\n"
                "    if len(splits['test']) < 20:\n"
                '        print({{\'caution\': f"only {{len(splits[\'test\'])}} held-out test lines: the rates move in large steps and carry no dispersion estimate; add lines before reading them"}})\n'
                'else:\n'
                '    t0 = time.perf_counter()\n'
                "    corpus_groups = fetch_corpus(cache_dir='weights/belfort')\n"
                '    corpus = read_corpus(corpus_groups)\n'
                '    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n'
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'lines': sum(len(v) for v in corpus_groups.values()), 'bytes': sum(len(r['image']) + len(r['text'].encode('utf-8')) for v in corpus_groups.values() for r in v), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                '# The training split must hold MIN_RECORDS; validation and test only need one line each (split_dataset checks that).\n'
                'try:\n'
                "    dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "    splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                '    disjoint = check_split_disjoint(splits)\n'
                'except (TypeError, ValueError, KeyError) as exc:\n'
                '    if not USE_BYOD:\n'
                '        raise  # the pinned sample failing its contract is a genuine error, not a refusal\n'
                '    refuse_byod(file_name, exc)\n'
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'chars': manifest['text_chars'], 'words': manifest['text_words']['total'], 'width': manifest['image_width'], 'height': manifest['image_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "example['image'].save('outputs/{stem}_example_line.png')\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'text': example['text']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty transcript': [{{**train_records[0], 'text': '   '}}, *train_records[1:8]],\n"
                "    'image above the side ceiling': [{{**train_records[0], 'image': Image.new('RGB', (MAX_IMAGE_SIDE + 1, 8))}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three split sizes and their digests, the example line with its transcript, and the four refusals (each names its rule).\n\n<details><summary>Check your reasoning</summary>Yes. The split keeps *images* apart, not *writers*: lines cut from the same page of the same minutes can fall on both sides, and the model can learn a hand's letter shapes from training lines and meet them again in the test lines. That makes the held-out rate optimistic for a new writer — how much is not measured here (the pinned rows expose no page or writer key). On your own data, split by page, writer or volume when your lines come from few sources.</details>"
            ),
        },
        {
            "md": (
                "## 5. Ask two prompts of a synthetic drawing through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a 384 × 384 white canvas "
                "drawn in code — a filled red square in the upper left and a filled blue circle in the lower right — with a "
                "captioning prompt and a counting question; a different image family from the handwritten lines, and a drawing "
                "the adapted model will be asked about again in Section 9. `validate_inputs` applies exactly the checks "
                "`generate` applies (exactly one image with sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` and at most "
                "`MAX_IMAGE_PIXELS`, a non-empty prompt of at most `MAX_TEXT_CHARS` characters, `max_new_tokens` in "
                "1..`MAX_NEW_TOKENS`) and one combined manifest records both requests; an over-long prompt is validated too and "
                "its rejection recorded as a finding. `generate` returns the decoded assistant turn with `new_tokens` and a "
                "`truncated` flag — **no score exists**, and the prompt and the token budget are caller-owned request "
                "parameters. The per-image `evaluation_report` over the two answers is `not-measurable`: open-ended answers "
                "have no intrinsic correctness signal, and reading them against what you can see is a sanity check on one "
                "drawing; whether the model is *good at handwriting* is what Section 6 measures on 140 lines with a metric. The "
                "inference-only card recorded a caption naming both shapes and a count that got the colours right.\n\n"
                "*Core concept.* The model generates its answer greedily, token by token, and attaches no score to it: a fluent "
                "sentence is not evidence that it is right.\n\n"
                "**Predict before running:** the drawing has one red square and one blue circle. Will the model name both "
                "shapes, both colours, and the count — and how would you know if it did not, without a metric?"
            ),
            "code": (
                "drawing = Image.new('RGB', (384, 384), (255, 255, 255))\n"
                "draw = ImageDraw.Draw(drawing)\n"
                "draw.rectangle((48, 48, 176, 176), fill=(220, 30, 30))\n"
                "draw.ellipse((208, 208, 336, 336), fill=(30, 60, 220))\n"
                "drawing_name = 'synthetic_square_circle_384'\n"
                "drawing_sha256 = hashlib.sha256(np.asarray(drawing).tobytes()).hexdigest()\n"
                "DRAWING_PROMPTS = ['Describe this image in one sentence.', 'How many shapes are in the image, and what colour is each one?']\n"
                "DRAWING_MAX_NEW_TOKENS = 96\n"
                "print({{'ceilings': {{'MAX_IMAGES': MAX_IMAGES, 'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_IMAGE_PIXELS': MAX_IMAGE_PIXELS, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DEFAULT_LINE_MAX_NEW_TOKENS': DEFAULT_LINE_MAX_NEW_TOKENS, 'DECODING': DECODING, 'TRANSCRIBE_PROMPT': TRANSCRIBE_PROMPT, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device, 'dtype': pipe.dtype}}}})\n"
                "manifests = [validate_inputs(drawing, prompt, max_new_tokens=DRAWING_MAX_NEW_TOKENS, do_sample=False, names=[drawing_name]) for prompt in DRAWING_PROMPTS]\n"
                "input_manifest = {{**manifests[0], 'prompt': None, 'prompt_chars': None, 'n_prompts': len(manifests), 'findings': [], 'prompts': manifests}}\n"
                "try:\n"
                "    validate_inputs(drawing, 'x' * (MAX_TEXT_CHARS + 1))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'drawing': drawing_name, 'sha256': drawing_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def ask_drawing(pipeline, label):\n"
                "    answers, timings = [], []\n"
                "    for prompt in DRAWING_PROMPTS:\n"
                "        started = time.perf_counter()\n"
                "        answers.append(pipeline.generate(drawing, prompt, max_new_tokens=DRAWING_MAX_NEW_TOKENS))\n"
                "        timings.append(round(time.perf_counter() - started, 3))\n"
                "    checks = {{\n"
                "        'answers_are_text': all(isinstance(a['text'], str) for a in answers),\n"
                "        'greedy_settings': all(a['generation']['do_sample'] is False and a['generation']['decoding'] == DECODING for a in answers),\n"
                "        'budget_honoured': all(a['new_tokens'] <= DRAWING_MAX_NEW_TOKENS for a in answers),\n"
                "        'identity_reported': all(a['model_id'] == MODEL_ID and a['model_revision'] == MODEL_REVISION for a in answers),\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'generate output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(answers, sample_kind='synthetic (drawn in this notebook)')\n"
                "    with open(f'outputs/{stem}_drawing_{{label}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'answers': answers, 'report': report}}, handle, indent=2, ensure_ascii=False)\n"
                "    print({{label: {{'seconds': timings, 'checks': checks, 'answers': [(a['prompt'], a['text'], a['new_tokens'], a['truncated']) for a in answers], 'verdict': report['verdict']}}}})\n"
                "    return answers, timings, checks, report\n\n\n"
                "frozen_answers, frozen_timings, frozen_checks, frozen_report = ask_drawing(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "**What to notice:** the structural checks, the two answers, `new_tokens` against the budget and the verdict.\n\n<details><summary>Check your reasoning</summary>In the recorded run the caption was `Two shapes are present. One is red and the other is blue.` and the counting answer `There are two shapes in the image. Shape A is red, and shape B is blue.`: the count and the colours are right, the shapes are not named. You know only because you drew the image yourself — the per-image report's verdict is `not-measurable`, because open-ended answers have no intrinsic correctness signal.</details>"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test lines\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `ocr_metrics` (carried in `metrics.py`): the "
                "**character error rate** and **word error rate** as micro averages — total Levenshtein edits over total "
                "reference characters or words, the corpus CER/WER of the handwriting-recognition literature — beside the "
                "macro (per-line mean) rates and the exact-match rate. Neither rate is capped: a hypothesis longer than its "
                "reference pushes the rate **above 1.0**, the signal that the model is generating text the line does not carry. "
                "The **empty-string** baseline predicts nothing and scores CER 1.0 exactly (every reference character is a "
                "deletion) — the floor any recogniser must beat to do better than silence. The **constant-transcript** baseline "
                "predicts one training transcript — the medoid, the line closest on average to the others — for every test "
                "line: what corpus statistics buy without reading the image. The **frozen model** is scored by `pipe.evaluate`, "
                "which asks `TRANSCRIBE_PROMPT` of every line in left-padded batches of `EVAL_BATCH_SIZE` under a "
                "`LINE_MAX_NEW_TOKENS` budget and returns the answers as hypotheses with the rates. Expect the frozen model "
                "**near the empty baseline**: the recorded runs measured 1.809 (hypotheses 1.70 times "
                "the reference length — a word or two read, the rest an English sentence about the image); read four of them "
                "under the references.\n\n"
                "*Evaluation practice.* **Predict before running:** an empty answer scores CER exactly 1.0. Will the frozen "
                "model — asked to transcribe French cursive that its documented training data does not include — score below 1.0, near it, or above it?"
            ),
            "code": (
                "METRICS = ('cer', 'wer', 'cer_macro', 'exact_match')\n"
                "LINE_MAX_NEW_TOKENS = 128  # @param {{type:\"integer\"}}\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_constant = constant_baseline(train_records, test_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'constant_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'note': baseline_constant['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, prompt=TRANSCRIBE_PROMPT, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'ref_chars': frozen_test['ref_chars'], 'hyp_chars': frozen_test['hyp_chars'], 'truncated': frozen_test['truncated'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for record, hypothesis in zip(test_records[:4], frozen_test['hypotheses'][:4], strict=True):\n"
                "    print({{'id': record['id'], 'reference': record['text'], 'frozen': hypothesis[:120]}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the three systems' CER and WER, `hyp_chars` against `ref_chars`, and the four example answers.\n\n<details><summary>Check your reasoning</summary>Well above it. In the recorded run the empty baseline scored CER 1.0, the constant-transcript (medoid) baseline 0.944 and the frozen model **1.809** (WER 2.215): its answers were 1.70 times as long as the references (10,515 characters against 6,169) — a word or two read, then an English sentence about the image. Every extra character is an insertion, which is how an uncapped CER climbs past 1.0: the model is generating, not reading.</details>"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the last decoder layers\n\n"
                "`pipe.adapt` trains only the last eight of the 32 SmolLM2 decoder layers and the final norm — 78,659,520 of "
                "507,482,304 parameters — while the SigLIP vision encoder, the connector, the embeddings, the output head and "
                "the first 24 decoder layers stay frozen. Each training line is the chat-templated user turn (the tiles' visual "
                "tokens and the instruction) followed by the assistant turn: the transcript's tokens and the end-of-utterance "
                "token; the loss is the **causal language-model cross-entropy** over the assistant turn with the user turn "
                "masked out — the checkpoint's own instruction-tuning objective. Because everything before layer 24 is frozen, "
                "its output for every training line is computed once under no gradient and cached (the **frozen-prefix "
                "cache**), and each step runs only the eight trainable layers, the norm and the output head on those cached "
                "states — the loss equals the full model's loss exactly, at a fraction of the cost. AdamW without weight decay "
                "at a fixed learning rate, gradient clipping at 1.0, seeded shuffling, no scheduler, no augmentation. Epoch 0 "
                "records the frozen model's validation rates; every epoch is scored on the 60 validation lines, and the epoch "
                "with the **lowest validation CER** is kept.\n\n"
                "Watch the validation CER fall from 1.641 to 0.804 (epoch 7 in the recorded "
                "runs) while the loss drops from about 3.41 to 0.02 — still about 0.8 edits per reference character on the validation lines, and CER counts insertions as well as substitutions and deletions, so 1 − CER is not the share of characters read correctly. The two sibling "
                "rows on the same split (GOT-OCR 2.0, Florence-2) squash a line into a square and report their own numbers in their "
                "own repositories (see the model card); they were not re-run or paired with this run.\n\n"
                "*Core concept.* Every call to `pipe.adapt` starts from the **pinned base**: tensors an earlier call changed are "
                "restored first, so epoch 0 is always the frozen model. Re-running this cell with a changed field is a fresh run, "
                "not continued training — but it replaces the default results that Sections 8 and 9 export.\n\n"
                "**Predict before running:** the training loss will fall every epoch. Will the last epoch also have the lowest "
                "validation CER?"
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "settings = {{'epochs': EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE}}\n"
                "if settings != {{'epochs': 8, 'lr': 1e-4, 'batch_size': 8}}:\n"
                "    print({{'note': 'changed settings: this run starts again from the pinned base and replaces the default results of Sections 8-9', 'settings': settings}})\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, prompt=TRANSCRIBE_PROMPT, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, max_new_tokens=LINE_MAX_NEW_TOKENS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'prompt': adapt_result['prompt'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'first_trainable_layer': adapt_result['first_trainable_layer'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'started_from': adapt_result['started_from'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '**What to notice:** epoch 0 (the frozen model, before any update), the loss column, the validation CER column and `best_epoch`.\n\n<details><summary>Check your reasoning</summary>No. In the recorded run the validation CER fell from 1.641 at epoch 0 (the frozen model) to 0.804 at epoch 7 and rose again to 1.002 at epoch 8, so epoch 7 was kept, while the training loss kept falling (about 3.41 to 0.02). A falling loss is not evidence of better reading; the validation split decides, and the test split is kept for Section 8.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test lines were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **CER** first (the measure the epoch was selected on — the recorded runs measured "
                "1.809 → **0.890**, past both baselines), then **WER** (2.215 → 1.143), "
                "then the hypothesis length (from 1.70 times the reference length to 0.96), "
                "then the exact-match rate (2 of the 140 lines read perfectly). The cell reports two **verdicts** — whether the "
                "adapted CER `improved`, showed `no gain` or got `worse` against the frozen one, and whether it is below the "
                "empty baseline's 1.0 — and records them in the evaluation report; a fine-tune that does not help is a finding, "
                "not an error, and Section 9 still exports, reloads and writes the result. The split is by line (Section 4): test "
                "lines can share pages and writers with training lines, so the rates are a within-collection estimate, optimistic "
                "for a new hand. One hundred and forty lines from one seeded "
                "split give **no dispersion estimate**; the deltas are sample-sanity evidence that the adaptation contract "
                "works, not a benchmark, and a result on one French council's minutes says nothing about other hands, other "
                "languages or other scripts until you measure them.\n\n"
                "**Predict before running:** will the adapted model beat the empty baseline (CER 1.0) on the micro rate? On the "
                "macro (per-line) rate?"
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, prompt=TRANSCRIBE_PROMPT, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, prompt=TRANSCRIBE_PROMPT, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'constant': round(baseline_constant[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['hypothesis_length'] = {{'ref_chars': adapted_test['ref_chars'], 'frozen_hyp_chars': frozen_test['hyp_chars'], 'adapted_hyp_chars': adapted_test['hyp_chars'], 'frozen_truncated': frozen_test['truncated'], 'adapted_truncated': adapted_test['truncated']}}\n"
                "# Reported verdicts, not assertions: a non-improving fine-tune is a result to record, and export and reload still run.\n"
                "delta_cer = adapted_test['cer'] - frozen_test['cer']\n"
                "adaptation_verdict = 'improved' if delta_cer < 0 else ('no gain' if delta_cer == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'adapted_vs_frozen_cer': adaptation_verdict, 'adapted_below_empty_baseline': bool(adapted_test['cer'] < baseline_empty['cer'])}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'prompt': TRANSCRIBE_PROMPT,\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'empty': {{k: v for k, v in baseline_empty.items() if k != 'rows'}}, 'constant': {{k: v for k, v in baseline_constant.items() if k != 'rows'}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'rows'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'rows'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'rows'}},\n"
                "    'per_line': [{{**frozen_row, 'frozen_hypothesis': frozen_hyp, 'adapted_cer': adapted_row['cer'], 'adapted_hypothesis': adapted_hyp}} for frozen_row, frozen_hyp, adapted_row, adapted_hyp in zip(frozen_test['rows'], frozen_test['hypotheses'], adapted_test['rows'], adapted_test['hypotheses'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'verdicts': comparison['verdicts'], 'adapted_beats_both_baselines': adapted_test['cer'] < min(baseline_empty['cer'], baseline_constant['cer'])}})"
            ),
        },
        {
            "md": (
                '**What to notice:** the four-way table, the `verdicts` row, the macro CER beside the micro CER, and the hypothesis-length row.\n\n<details><summary>Check your reasoning</summary>On the micro rate, yes; per line, no. In the recorded run the verdict was *improved*: test CER 1.809 → 0.890 and WER 2.215 → 1.143, with the answers shrinking from 1.70 to 0.96 times the reference length and 2 of 140 lines exact. But the macro CER was 1.181 — the average line still had more edits than reference characters, because some lines still run on (CER above 1.0 on 24 of the 140). The micro rate is dominated by the long lines the model reads best.</details>'
            ),
        },
        {
            "md": (
                "## 9. Look at the lines, ask the drawing again, export the adapter and reload it\n\n"
                "Six held-out lines are written as panels (`outputs/{stem}_examples/`: the line image with the reference, the "
                "frozen answer and the adapted answer beneath it) so the numbers can be checked by eye: the adapted rows should "
                "read the cursive the frozen rows described in English. The drawing from Section 5 is then asked the same two "
                "prompts by the adapted model — the decoder that was tuned answers every prompt, so this is a small look at what "
                "the adaptation did *outside* its instruction and its corpus: the recorded runs measured before adaptation `Describe this image in one sentence.` → `Two shapes are present. One is red and the other is blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.`; after adaptation `Describe this image in one sentence.` → `Two colors, red and blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.` — one "
                "drawing of evidence, not a measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the eight decoder layers and the norm, about 315 MB in "
                "float32 — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "instruction, the training configuration and the epoch history (OUT8). `SmolVLMPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, refuses any tensor outside the last eight decoder layers and the norm, and overlays the tensors "
                "onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell asserts "
                "identical transcripts on eight test lines (VER4) — a contract check, so it stays a hard check.\n\n"
                "**Predict before running:** only the decoder's last layers were tuned, on one instruction. Will the drawing's "
                "caption and count change?"
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "for record, frozen_hyp, adapted_hyp in zip(test_records[:6], frozen_test['hypotheses'][:6], adapted_test['hypotheses'][:6], strict=True):\n"
                "    line = record['image']\n"
                "    width = min(1400, line.width)\n"
                "    line = line.resize((width, max(1, round(line.height * width / record['image'].width))))\n"
                "    sheet = Image.new('RGB', (max(width, 1400), line.height + 96), (255, 255, 255))\n"
                "    sheet.paste(line, (0, 0))\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    for i, (tag, text) in enumerate((('REF', record['text']), ('FROZEN', frozen_hyp), ('ADAPTED', adapted_hyp))):\n"
                "        marker.text((8, line.height + 6 + i * 28), f'{{tag}}: {{text[:140]}}', fill=(20, 20, 20) if tag != 'FROZEN' else (150, 40, 40), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'rows': ['reference', 'frozen answer', 'adapted answer']}})\n\n"
                "adapted_answers, adapted_timings, adapted_checks, adapted_report = ask_drawing(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'prompt': artifact_manifest['adapter']['prompt'], 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = SmolVLMPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [item['text'] for item in pipe.transcribe([r['image'] for r in test_records[:8]], prompt=TRANSCRIBE_PROMPT, max_new_tokens=LINE_MAX_NEW_TOKENS)]\n"
                "after = [item['text'] for item in reloaded.transcribe([r['image'] for r in test_records[:8]], prompt=TRANSCRIBE_PROMPT, max_new_tokens=LINE_MAX_NEW_TOKENS)]\n"
                "parity = {{'identical_lines': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_lines'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'language': CORPUS_LANGUAGE, 'row_groups': sorted(ROW_GROUP_PINS), 'shard_bytes': CORPUS_BYTES}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'drawing': {{'name': drawing_name, 'sha256': drawing_sha256, 'prompts': DRAWING_PROMPTS}}, 'frozen': {{'answers': frozen_answers, 'seconds': frozen_timings, 'checks': frozen_checks, 'report': frozen_report}}, 'adapted': {{'answers': adapted_answers, 'seconds': adapted_timings, 'checks': adapted_checks, 'report': adapted_report}}, 'output_files': ['outputs/{stem}_drawing_frozen.json', 'outputs/{stem}_drawing_adapted.json']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pillow': PIL.__version__, 'device': pipe.device, 'source': pipe.source, 'dtype': pipe.dtype}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                '**What to notice:** the line panels, the two drawing answers before and after, and `identical_lines` against `of`.\n\n<details><summary>Check your reasoning</summary>The caption did: in the recorded run it went from `Two shapes are present. One is red and the other is blue.` to `Two colors, red and blue.` — shorter and vaguer — while the counting answer was unchanged. The decoder that was tuned answers every prompt, so tuning one instruction moves the others; one drawing is evidence of that, not a measurement. Reload parity held: 8 of 8 identical transcripts.</details>'
            ),
        },
        {
            "md": (
                "## 10. Change one thing: the instruction's wording (optional)\n\n"
                "*Evaluation practice.* A **Predict → Change one thing → Run → Observe → Explain** activity, off by default so "
                "Run all is unaffected. Set `RUN_EXPERIMENT = True`, edit **only** `EXPERIMENT_PROMPT` (a form field — never the "
                "carried module), and run this cell after Sections 4–9. It loads its **own** frozen pipeline from the verified "
                "snapshot (the default `pipe` is adapted by now and is not touched), scores the 140 test lines with the new "
                "wording and prints it beside the frozen model under `TRANSCRIBE_PROMPT` from Section 6. It writes only to "
                "`outputs/{stem}_experiment/` and checks that the default exports are byte-identical afterwards. No training "
                "runs: expect about three minutes on a T4.\n\n"
                "**Predict:** the default instruction is in English and the lines are French. If the prompt says *French "
                "handwriting* and asks for nothing else, will the frozen CER fall, stay, or rise?"
            ),
            "code": (
                "RUN_EXPERIMENT = False  # @param {{type:\"boolean\"}}\n"
                "EXPERIMENT_PROMPT = 'Transcribe the French handwriting in this image. Answer with the text only.'  # @param {{type:\"string\"}}\n\n"
                "if not RUN_EXPERIMENT:\n"
                "    print({{'experiment': 'skipped (RUN_EXPERIMENT = False); the default path above is complete'}})\n"
                "else:\n"
                "    canonical_files = {{'adapter': artifact_dir / 'adapter.safetensors', 'evaluation_report': Path('outputs/{stem}_evaluation_report.json'), 'result': Path('outputs/{stem}_result.json')}}\n"
                "    canonical = {{name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in canonical_files.items()}}\n"
                "    experiment_dir = Path('outputs/{stem}_experiment')\n"
                "    shutil.rmtree(experiment_dir, ignore_errors=True)\n"
                "    experiment_dir.mkdir(parents=True)\n"
                "    # Its own frozen pipeline from the verified snapshot: the default pipe (adapted by now) is untouched.\n"
                "    experiment_pipe = SmolVLMPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    experiment_test = experiment_pipe.evaluate(test_records, prompt=EXPERIMENT_PROMPT, max_new_tokens=LINE_MAX_NEW_TOKENS, batch_size=EVAL_BATCH_SIZE)\n"
                "    side_by_side = {{\n"
                "        'prompt': {{'default': TRANSCRIBE_PROMPT, 'experiment': EXPERIMENT_PROMPT}},\n"
                "        'frozen_test': {{metric: {{'default_prompt': round(frozen_test[metric], 3), 'experiment_prompt': round(experiment_test[metric], 3)}} for metric in METRICS}},\n"
                "        'hypothesis_chars': {{'reference': frozen_test['ref_chars'], 'default_prompt': frozen_test['hyp_chars'], 'experiment_prompt': experiment_test['hyp_chars']}},\n"
                "        'adapted_test_for_reference': {{metric: round(adapted_test[metric], 3) for metric in METRICS}},\n"
                "    }}\n"
                "    for key, row in side_by_side.items():\n"
                "        print({{key: row}})\n"
                "    for record, hypothesis in zip(test_records[:3], experiment_test['hypotheses'][:3], strict=True):\n"
                "        print({{'id': record['id'], 'reference': record['text'], 'experiment_prompt': hypothesis[:120]}})\n"
                "    with open(experiment_dir / 'experiment_report.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'side_by_side': side_by_side, 'test_metrics': {{k: v for k, v in experiment_test.items() if k != 'rows'}}}}, handle, indent=2, ensure_ascii=False)\n"
                "    unchanged = {{name: hashlib.sha256(path.read_bytes()).hexdigest() == canonical[name] for name, path in canonical_files.items()}}\n"
                "    if not all(unchanged.values()):\n"
                "        raise RuntimeError(f'the experiment changed a default export: {{unchanged}}')\n"
                "    print({{'default_exports_unchanged': unchanged, 'experiment_outputs': str(experiment_dir)}})\n"
                "    del experiment_pipe"
            ),
        },
        {
            "md": (
                "**Observe → Explain.** Compare the two `frozen_test` columns and the hypothesis lengths, then read the three "
                "example answers.\n\n"
                "<details><summary>Check your reasoning</summary>No experiment run is recorded yet, so the number is yours to "
                "explain. What the default run tells you: the frozen model's CER of 1.809 came mostly from answers 1.70 times "
                "too long — English sentences about the image. A wording that asks for the text only attacks exactly that, so "
                "the hypothesis length is the first thing to read: if it falls toward the reference length and the CER falls "
                "with it, the frozen model's errors were partly the prompt's; if the CER barely moves, they were the model's. "
                "Either way the fine-tuned model (0.890) shows how much more training buys than wording.</details>"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "An instruction-tuned vision-language model answers in the register it was tuned on, and asked to transcribe "
        "nineteenth-century cursive French it reads a word or two and describes the rest in English: the frozen model scores "
        "a character error rate of 1.809 on the Belfort lines. A bounded fine-tuning of the last eight decoder layers "
        "on 600 transcribed lines moves it to 0.890 CER and 1.143 WER in the recorded runs "
        "(2 of the held-out lines exact), with a 315 MB adapter that reloads line-for-line. That is the claim: "
        "the adaptation contract works end to end on one instruction of a chat model with a real labelled set, and the numbers "
        "it produces are read as micro and macro rates, against two non-adapted baselines and the frozen model, with the "
        "hypothesis length beside them rather than in isolation. The two sibling rows on the same split (GOT-OCR 2.0 and Florence-2, both squashing the line into a "
        "square; their numbers are in their own repositories and this model card) ran separate fixed recipes on one seeded 140-line split with no paired test and no dispersion estimate, so the three adapted CERs are not a ranking, and the aspect-ratio difference is not shown to cause any gap; its adapted transcripts are French council prose in the right register and at the right length (0.96 times the reference) but rarely the right words — 10 of 140 lines under 0.5 CER, 24 still above 1.0 — which in this one run looks like learning the language of the corpus more than reading the lines; one run does not establish that mechanism.\n\n"
        "The test split is 140 lines from one seeded draw of one 800-line sample, the validation split that picks the epoch is "
        "60, and both rates are corpus edit distances over one crowdsourced transcription — not a benchmark, not a measure of "
        "reading order or layout. So a result here says the contract works on one council's minutes, not that the adapted "
        "model handles other hands, other languages, other scripts or your scans. The decoder that was tuned answers every "
        "prompt: the drawing re-asked in Section 9 is one image of evidence about what the tuning did to captioning and "
        "counting (before adaptation `Describe this image in one sentence.` → `Two shapes are present. One is red and the other is blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.`; after adaptation `Describe this image in one sentence.` → `Two colors, red and blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.`), not a measurement, and a deployment that needs the other prompts must measure them "
        "after adapting. The decoder was adapted, not the vision encoder: what the tiles cannot resolve stays unread.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and constant-transcript rates on *your* transcripts, "
        "and the frozen model's hypothesis length, are the numbers to read before any adapted one. **Rates above 1.0:** an "
        "uncapped CER tells you the model is generating, not reading; a capped one would hide it. **Leakage:** keep every image "
        "in one split (the contract de-duplicates by decoded pixels) and split by page, writer or volume when your lines come "
        "from few sources — lines cut from the same page share a hand.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled line set, validate the "
        "demonstrated dataset contract without leakage, execute the inference contract for two prompts and a bounded "
        "fine-tuning of one instruction with the model's own objective, evaluate against two non-adapted baselines and the "
        "frozen model on a line-disjoint split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, transcription quality on any other hand, language or "
        "document family, caption or question-answering quality after adaptation, or production fitness.\n\n"
        "**Optional experiments (off by default; each names its field and what to run):** Section 10 changes the "
        "instruction's wording (`EXPERIMENT_PROMPT`) in its own frozen pipeline and prints it beside the default; run that "
        "cell again for each wording. Changing `EPOCHS`, `LEARNING_RATE` (for example `2e-4`, a faster, noisier curve; the "
        "build sweep found lr 5e-5 and four layers leave the adapted CER above 1.0) or `BATCH_SIZE` in Section 7 and choosing "
        "**Run after** starts again from the pinned base and replaces the default results and exports; the verdicts in "
        "Section 8 report a run that does not improve instead of stopping. Lowering `LINE_MAX_NEW_TOKENS` in Section 6 to "
        "`64` and choosing **Run after** from Section 6 shows how the truncation count changes. BYOD: `USE_BYOD` and "
        "`BYOD_PATH` in Section 4, then **Run after** from Section 4, and read the two baselines before the adapted "
        "number.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"This notebook needs a Linux x86_64 runtime\"** — you are on Windows, macOS or an ARM "
        "machine. Use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n"
        "- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a "
        "complete environment is reused, an incomplete one is finished. If it repeats, the network is blocking or altering "
        "`files.pythonhosted.org` or `pypi.org`.\n"
        "- **\"The isolated environment's Python process exited\"** — usually out of memory (Section 9 holds the adapted and "
        "the reloaded model, Section 10 a third). Restart the session and choose **Run all**; leave Section 10 off on a small "
        "GPU.\n"
        "- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable, so the "
        "cells after it keep working. After a session restart, run from the top.\n"
        "- **Where the environment lives** — Section 1 builds it in a folder named `dimer_isolated_env_<lock digest>` in the working "
        "directory. Set the environment variable `DIMER_ISOLATED_ENV` before Section 1 to build it somewhere else.\n"
        "- **No GPU / it is very slow** — the default path needs a CUDA runtime in practice (about 32 minutes on a T4, hours "
        "on CPU). Choose **Runtime → Change runtime type → T4 GPU** and run all again.\n"
        "- **`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3** — a staged file is "
        "incomplete or altered: delete it from the snapshot folder Section 3 prints as `weights_dir` and run Section 3 again "
        "(it needs `huggingface.co`).\n"
        "- **Section 4 cannot fetch a row group, or one fails its digest** — `fetch_corpus` names it; the Hub may be briefly "
        "unreachable. Run Section 4 again (cached row groups are re-hashed); delete `weights/belfort/` if a cached file is "
        "corrupt.\n"
        "- **CUDA out of memory in Section 7** — set `BATCH_SIZE = 4` and choose **Run after** from Section 7 (the numbers "
        "will differ a little from the recorded run).\n"
        "- **BYOD: \"BYOD_PATH … does not exist\"** — the path is relative to the working directory printed in the "
        "message.\n"
        "- **BYOD: \"the upload dialog exists only in Google Colab\"** — on Kaggle or Jupyter, put the zip in the runtime and "
        "set `BYOD_PATH`.\n"
        "- **BYOD: \"Upload exactly one .zip file\"** — the dialog was cancelled or several files were chosen; run Section 4 "
        "again.\n"
        "- **BYOD: \"transcripts.csv line N (file …): …\"** — that row names an image that is missing, undecodable or too "
        "large, or a transcript that is empty or too long; fix that row.\n"
        "- **BYOD: \"two images share the file name …\"** — rows name images by file name; rename one of the two.\n"
        "- **BYOD: \"… image file(s) have no transcripts.csv row\"** — every image needs a row (non-image files and "
        "`__MACOSX` entries are skipped).\n"
        "- **BYOD: \"… leave the training split with … supply at least 12 distinct images\"** — the set is too small for "
        "the split.\n\n"
        "## Glossary\n\n"
        "- **Chat template / user turn / assistant turn** — the tokenizer's recipe for wrapping the image tokens and the "
        "prompt (the user turn) before the model writes its answer (the assistant turn).\n"
        "- **Greedy decoding** — always taking the most likely next token; deterministic on one device, no score attached.\n"
        "- **CER / WER** — character and word error rates: Levenshtein edits (insertions, deletions, substitutions) over "
        "reference characters or words. **Micro** sums over the corpus; **macro** averages per line. Neither is capped at 1, "
        "and 1 − CER is not the share of characters read correctly.\n"
        "- **Exact match** — the share of lines read with no error.\n"
        "- **Empty / constant (medoid) baseline** — predict nothing; predict for every line the training transcript closest "
        "on average to the others.\n"
        "- **Causal language-model loss** — cross-entropy of each next token of the target; here only the assistant turn "
        "(the transcript) is scored.\n"
        "- **Frozen-prefix cache** — the hidden states after the frozen layers, computed once per line, so training runs "
        "only the tuned tail.\n"
        "- **Epoch / learning rate / validation selection** — one pass over the training lines; the update step size; "
        "choosing the epoch on lines that are neither trained on nor used for the final score.\n"
        "- **Held-out test split / leakage** — lines never used for training or selection; the same image, page or writer "
        "on both sides of a split.\n"
        "- **Adapter / reload parity** — the trained tensors only (safetensors) overlaid on the pinned base; the reloaded "
        "pipeline reads the same lines identically.\n"
        "- **Drift** — what tuning one instruction does to the model's answers to other prompts.\n"
        "- **BYOD** — bring your own data: your transcribed lines through the same cells.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional — fill in from **your** run, not the recorded one:\n\n"
        "- The task was ___ on ___ held-out lines.\n"
        "- The empty baseline scored CER ___, the constant baseline ___ and the frozen model ___.\n"
        "- After fine-tuning, CER moved from ___ to ___ (macro ___) and WER from ___ to ___; the verdict was ___.\n"
        "- The drawing's caption changed from ___ to ___, because ___.\n"
        "- One reason not to trust this number on my own documents yet: ___ (for example shared writers across the split, "
        "the macro rate, or drift).\n\n"
        "## References\n\n"
        "[1] A. Marafioti *et al.*, \"SmolVLM: Redefining small and efficient multimodal models,\" arXiv:2504.05299, 2025. [Online]. Available: https://arxiv.org/abs/2504.05299\n\n"
        "[2] Hugging Face, \"smollm,\" GitHub repository. [Online]. Available: https://github.com/huggingface/smollm\n\n"
        "[3] Upstream model checkpoint. [Online]. Available: https://huggingface.co/{MODEL_ID}\n\n"
        "[4] K. Valcorza, \"smolvlm-vision-language-pipeline,\" README, weight provenance and model card. [Online]. Available: https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/README.md, https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/docs/WEIGHTS.md; model card: https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/MODEL_CARD.md\n\n"
        "[5] S. Tarride *et al.*, \"Handwritten text recognition from crowdsourced annotations,\" in *Proc. 7th Int. Workshop Historical Document Imaging and Processing (HIP)*, 2023, doi: 10.1145/3604951.3605517. Dataset (Teklia, MIT): https://huggingface.co/datasets/Teklia/Belfort-line\n\n"
        "[6] K. Valcorza, sibling rows on the same split. [Online]. Available: https://github.com/kurtvalcorza/got-ocr2-pipeline, https://github.com/kurtvalcorza/florence2-vision-language-pipeline\n\n"
        "[7] DIMER Notebook Specification 2.5 and Model Card Specification 1.1, fleet specs in the ml-worker repository."
    ),
}
