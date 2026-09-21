# Release verification

`tutorials/smolvlm_vision_language_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook. The earlier `TASK-INFERENCE` notebook's Colab Tesla T4 run (2026-09-13, retained under
`verification/2026-09-13/`) is history for a superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 13-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the Belfort-line
  parquet-conversion revision `c4a74bbd39f2df314752e7e6026649a39d365cbb` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `SmolVLMPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path, `read_corpus`,
  `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`, `validate_dataset` per
  split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the ceiling print,
  `validate_inputs` per prompt with the over-long-prompt refusal probe, `generate` for the two prompts with the
  structural checks and the `evaluation_report` on the drawing, `empty_baseline`, `constant_baseline`,
  `pipe.evaluate` on the frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the
  validation and test splits after adaptation with the two CER assertions, the two prompts asked of the drawing after
  adaptation, the example panels, `pipe.save_artifact`, `SmolVLMPipeline.from_artifact` and the transcript-parity
  assertion, and the result fields `weight_file` / `weight_format` / `weight_sha256`, the `corpus` block), the seven expected `outputs/` paths, the learner-facing statements (Apache-2.0 weights, greedy decoding,
  generated text with no score, adaptation of one instruction on transcribed lines, the two non-adapted baselines, CER
  and WER, rates above 1.0, the empty-string and constant-transcript baselines, the causal language-model loss, the
  frozen-prefix cache, lowest validation CER, no dispersion estimate, no score exists, float32 on every device, the
  leakage guidance, the single-image single-turn scope, the snapshot note, the troubleshooting block) and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on the primary
  path, a mutable `revision='main'`, direct `from transformers import` / `AutoModelForImageTextToText` /
  `AutoProcessor` / `apply_chat_template(` / `model.generate(` / `torch.inference_mode(` / `from
  huggingface_hub import` / `urllib.request` / `pyarrow` / `safetensors` imports / `torch.optim` / `.backward(` /
  `requires_grad` / `pipe._model` / `pipe._processor` / `extractall(` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `huggingface-hub`, `safetensors`, `numpy`,
`pillow` and `pyarrow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py
--check`, and the unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab GPU runtime (CUDA; a CPU runtime is not practical for the default path) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Kaggle script kernel (pre-flight only) | Fresh GPU container that clones the candidate branch, installs the pins and runs `tests/test_model_backed.py` plus the package-API recipe probe | Builder pre-flight to catch defects and fix the recipe before spending a notebook run; **not** promotion evidence for the notebook blob |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CUDA runtime (Colab GPU, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/smolvlm-500m-instruct/` or the row-group cache `weights/belfort/` (the standalone path writes
   the manifest itself, stages the missing files from the Hub and reads the pinned row groups over range requests, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `LINE_MAX_NEW_TOKENS = 128`, `EPOCHS = 8`, `LEARNING_RATE = 1e-4`,
   `BATCH_SIZE = 8`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `pyarrow==25.0.1` (an interpreter restart after the install is expected where
   the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `SmolVLMPipeline`, `verify_snapshot`, `stage_missing_files`,
     `validate_inputs`, `build_messages`, `evaluation_report`, `edit_distance`, `normalise_text`, `ocr_metrics`,
     `empty_baseline`, `constant_baseline`, `medoid_transcript`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`, `load_byod_dataset`,
     `write_dataset_csv`, the instruction constant and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 13 manifest entries fetched from `HuggingFaceTB/SmolVLM-500M-Instruct` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (13 files, the 1.02 GB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory on `cuda:0` in float32 ;
   - Section 4: `fetch_corpus` reading the eight pinned row groups over HTTPS range requests with every SHA-256 and
     byte total matching (800 lines, about 44 MB); the seeded split into 600 / 60 / 140 with `check_split_disjoint`
     reporting no shared image and the three dataset digests printed; `outputs/…_train.csv` and
     `outputs/…_example_line.png` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings surfaced; the drawing rendered; the combined input manifest written to
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the over-long-prompt
     probe); the two prompts answered with every structural check `True`, `outputs/…_drawing_frozen.json` written and
     the `evaluation_report` verdict `not-measurable` (open-ended answers; the inference-only card recorded a caption
     naming both shapes — an observation, not an assertion);
   - Section 6: the empty baseline (CER 1.0 exactly), the constant-transcript baseline (≈ 0.94) and the frozen model's
     test rates (≈ 1.809 CER / 2.215 WER in the Tesla T4 build record — the frozen model does not transcribe the lines: it answers in English about the handwriting, invents dates and sentences, and loops on one word until the 128-token budget — 14 of 140 generations truncated and 10,515 hypothesis characters for 6,169 reference characters, so the CER passes 1.0 by over-generation rather than silence) with four hypotheses printed under their references;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 78,659,520 trainable of 507,482,304 parameters,
     `first_trainable_layer` 24, and an eight-epoch history with the validation CER falling (build record:
     1.641 → 1.145 / 1.031 / 1.020 / 1.128 / 0.928 / 0.861 / 0.804 / 1.002, `best_epoch` 7);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the hypothesis
     lengths and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test CER is below the frozen
     one and below 1.0 — 0.890 against 1.809 in the build record, WER 2.215 →
     1.143; the adapted model also clears the constant baseline, reported, not asserted);
   - Section 9: six example panels under `outputs/…_examples/`; the two prompts asked of the drawing again by the
     adapted model with `outputs/…_drawing_adapted.json` (build record: before adaptation `Describe this image in one sentence.` → `Two shapes are present. One is red and the other is blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.`; after adaptation `Describe this image in one sentence.` → `Two colors, red and blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.` — a recorded observation,
     not an assertion); `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (37
     tensors, about 315 MB) and `SmolVLMPipeline.from_artifact` reloading it with 8/8 identical transcripts on
     eight test lines (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model
     identity and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block,
     the inference-contract records before and after adaptation, the comparison, the artifact digest, the reload
     parity, the runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `smolvlm_vision_language_colab.ipynb` (`E2E`) | `385b214` / `5b2c5b22` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-smolvlm-vision-language` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 36 files, 1062 MB staged from the Hub into a clean cache; comparison {cer: {empty: 1, constant: 0.944, frozen: 1.809, adapted: 0.89}, wer: {empty: 1, constant: 0.999, frozen: 2.215, adapted: 1.143}, cer_macro: {empty: 1, constant: 0.952, frozen: 2.078, adapted: 1.181}, exact_match: {empty: 0, constant: 0, frozen: 0.007, adapted: 0.014}, delta_vs_frozen: {cer: -0.919, wer: -1.072, cer_macro: -0.897, exact_match: 0.007}, hypothesis_length: {ref_chars: 6169, frozen_hyp_chars: 10515, adapted_hyp_chars: 5902, frozen_truncated: 14, adapted_truncated: 4}}; drawing / scene / page check frozen vs adapted {frozen: [(15, False), (20, False)], adapted: [(8, False), (20, False)]}; reload parity {identical_lines: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-smolvlm-vision-language/v3/evidence/` in the workspace |
| `smolvlm_vision_language_colab.ipynb` (`TASK-INFERENCE`, superseded) | `fe478b69e92bcc144cb2a38e7f12e5bbe65a1873` / `8e97c78fc79e38124910c3d40850874ac5fb0194` | 2026-09-13 | Colab CLI → isolated Python 3.12.3, Tesla T4 | PASS — 8/8 cells; [retained run](verification/2026-09-13/README.md); evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/smolvlm_vision_language_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/smolvlm_vision_language_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-21 | `ef38442` / `cb42b8ca` (pre-flight: the build-record placeholders still unfilled in the prose, code identical) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-smolvlm-vision-language` v2) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 2084.6 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 36 files, 1062 MB staged; the metrics the build record quotes |
| 2026-09-21 | `385b214` / `5b2c5b22` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-smolvlm-vision-language` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 2024.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 36 files, 1062 MB staged from the Hub into a clean cache; comparison {cer: {empty: 1, constant: 0.944, frozen: 1.809, adapted: 0.89}, wer: {empty: 1, constant: 0.999, frozen: 2.215, adapted: 1.143}, cer_macro: {empty: 1, constant: 0.952, frozen: 2.078, adapted: 1.181}, exact_match: {empty: 0, constant: 0, frozen: 0.007, adapted: 0.014}, delta_vs_frozen: {cer: -0.919, wer: -1.072, cer_macro: -0.897, exact_match: 0.007}, hypothesis_length: {ref_chars: 6169, frozen_hyp_chars: 10515, adapted_hyp_chars: 5902, frozen_truncated: 14, adapted_truncated: 4}}; drawing / scene / page check frozen vs adapted {frozen: [(15, False), (20, False)], adapted: [(8, False), (20, False)]}; reload parity {identical_lines: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-smolvlm-vision-language/v3/evidence/` in the workspace |
| 2026-09-21 | package API at `ef38442` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-smolvlm-e2e` v3 — v1 and v2 of 2026-09-20 at `0723ea9` were the recipe probes that chose the eight-layer arm; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (7 passed, 14 warnings in 146.25s (0:02:26)) and the recipe probe: the eight pinned row groups read over range requests (800 lines, digest match), empty and constant baselines, frozen transcription instruction on the 140 test lines, `adapt(epochs=8, lr=1e-4, batch_size=8)` with validation-CER selection, adapted evaluation, artifact round trip | 2066 s | **PASS** — 7 passed, 14 warnings in 146.25s (0:02:26); the notebook's 8 code cells re-executed through the package API in 1706 s with peak CUDA memory 9.32 GB; the metrics it produced are the ones the notebook run above recorded (same seed, same split, same recipe) |
| 2026-09-13 | `fe478b69e92bcc144cb2a38e7f12e5bbe65a1873` / `8e97c78fc79e38124910c3d40850874ac5fb0194` (`TASK-INFERENCE`, superseded) | Colab CLI → fresh Python 3.12.3 venv/interpreter; Tesla T4, 15,360 MiB | Unchanged default sample, no repository checkout, empty per-model cache and weights | 194.3 s / 198.3 s | PASS — 8/8 cells; [retained run](verification/2026-09-13/README.md); not evidence for the `E2E` blob |

## Current status

**Release-grade.** The `E2E` notebook blob `5b2c5b22` (committed at `385b214`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (11/11 ok (1 restart after install cell), 2024.2 s, 36 files, 1062 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The pre-flight rows above (the package-API probe and the notebook pre-flight of the previous blob) and the superseded TASK-INFERENCE run are history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should weigh: the sample is nineteenth-century French cursive, far outside the checkpoint's printed and
scene-text training distribution, which is why the frozen transcription instruction sits near the empty baseline (the frozen model does not transcribe the lines: it answers in English about the handwriting, invents dates and sentences, and loops on one word until the 128-token budget — 14 of 140 generations truncated and 10,515 hypothesis characters for 6,169 reference characters, so the CER passes 1.0 by over-generation rather than silence) and why the gain is a
repair of a domain gap, not evidence about other hands or scripts; the rates are uncapped micro
CER/WER over one crowdsourced transcription and the notebook says so; the 60-line validation split selects the epoch;
the SigLIP encoder and the connector are frozen, so what they cannot resolve in a 128-px line tiled at 512 px
stays unread; the decoder that was tuned answers every prompt, and the drawing asked again after adaptation is the only
evidence about what happened to captioning and counting. Greedy decoding is deterministic on a fixed device and dtype, but the
training of four decoder layers is not bit-reproducible across GPUs, so a Kaggle number a few hundredths off the build
record is the expected spread, not a finding. The sibling rows on the same split — GOT-OCR 2.0 at 0.759 and Florence-2 at
0.797 CER with the same recipe — are compared in the card.
