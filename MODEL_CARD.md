---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-text-to-text
base_model: HuggingFaceTB/SmolVLM-500M-Instruct
---

# SmolVLM-500M-Instruct (DIMER package v0.1.0) — Vision-Language Model (Image + Text Chat Generation)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HuggingFaceTB%2FSmolVLM--500M--Instruct-ffcc4d?style=flat)](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-huggingface%2Fsmollm-181717?style=flat&logo=github&logoColor=white)](https://github.com/huggingface/smollm)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2504.05299-b31b1b.svg)](https://arxiv.org/abs/2504.05299)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-smolvlm--vision--language--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent). The package is exercised through its test suite (`tests/`) and the run instructions in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up, not a claim this card makes.

---

###### Description

`HuggingFaceTB/SmolVLM-500M-Instruct` is the 500 M-parameter instruction-tuned member of Hugging Face's SmolVLM family (Marafioti et al., arXiv:2504.05299), pinned here to revision `a7da5b986cb59b408707209984f360a5f4ad7e47`. It is an Idefics3-architecture model (`config.json` `model_type: idefics3`): a 93 M-parameter SigLIP-derived vision encoder (12 layers, hidden 768, 512-px input, 16-px patches) whose patch features are pixel-shuffled with `scale_factor` 4 into 64 visual tokens per 512×512 tile and projected into a 32-layer, 960-d Llama-style decoder with a 49280-token vocabulary and 8192-position context — the text decoder is SmolLM2-360M-Instruct, which DIMER already hosts as its own profile (upstream README `base_model`). At inference the processor resizes an image to a longest edge of 2048 px, splits it into 512-px tiles plus a global view, interleaves `<image>` tokens with the chat-templated prompt, and the decoder generates an answer token by token; adaptation is by prompt only — no training or in-context examples happen in this repository. What this repository adds is packaging: `SmolVLMPipeline` in `src/smolvlm_vision_language_pipeline/pipeline.py`, digest verification of the local snapshot (`verify_snapshot`, `stage_missing_files`), input validation, a single-image single-turn chat contract with the generation settings echoed in every result, and a CPU smoke run; it exposes no metric because none can be computed without a labelled VQA set.

#### Intended Use and Limitations

###### Primary Intended Uses

The task is image-conditioned text generation: input one PIL image and one user prompt of up to `MAX_TEXT_CHARS = 2000` characters; output the decoded assistant turn (`text`), the number of new tokens, a `truncated` flag and the decoding settings. Envisioned applications are image captioning and description for accessibility or indexing, short visual question answering ("what colour is the sign?"), reading text or charts in documents (upstream states 25 % of the training mixture is document understanding and 18 % captioning), and on-device or CPU-bound assistants where a sub-1 B model is wanted. In a larger system the pipeline is an inference component producing free text for a human reader or a downstream extractor, not a decision engine; its `text` field carries no confidence and must be treated as a draft.

###### Primary Intended Users

The intended users are machine-learning engineers, application developers and researchers integrating a small vision-language model into research prototypes, internal enterprise tooling, or the DIMER model workbench. The pipeline assumes its users understand that generated text is not a verified fact about the image — the smoke run below produced a fluent, mostly correct description that also asserted a geometric detail the image does not contain — that greedy decoding is deterministic but not "correct", that English is the only supported language (upstream README `language: en`), and that any deployment needs a labelled evaluation set of its own images and questions. It is not designed for hobbyist "ask and trust" use.

###### Out-of-scope use cases

1. **Capability boundary:** not object detection, segmentation, grounding with coordinates, OCR with layout, image generation, or multi-turn dialogue (v0.1.0 renders exactly one user turn); `florence2-vision-language-pipeline` is the sibling for boxes and region OCR. Non-English prompts and answers are unsupported. Video is unsupported.
2. **Input boundary:** exactly `MAX_IMAGES = 1` image per call (`ValueError` otherwise); only `PIL.Image.Image` (`TypeError` otherwise); any side above `MAX_IMAGE_SIDE = 4096` px or below 1 px is rejected; prompts above 2000 characters or empty are rejected; `max_new_tokens` is capped at `MAX_NEW_TOKENS = 512` (default 128), so long answers are cut and flagged `truncated: true`. The processor resizes the longest edge to 2048 px and tiles at 512 px (`preprocessor_config.json`), so fine print in very large images is lost.
3. **Decision boundary:** not for autonomous or high-impact decisions — content moderation, medical image reading, identity or document verification, safety monitoring — without a human reviewing each answer and a locally measured error rate; upstream's own card rules out "critical automated decision-making".

#### Factors

###### Groups

The pipeline is not human-centric by design — it answers questions about arbitrary images — but the model will describe people when they appear, and the training mixtures (The Cauldron and Docmatix, upstream README) aggregate many public vision-language datasets that contain photographs of people. Neither the upstream card nor the SmolVLM paper reports caption or VQA accuracy broken down by age, gender, skin type or region, and the aggregated corpus is not group-audited. The obligation transfers to the operator: before deployment, measure answer accuracy on a labelled sample of their own images and questions stratified by the groups relevant to their application (for example description quality across skin tones, or refusal behaviour on images of children) and treat a material gap as a blocker. This repository measures nothing of the kind.

###### Instrumentation

The Cauldron and Docmatix are curated from public datasets — photographs, screenshots, rendered documents, charts and scanned PDFs — of undocumented camera, scanner, rendering and compression provenance; Docmatix pairs rendered document pages with synthetic questions, so part of the "instrument" is a generation pipeline rather than a sensor. The pipeline consumes decoded pixel arrays: resolution, JPEG artefacts, rotation, skew and colour profile reach the model as changed pixel statistics after the longest-edge-2048 resize and 512-px tiling with bilinear resampling (`resample: 1`) and mean/std 0.5 normalisation. The pipeline does not detect blur, low contrast, rotation or a changed capture device; it only rejects non-image types and sides outside 1–4096 px. Document-reading deployments should validate on the scanner and DPI actually in use.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `pillow==11.3.0` (exact pins in `pyproject.toml`). CUDA is optional: `from_pretrained` picks `cuda:0` when available with bfloat16, else CPU with float32; the DIMER build environment is CPU-only (`CUDA_VISIBLE_DEVICES=-1`). On this repository's smoke run (Windows venv, CPU, float32, one 256×256 synthetic white image with a centred red square, prompt "Describe the image.") loading the verified snapshot took 5.71 s and generating 128 tokens took 17.40 s (23.11 s total); a second identical call returned byte-identical text. The CUDA/bfloat16 path is not executed in this repository. Data environment: inputs are assumed to be natural photographs, screenshots or document images resembling The Cauldron/Docmatix mixture, upright, with the subject legible at 512-px tiles; medical, satellite, line-art or heavily rotated inputs fall outside that assumption and degrade in ways the pipeline does not measure.

#### Metrics

###### Performance Measures

The pipeline reports no performance measure: open-ended generation has no intrinsic correctness signal, and VQA accuracy, CIDEr for captions or document-QA exact match each need a labelled evaluation set with reference answers that the caller must supply. The code exposes only run-level facts — `new_tokens` and `truncated` — which describe the generation, not its quality. Upstream publishes its benchmark results in the pinned README only as an embedded image (no table), so no number is reproducible from the snapshot text; this pipeline has not reproduced any upstream evaluation and quotes none as its own. A caller who needs a number should run a labelled VQA or caption set through `generate` and score `text` against references with an external scorer, stating the scorer and its normalisation.

###### Decision thresholds

The default decision rule is greedy decoding: `do_sample=False` (`DECODING = "greedy"`), so at every step the highest-probability token wins — an implicit argmax over the 49280-token vocabulary with no minimum probability — and generation stops at end-of-utterance or at `max_new_tokens`. No acceptance threshold on answer quality was set during development and none is shipped: the model emits no confidence, so there is nothing to threshold. Passing `do_sample=True` switches to multinomial sampling at the library's default temperature and makes outputs vary run to run; this is echoed as `decoding: "sampling"`. A deployment that needs to reject weak answers must build its own check — a reference-answer match, a second-model judge, or a human review queue — trading the cost of accepting a wrong answer (false positive) against the cost of discarding a right one (false negative).

###### Approaches to uncertainty and variability

This pipeline reports no accuracy number, so there is no estimation procedure or dispersion to state; upstream figures are point estimates by the upstream authors with no reported interval. With the default greedy decoding, output is deterministic given the same weights, device, dtype and library versions — confirmed on the smoke run, where two consecutive calls on the same image and prompt returned identical text — and no seed is needed; CPU float32 versus GPU bfloat16 can change near-tied tokens and hence the wording. With `do_sample=True` the output is stochastic and the caller must set `torch.manual_seed` themselves; the pipeline does not seed. No probability or confidence is emitted for the answer; a caller who needs one must read token log-probabilities through the library's `output_scores` path and calibrate them on labelled data, neither of which this repository does.

#### Ethical considerations and biases

###### Data

Upstream states the model was trained on The Cauldron and Docmatix "with emphasis on document understanding (25 %) and image captioning (18 %)" (pinned README), on top of the SmolLM2-360M-Instruct text decoder and a SigLIP vision encoder whose own pretraining corpora are disclosed only at the level of their respective cards; the disclosure stops there — no per-image licensing, consent status or demographic composition is given for the aggregated datasets, and public vision-language corpora are known to contain photographs of identifiable people and documents with personal information, so the presence of personal data is not ruled out. This repository distributes code, tests and documentation; the 1.02 GB `model.safetensors` snapshot is git-ignored and staged locally under `weights/smolvlm-500m-instruct/` with a manifest, and no sample data is shipped. The operator must audit the images and prompts they submit for personal, confidential or proprietary content — the model will read and repeat text visible in an image — and the pipeline performs no such check.

###### Human Life

The pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, housing or any other domain central to human life, and it has not been validated or certified for any of them by anyone; upstream's card likewise prohibits "evaluating or scoring individuals" and "critical automated decision-making". Its only validation is the offline unit suite (15 tests) and one CPU smoke run in this repository. Where a sensitive use is foreseeable — describing medication packaging, reading identity documents, summarising a medical chart image — it is admissible only with a human reviewer on every consequential answer, an independent domain evaluation on representative data, and whatever regulatory clearance the domain requires.

###### Mitigations

Implemented and inspectable in `src/smolvlm_vision_language_pipeline/pipeline.py`: (1) supply chain — `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` re-hashes every file in `weights/smolvlm-500m-instruct/dimer-base-manifest.json` (14 entries) and raises on the first size or SHA-256 mismatch before any weight is loaded; `stage_missing_files` fetches only manifest-listed files at the pinned revision and refuses a manifest naming another model; the local path is loaded with `local_files_only=True` and the Hub path only with `allow_download=True` and `revision=MODEL_REVISION`; `trust_remote_code=False` on both loaders (the Idefics3 classes are native to `transformers`). (2) Input integrity — `_validate` rejects non-PIL input, more or fewer than one image, sides outside 1–4096 px, non-string, empty or over-long prompts, and `max_new_tokens` outside 1–512 before the model runs; malformed runner output raises `RuntimeError`. (3) Reproducibility — exact `==` pins, `model.eval()`, greedy decoding by default with the settings echoed in every result, `model_id`/`model_revision` in every result. (4) Refusals — no multi-image, multi-turn, training or fine-tuning API is exposed; a missing snapshot with `allow_download=False` raises `FileNotFoundError`. No statistical mitigation is applied because the pipeline does not train.

###### Risks and harms

Fluent hallucination: the model produces confident prose that can be wrong in detail — on the smoke run it correctly named a red square on a white background but also stated the square's corners touch the image edges, which they do not — and the operator or reader bears the harm when such text is trusted; likelihood is high on any image with fine detail. Repetition and truncation: greedy decoding can loop, and answers cut at `max_new_tokens` end mid-sentence (`truncated: true` on the smoke run). Bias from aggregated web-derived data: descriptions of people may carry stereotyped or demeaning language and accuracy may vary by demographic group; data subjects and third parties bear that harm. Prompt injection through images: text visible in an image can steer the answer. Automation bias: a fluent caption is checked less carefully than a raw image. Privacy: the model transcribes personal data visible in an image. Magnitude ranges from a wrong alt-text to a wrongly summarised document.

###### Use cases

The pipeline must not be used for surveillance, biometric or demographic profiling, or social scoring — including describing, identifying or inferring attributes of individuals for tracking or classification — nor for "unauthorized surveillance", "harassment or abuse", "spam generation" or "disinformation campaigns", which the upstream card lists as malicious uses. It must not support unlawful discrimination in employment, housing, credit, insurance, education or healthcare access, nor deceptive or manipulative applications such as generating fabricated descriptions presented as evidence of what an image contains. Any use that violates the Apache-2.0 terms of the upstream weights or the DIMER deployment terms is prohibited. The developers identify these because free-text description of arbitrary images is exactly the primitive such misuses need; no further prohibited use is identified beyond them.

## Immutable provenance

- Model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`
- Snapshot manifest: `weights/smolvlm-500m-instruct/dimer-base-manifest.json`, 14 files, `totalBytes` 1019893574
- `model.safetensors` SHA-256: `d05b567eeaf534e83d375551f068ed57b5f52d37c657197f644af5ef9db091a2` (1015025832 bytes)
- `config.json` SHA-256: `daacbbca6af3c34e50466c72aed2df4553a08084f6a0551d4ae420a241cb66c6` (7339 bytes)
- Weight format: SafeTensors (bfloat16 as shipped); loader `AutoModelForImageTextToText.from_pretrained(<snapshot dir>, local_files_only=True, trust_remote_code=False)` resolving to `Idefics3ForConditionalGeneration`, with `AutoProcessor` → `Idefics3Processor`

## Input/output contract

- `SmolVLMPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)`
- `generate(images, prompt, *, max_new_tokens=128, do_sample=False)` — `images`: one `PIL.Image.Image` or a list of exactly one; sides 1–4096 px; any mode (converted to RGB). `prompt`: non-empty `str` ≤ 2000 characters. Returns `{"text", "prompt", "image_size", "new_tokens", "truncated", "generation": {"max_new_tokens", "do_sample", "decoding"}, "device", "dtype", "source", "model_id", "model_revision"}`.
- `build_messages(prompt)` — the one-image, one-text user turn passed to the snapshot's chat template.
- `verify_snapshot(path=None)`, `stage_missing_files(path=None, *, allow_download=False, downloader=None)`.
- No metric helper: evaluation needs caller-supplied references and an external scorer.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12. The build venv carries `torch 2.14.0+cu130`.
- Precision: float32 on CPU (the measured path); bfloat16 on CUDA (not executed here). Preprocessing per `preprocessor_config.json`: longest edge 2048, 512-px tiles, bilinear, mean/std 0.5; 64 visual tokens per tile (`processor_config.json` `image_seq_len`).
- Measured (Windows venv `dimer-next16`, CPU, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, 2026-09-12): device `cpu`, dtype `float32`, source `local-snapshot`; load 5.71 s, generate 17.40 s for 128 new tokens (`truncated: true`), total 23.11 s; output began "The image depicts a simple, two-dimensional geometric shape. The shape is a square … colored in a bright red hue … against the white background"; a repeat call returned identical text; exit 0.
- Tests: `pytest -q -o addopts= tests` — 15 passed, offline, no weights required; `ruff check src tests` clean.

## References

- Marafioti, Zohar, Farré, Noyan, Bakouch, Cuenca, Zakka, Ben Allal, Lozhkov, Tazi, Srivastav, Lochner, Larcher, Morlon, Tunstall, von Werra, Wolf. SmolVLM: Redefining small and efficient multimodal models. arXiv:2504.05299 (2025). https://arxiv.org/abs/2504.05299
- Idefics3 architecture reference cited by the upstream card: https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3
- Upstream card: https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct; source and fine-tuning notebook: https://github.com/huggingface/smollm
- Training data: https://huggingface.co/datasets/HuggingFaceM4/the_cauldron, https://huggingface.co/datasets/HuggingFaceM4/Docmatix
- Text decoder profile: `HuggingFaceTB/SmolLM2-360M-Instruct` (DIMER language-model pipeline)
