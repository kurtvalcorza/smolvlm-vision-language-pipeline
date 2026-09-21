# SmolVLM Vision-Language Pipeline

DIMER pipeline for **`HuggingFaceTB/SmolVLM-500M-Instruct`** — image + text chat generation (captioning, visual question answering, document reading) — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. On top of inference it carries the **adaptation contract for one instruction** (`Transcribe the handwritten text in this image.`): corpus-level character and word error rates over transcribed lines, two non-adapted baselines, a bounded fine-tuning of the last eight SmolLM2 decoder layers on cached prefix hidden states, and a verified safetensors adapter that reloads against the pinned base.

## Upstream alignment

- Model: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`
- Upstream weight license: Apache-2.0
- Upstream task: image-text-to-text (Idefics3 architecture; SmolLM2-360M-Instruct decoder, SigLIP-derived encoder), English
- Repository adaptation: **bounded supervised fine-tuning of one instruction** — the last eight of the 32 SmolLM2 decoder layers and the final norm (78,659,520 of 507,482,304 parameters) on `{id, image, text}` line records with the causal language-model loss over the assistant turn; the SigLIP vision encoder, the connector, the embeddings, the output head and the first 24 decoder layers stay frozen. Trained tensors are exported as a safetensors adapter with a manifest and overlaid on a freshly loaded, re-verified base. Every other prompt is inference-only; the tuned decoder answers them too, which the tutorial shows on one drawing and the card records.

## Quick start

```python
from PIL import Image
from smolvlm_vision_language_pipeline import SmolVLMPipeline

pipe = SmolVLMPipeline.from_pretrained()                # cuda:0 when visible, else cpu; float32 everywhere
result = pipe.generate(Image.open("photo.jpg"), "Describe the image.", max_new_tokens=128)
print(result["text"], result["truncated"], result["generation"])

from smolvlm_vision_language_pipeline import TRANSCRIBE_PROMPT, fetch_sample_dataset
splits = fetch_sample_dataset()                    # 800 digest-pinned Belfort handwritten lines, 600 / 60 / 140
print(pipe.evaluate(splits["test"])["cer"])        # frozen corpus CER for TRANSCRIBE_PROMPT
pipe.adapt(splits["train"], splits["validation"])  # last eight decoder layers, lowest-validation-CER epoch kept
print(pipe.evaluate(splits["test"])["cer"])
pipe.save_artifact("outputs/adapter")
again = SmolVLMPipeline.from_artifact("outputs/adapter")   # re-verifies the base, checks the manifest, overlays
```

`text` is free-form generated prose with no confidence attached; `generation` echoes `max_new_tokens`, `do_sample` and the decoding mode (`greedy` unless `do_sample=True`). `transcribe` and `evaluate` ask one prompt of many images in left-padded batches (the tile count and hence the prompt length vary with the image). Image ceilings: sides within 1..16,384 px and at most 4096² pixels (the processor resizes to a 2048-px longest edge and tiles, so the ceilings bound decode and resize memory, not model cost).

## Adaptation contract

- **Records:** `{id, image, text}` — a PIL image (sides within the ceilings) and its transcript (1..512 characters after whitespace runs are collapsed); `validate_dataset` checks the structure, `split_dataset` de-duplicates by decoded pixels and `check_split_disjoint` asserts no image is shared. The default sample (`samples.py`) is the first eight parquet row groups of the Belfort-line test shard (`Teklia/Belfort-line`, MIT; nineteenth-century French council minutes in cursive) read over HTTPS range requests at an immutable Hub revision, each row group refused on any SHA-256 or byte-total mismatch — the same digest-pinned sample and split as the sibling `got-ocr2-pipeline` and `florence2-vision-language-pipeline` rows; `load_byod_dataset` reads a zip or directory of line images plus `transcripts.csv`.
- **Measures (`metrics.py`):** `ocr_metrics` — micro CER and WER (total edits over total reference characters or words), macro rates, exact match, and the hypothesis length; uncapped, so a rate above 1.0 means the model generates text the line does not carry. `empty_baseline` (CER 1.0 by construction) and `constant_baseline` (the medoid training transcript for every line).
- **Fine-tuning:** `adapt(train, val, *, prompt=TRANSCRIBE_PROMPT, epochs=8, lr=1e-4, batch_size=8, seed=0)` caches the hidden states entering decoder layer 24 for every training line (the tiles' visual tokens, the chat-templated user turn and the assistant turn, one frozen forward each), then trains layers 24–31 and the final norm on those states with the causal LM loss over the assistant turn, AdamW (no weight decay), gradient clipping at 1.0 and seeded shuffling; the loss equals the full model's loss exactly. Epoch 0 records the frozen validation rates; the epoch with the lowest validation CER is kept; on any exception the frozen weights are restored.
- **Artifact:** `save_artifact` writes `adapter.safetensors` (about 315 MB) + `manifest.json` (`org.valcorza.smolvlm-500m-instruct.adapter.v1`: base identity and weight digest, tensor names, file size and SHA-256, the instruction, configuration, history); `from_artifact` re-verifies the base and checks the manifest, digest and exact tensor set before deserialising.
- **Build record (Tesla T4, seed 42 split):** frozen CER 1.809 / WER 2.215 on the 140 held-out lines (the frozen model does not transcribe the lines: it answers in English about the handwriting, invents dates and sentences, and loops on one word until the 128-token budget — 14 of 140 generations truncated and 10,515 hypothesis characters for 6,169 reference characters, so the CER passes 1.0 by over-generation rather than silence), adapted **0.890** / **1.143** (epoch 7 of 8, 2 lines exact), reload parity 8/8; the drawing's two prompts after adaptation: before adaptation `Describe this image in one sentence.` → `Two shapes are present. One is red and the other is blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.`; after adaptation `Describe this image in one sentence.` → `Two colors, red and blue.`; `How many shapes are in the image, and what colour is each one?` → `There are two shapes in the image. Shape A is red, and shape B is blue.`. One seeded split of one 800-line sample; no dispersion estimate. The sibling rows on the same split: GOT-OCR 2.0 0.759, Florence-2 0.797.

## Weights layout

```
weights/smolvlm-500m-instruct/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + sha256 (verified on every load)
  config.json                # Idefics3ForConditionalGeneration, text/vision sub-configs
  preprocessor_config.json   # longest_edge 2048, 512-px tiles, mean/std 0.5
  processor_config.json, chat_template.json, tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
  added_tokens.json, special_tokens_map.json, generation_config.json, README.md
  model.safetensors          # 1015025832 bytes, git-ignored
```

`from_pretrained()` calls `stage_missing_files()` then `verify_snapshot()` and refuses to load if any file is missing or its SHA-256 differs from the manifest; the snapshot is then loaded with `local_files_only=True` and `trust_remote_code=False`. Without a snapshot, `allow_download=True` loads from the Hub at `revision=a7da5b986cb59b408707209984f360a5f4ad7e47`; the default is to refuse. To stage the snapshot: `hf download HuggingFaceTB/SmolVLM-500M-Instruct --revision a7da5b986cb59b408707209984f360a5f4ad7e47 --local-dir weights/smolvlm-500m-instruct`, then write the manifest.

## Tests and smoke

```
pip install -e . --no-deps
pytest -q -o addopts= tests      # offline, no weights needed (tests/test_model_backed.py runs only where the snapshot is staged)
python -c "from PIL import Image, ImageDraw; from smolvlm_vision_language_pipeline import SmolVLMPipeline; im = Image.new('RGB', (256, 256), 'white'); ImageDraw.Draw(im).rectangle([64, 64, 192, 192], fill='red'); print(SmolVLMPipeline.from_pretrained(device='cpu').generate(im, 'Describe the image.')['text'])"
```

Measured on CPU (float32, Windows venv, 2026-09-12): load 5.71 s, 128 tokens in 17.40 s; the answer named a red square on a white background.

## Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/tutorials/smolvlm_vision_language_colab.ipynb)

`tutorials/smolvlm_vision_language_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, the model identity, the manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path stages and verifies the pinned snapshot, fetches the eight pinned Belfort row groups and splits the 800 lines 600 / 60 / 140, asks two prompts of a synthetic drawing through the inference contract, measures the frozen transcription instruction's CER and WER on the held-out lines beside the empty and constant baselines, runs `adapt` with validation-CER epoch selection, scores the held-out lines again, writes six line panels and asks the drawing again with the adapted model, and exports the adapter and reloads it with verified transcript parity. BYOD is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Release-grade** — the `E2E` notebook blob `5b2c5b22` (committed at `385b214`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-21 (11/11 ok (1 restart after install cell), 2024.2 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documents

- [`MODEL_CARD.md`](MODEL_CARD.md) — MODEL_CARD_SPEC 1.1 card
- [`docs/WEIGHTS.md`](docs/WEIGHTS.md) — weight provenance and hosting
- [`STATUS.md`](STATUS.md) — release status

## Licensing

Repository code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; the Belfort-line sample is MIT and is not redistributed; see `docs/WEIGHTS.md`.

Current source update: snapshot validation now runs before model-library imports (Kokoro also validates the language first), so rejected requests fail with the intended validation error even when model libraries are absent. The standalone notebook was regenerated from this source. The retained 2026-09-13 GPU run identifies the earlier notebook blob; the regenerated notebook has not had a fresh GPU execution. Status remains **Candidate**.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
