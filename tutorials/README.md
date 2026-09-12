# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/tutorials/smolvlm_vision_language_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-HuggingFaceTB%2FSmolVLM--500M--Instruct-ffcc4d?style=flat)](https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct)
[![Upstream](https://img.shields.io/badge/Upstream-huggingface%2Fsmollm-181717?style=flat&logo=github&logoColor=white)](https://github.com/huggingface/smollm)
[![arXiv](https://img.shields.io/badge/arXiv-2504.05299-b31b1b.svg)](https://arxiv.org/abs/2504.05299)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `smolvlm_vision_language_colab.ipynb` | `TASK-INFERENCE` | SmolVLM-500M-Instruct image captioning and VQA on one synthetic drawing with two prompts; greedy deterministic decoding with an explicit token budget and `truncated` flag; free-text output with no score; no metric exists or is reported; hallucination risk stated | CPU float32 (CUDA bfloat16 used automatically when available) | one image file plus a `PROMPTS` form field, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `SmolVLMPipeline` from the repository public API rather than reimplementing model inference; the pipeline pins the immutable upstream revision, stages the missing weight file through `stage_missing_files(..., allow_download=True)`, loads only from a digest-verified local snapshot (`verify_snapshot`) with `trust_remote_code=False`, and wraps the prompt in the chat template. The notebook never calls `transformers`, `huggingface_hub`, `apply_chat_template` or `.generate(**` directly.
- Generation semantics (INF8/INF9/UNC1): greedy decoding (`do_sample=False`, `DECODING`) is the demonstrated default and stated as deterministic on a fixed device/dtype; `max_new_tokens` is explicit per call and the `truncated` flag is surfaced; the output is free text with no score or correctness signal, and the model card's hallucination observation (invented corner placement) is quoted.
- No intrinsic metric exists (EVAL9): the repository ships no metric helper; the notebook says so and names what a real evaluation needs (QA pairs with reference answers for VQA accuracy, reference captions for CIDEr, document-QA gold answers). Recorded `SHOULD` deviation: EVAL11 (no baseline — none is meaningful for open-ended generation without labelled data).
- Ceilings `MAX_IMAGES` = 1, `MAX_IMAGE_SIDE`, `MAX_TEXT_CHARS`, `MAX_NEW_TOKENS`/`DEFAULT_MAX_NEW_TOKENS` and `DECODING` are surfaced before the model runs; the processor's longest-edge-2048 resize and 512-px tiling are stated (DAT22/DAT23).
- The text decoder's identity as the existing DIMER `SmolLM2-360M-Instruct` language-model profile is cross-referenced (ID5).
- The default sample is a synthetic drawing generated in code; `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
