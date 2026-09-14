# SmolVLM-500M-Instruct — Colab T4 execution evidence

Execution date (UTC): 2026-09-13. Outcome: **PASS — 8/8 unchanged code cells**, with GPU device `cuda:0`.

| Provenance | Value |
|---|---|
| Tested repository commit | `fe478b69e92bcc144cb2a38e7f12e5bbe65a1873` |
| Source notebook Git blob | `8e97c78fc79e38124910c3d40850874ac5fb0194` |
| Source notebook SHA-256 | `7ecfe91336154130a6a6523ebab2e0e0e0a3dbbeec85ac341bfeaa3c8ffdc405` |
| Embedded source revision | `f78ac0ce3cbbedfb6326a4f5f844134c3cb3c06c` |
| Model | `HuggingFaceTB/SmolVLM-500M-Instruct@a7da5b986cb59b408707209984f360a5f4ad7e47` |
| GPU / driver | `Tesla T4, 15360 MiB, 580.82.07` |
| Runtime | `{"device": "cuda:0", "dtype": "bfloat16", "pillow": "11.3.0", "python": "3.12.3", "source": "local-snapshot", "torch": "2.14.0+cu130", "transformers": "4.57.6"}` |
| Sum of code-cell wall times | 194.264 s |
| Total with environment setup and bookkeeping | 198.308 s |

## Execution method

Colab CLI 0.6.0 ran a driver from WSL `claude-science` on one Tesla T4 VM. The driver created a separate Python 3.12.3 virtual environment for this notebook and launched a new interpreter. Each original code cell was executed sequentially with `exec(compile(...))`; source cells and default form parameters were unchanged. The executor source is retained as [executor-source.txt](executor-source.txt), an evidence artifact rather than repository tooling. The VM had no repository checkout. The weights directory and isolated Hugging Face cache were empty before this notebook ran; all snapshot entries were downloaded and SHA-256 verified by the embedded pipeline.

The hosted Colab kernel used Python 3.13.15. An earlier direct `.ipynb` attempt was aborted during installation after that mismatch was confirmed. The successful result here uses the repository-supported Python 3.12 interpreter. A completed native hosted-kernel run is not claimed.

CLI transport prerequisite: PyPI `jupyter-kernel-client==1.0.2` lacked `KernelClient`; the CLI environment used Google’s fork at `f18e982c3265df5e923aa9def101ab3fd737e139` (distribution 0.8.0). This affects the CLI host, not the notebook runtime pins.

## Observations

- **Describe this image in one sentence.** The image contains a red square and a blue circle on a white background. (16 tokens, truncated=False, 7.803 s).
- **How many shapes are in the image, and what colour is each one?** There are two shapes in the image. Shape A is red, and shape B is blue. (20 tokens, truncated=False, 3.486 s).

All four output sanity checks passed. Both requests used greedy decoding and a 96-token ceiling. The evaluation verdict is `not-measurable`, with no metrics on this one synthetic image.

## Verification and retained files

The read-back checks confirmed the exact source hash and Git blob, unchanged code cells, eight error-free executed cells, runtime pins, CUDA inference, accepted inputs, and the recorded rejection probe. Model-specific sanity checks and exported artifacts were checked after download. See [execution-record.json](execution-record.json) for the individual checks and per-cell timings.

- [Executed notebook](smolvlm_vision_language_colab_output.ipynb)
- [Execution log](execution.log)
- [Installed distributions](packages-after.json)
- [Artifact SHA-256 manifest](artifacts.sha256)
- [smolvlm_vision_language_answers.csv](outputs/smolvlm_vision_language_answers.csv)
- [smolvlm_vision_language_evaluation_report.json](outputs/smolvlm_vision_language_evaluation_report.json)
- [smolvlm_vision_language_input_manifest.json](outputs/smolvlm_vision_language_input_manifest.json)
- [smolvlm_vision_language_result.json](outputs/smolvlm_vision_language_result.json)

The CLI stopped the shared runtime after all four notebook tests; a subsequent `colab sessions` call returned no active sessions. Both cleanup outputs are retained in the execution record. Repository release status remains **Candidate** pending evidence review; this record does not promote it.

## AI Assistance Disclosure

This tutorial and its accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
