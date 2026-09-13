"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "smolvlm_vision_language_pipeline",
    "repo_name": "smolvlm-vision-language-pipeline",
    "stem": "smolvlm_vision_language",
    "notebook_name": "smolvlm_vision_language_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "SmolVLMPipeline",
    "weights_key": "smolvlm-500m-instruct",
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "SmolVLM-500M-Instruct — DIMER image captioning and VQA tutorial (standalone)",
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
    "capability": "image + text to text generation (captioning and visual question answering) using the pinned SmolVLM-500M-Instruct weights",
    "intro": (
        "At inference one image and one user prompt are wrapped in the model's chat template, the processor resizes the "
        "image to a longest edge of 2048 px and splits it into 512 px tiles (64 visual tokens each), and the "
        "Idefics3-architecture model — a SigLIP-derived vision encoder feeding the SmolLM2-360M-Instruct text decoder, "
        "which is itself already a DIMER language-model profile — generates the assistant turn. Decoding is **greedy by "
        "default** (`do_sample=False`, `DECODING = \"greedy\"`), so a rerun on the same device, dtype and library versions "
        "reproduces the same text; `do_sample=True` is exposed for callers who want varied wording and gives up that "
        "determinism. The output is **free text with no score, no probability and no correctness signal**: the model "
        "writes fluent prose whether or not it is right — the model card's smoke run correctly named a red square but also "
        "claimed its corners touched the image edges, which they did not — so every answer must be read as a hypothesis. "
        "**No adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting — the "
        "pinned checkpoint is used as published. What upstream supplies is the model, processor and chat template; what "
        "the carried pipeline module adds is manifest verification, input validation and ceilings, prompt assembly, a "
        "fixed output contract with `new_tokens`/`truncated` run facts, and the `validate_inputs` and `evaluation_report` "
        "helpers."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, generate a synthetic image (or "
        "upload your own), stage and digest-verify the immutable upstream snapshot, surface the ceilings and the decoding "
        "contract and validate every prompt into one input manifest through the pipeline's own validation stage, run a "
        "captioning prompt and a question prompt through the public API with explicit generation settings, read the output "
        "contract correctly (including the truncation flag), understand from the evaluation report why the verdict is "
        "always `not-measurable` here and what labelled data a real evaluation needs, and export the answers plus "
        "provenance."
    ),
    "exclusions": (
        "multi-image or video input (`MAX_IMAGES` = 1), object detection or grounding with coordinates, OCR with layout, "
        "text-only chat, batched inference, fine-tuning, or any accuracy claim. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available (bfloat16 there, the dtype the checkpoint ships in); on the model card's CPU smoke the snapshot loaded in 5.7 s and 128 new tokens took 17.4 s, so the two-prompt default takes about a minute on a hosted CPU runtime. The pinned `torch==2.14.0` install and the 1.0 GB checkpoint are the largest downloads of the run.",
        "- **Knowledge:** basic Python and PIL image handling; what greedy decoding is and why free text has no intrinsic accuracy.",
        "- **Data:** the default sample is a synthetic image generated in code; BYOD is exactly one image file, gated off by default, any mode (converted to RGB), with both sides between 1 and `MAX_IMAGE_SIDE` = 4096 px. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded images remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Generate the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a 384 × 384 white canvas drawn in this cell with a filled red square in "
                "the upper left and a filled blue circle in the lower right — so it needs no download, contains no personal "
                "data, and is reproducible from code (no randomness, no seed; its pixel SHA-256 is printed and exported). Two "
                "prompts are asked about it: a captioning prompt and a counting question. The drawing has **no reference "
                "answers** the notebook asserts, so every answer it produces is smoke/sanity evidence that the code path works "
                "— you can judge the answers by eye, but that is a reading, not a measurement, and the model card's smoke run "
                "shows how a fluent answer can be wrong in detail.\n\n"
                "BYOD is optional and disabled by default. Edit the `PROMPTS` form field (one prompt per `|`, each at most "
                "`MAX_TEXT_CHARS` characters) to ask your own questions; the upload stays inside this runtime. Nothing is "
                "validated in this cell — the next section hands every prompt to the pipeline's own validation stage, which is "
                "the only checker."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "PROMPTS = 'Describe this image in one sentence. | How many shapes are in the image, and what colour is each one?'  # @param {{type:\"string\"}}\n"
                "MAX_NEW_TOKENS_PER_PROMPT = 96  # @param {{type:\"integer\"}}\n\n"
                "prompts = [prompt.strip() for prompt in PROMPTS.split('|') if prompt.strip()]\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    if len(uploaded) != 1:\n"
                "        raise ValueError(f'upload exactly one image, got {{len(uploaded)}}')\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    # Deterministic drawing: a red square and a blue circle on white.\n"
                "    image = Image.new('RGB', (384, 384), (255, 255, 255))\n"
                "    draw = ImageDraw.Draw(image)\n"
                "    draw.rectangle((48, 48, 176, 176), fill=(220, 30, 30))\n"
                "    draw.ellipse((208, 208, 336, 336), fill=(30, 60, 220))\n"
                "    image_name = 'synthetic_square_circle_384'\n"
                "    sample_kind = 'synthetic (drawn in this cell)'\n\n"
                "sample_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample': image_name, 'sample_kind': sample_kind, 'mode': image.mode, 'size': image.size, 'prompts': prompts, 'max_new_tokens_per_prompt': MAX_NEW_TOKENS_PER_PROMPT, 'pixel_sha256': sample_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Validate every prompt → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `generate` applies "
                "— exactly one image (`MAX_IMAGES` = 1) of type `PIL.Image.Image` with both sides in 1..`MAX_IMAGE_SIDE` px, a "
                "non-empty prompt of at most `MAX_TEXT_CHARS` characters, and `max_new_tokens` in 1..`MAX_NEW_TOKENS` — and "
                "returns an **input manifest** naming the schema and ceilings, the image's observed mode and size, the prompt "
                "and its length, and the exact generation settings (including whether decoding is `greedy` or `sampling`). One "
                "prompt is validated per entry, and the combined manifest — a top-level record plus one sub-manifest per prompt "
                "under `prompts` — is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks like, the "
                "cell also validates an over-long prompt and records the pipeline's own error message as a finding.\n\n"
                "**What the pipeline changes about your image:** the processor resizes it so the longest edge is 2048 px "
                "(aspect ratio preserved) and splits it into 512 px tiles, each becoming 64 visual tokens; nothing is cropped "
                "away. An answer that uses the whole `max_new_tokens` budget is reported as `truncated` — a cut-off answer, not "
                "a complete one."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MAX_IMAGES': MAX_IMAGES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "manifests = [validate_inputs(image, prompt, max_new_tokens=MAX_NEW_TOKENS_PER_PROMPT, do_sample=False, names=[image_name]) for prompt in prompts]\n"
                "input_manifest = {{**manifests[0], 'prompt': None, 'prompt_chars': None, 'n_prompts': len(manifests), 'findings': [], 'prompts': manifests}}\n"
                "# Demonstrate rejection on a prompt that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, 'x' * (MAX_TEXT_CHARS + 1))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-long-prompt-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Generate answers and interpret them\n\n"
                "Each call to `generate(images, prompt, *, max_new_tokens=..., do_sample=False)` returns `text` (the decoded "
                "assistant turn, special tokens stripped), the `prompt`, `image_size`, `new_tokens`, `truncated` (true when the "
                "answer used the whole `max_new_tokens` budget and was cut off), the `generation` settings actually used "
                "(`max_new_tokens`, `do_sample`, `decoding`), device, dtype, source and model identity. **Output semantics:** "
                "`text` is free-form generated language with **no score and no correctness signal**; a fluent, specific answer "
                "is not evidence that it is right. With greedy decoding the same inputs reproduce the same text on a fixed "
                "device, dtype and library version — that is a reproducibility property, not a quality one. The checks below "
                "are plumbing checks (text returned, settings as requested), and the truncation flag tells you whether an "
                "answer was cut. The printed seconds are measured on this runtime for this image and include the first-call "
                "warm-up."
            ),
            "code": (
                "import time\n\n"
                "results = []\n"
                "answers = []\n"
                "for prompt in prompts:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.generate(image, prompt, max_new_tokens=MAX_NEW_TOKENS_PER_PROMPT, do_sample=False)\n"
                "    results.append(result)\n"
                "    answers.append({{'prompt': prompt, 'text': result['text'], 'new_tokens': result['new_tokens'], 'truncated': result['truncated'], 'generation': result['generation'], 'seconds': round(time.perf_counter() - started, 3)}})\n"
                "    print({{'prompt': prompt, 'seconds': answers[-1]['seconds'], 'new_tokens': result['new_tokens'], 'truncated': result['truncated'], 'generation': result['generation']}})\n"
                "    print('answer:', result['text'])\n"
                "checks = {{\n"
                "    'one_answer_per_prompt': len(answers) == len(prompts),\n"
                "    'answers_are_text': all(isinstance(a['text'], str) and a['text'].strip() for a in answers),\n"
                "    'greedy_as_requested': all(a['generation']['do_sample'] is False and a['generation']['decoding'] == DECODING for a in answers),\n"
                "    'budget_as_requested': all(a['generation']['max_new_tokens'] == MAX_NEW_TOKENS_PER_PROMPT for a in answers),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'generate output failed a sanity check: {{checks}}')\n"
                "print({{'checks': checks, 'any_truncated': any(a['truncated'] for a in answers)}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report — and for this "
                "capability its verdict is **always `not-measurable`**. Open-ended image-conditioned generation has no "
                "intrinsic correctness signal, and this repository ships **no metric helper**, so there is nothing to compute "
                "and nothing is invented. The report records what was generated (each prompt, its `new_tokens` and whether it "
                "was `truncated`, and the generation settings), states the score semantics, and names what a real evaluation "
                "would need: labelled data matched to the use — question-answer pairs with reference answers for VQA accuracy, "
                "reference captions for a caption metric such as CIDEr, or document pages with gold answers for document-QA "
                "exact match — plus the caller's own scoring code over enough items to state a dispersion. You can compare the "
                "answers with what you see (a red square and a blue circle), but that reading is a sanity check on one "
                "drawing, not a measurement. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric is reported: free-text answers have no intrinsic correctness signal and the repository ships no metric helper.')"
            ),
        },
        {
            "md": (
                "## 8. Export answers and provenance\n\n"
                "Two files are written under `outputs/`: `{stem}_result.json` with an `answers` list carrying, per prompt, the "
                "prompt, the generated text, `new_tokens`, `truncated`, the generation settings and seconds (so every answer "
                "maps back to its prompt and the image), the plumbing checks, the evaluation report, the input manifest, the "
                "ceilings in force, the sample identity (name, kind, size, pixel digest), the notebook's source (repository, "
                "revision, embedded module digest, generator), the model identifier, the immutable model revision, the model "
                "licence, and the runtime identity (Python, `torch`, `transformers`, Pillow, device, dtype); and "
                "`{stem}_answers.csv` with explicit `index`, `prompt`, `new_tokens`, `truncated` and `text` columns so prompt "
                "order survives downstream use. No credentials are involved in any step, so none can reach the export."
            ),
            "code": (
                "import csv\n\n"
                "payload = {{\n"
                "    'answers': [{{'index': index, **answer}} for index, answer in enumerate(answers)],\n"
                "    'sanity_checks': checks,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'ceilings': {{'MAX_IMAGES': MAX_IMAGES, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_TEXT_CHARS': MAX_TEXT_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}},\n"
                "    'sample': {{'name': image_name, 'kind': sample_kind, 'width': image.width, 'height': image.height, 'pixel_sha256': sample_sha256}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'pillow': PIL.__version__,\n"
                "        'device': pipe.device,\n"
                "        'dtype': pipe.dtype,\n"
                "        'source': pipe.source,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['index', 'prompt', 'new_tokens', 'truncated', 'text'])\n"
                "    for index, answer in enumerate(answers):\n"
                "        writer.writerow([index, answer['prompt'], answer['new_tokens'], answer['truncated'], answer['text']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The answers are generated language, not measurements: they carry no score, no probability and no signal of "
        "correctness, and the model can state details that are not in the image — the model card's smoke run named a red "
        "square correctly and then invented that its corners touched the edges. On the synthetic drawing the answers are "
        "plumbing evidence only; the evaluation report is `not-measurable` because no metric exists without labelled "
        "question-answer pairs or reference captions, and a real evaluation needs such a set in your domain plus your own "
        "scoring code. The pipeline accepts one image and one prompt per call, resizes the longest edge to 2048 px and tiles "
        "it, caps answers at `MAX_NEW_TOKENS` (a `truncated` answer is cut, not complete), and exposes no grounding "
        "coordinates, no OCR layout, no text-only chat and no batching. Greedy decoding is deterministic on a fixed device "
        "and dtype (CPU float32 and CUDA bfloat16 can produce different text); `do_sample=True` trades that determinism for "
        "varied wording. The text decoder, SmolLM2-360M-Instruct, is the same model behind the DIMER language-model profile "
        "of that name, so its language-side limits (small model, English-centric instruction tuning) apply here too.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model snapshot, validate the demonstrated input against the enforced ceilings, "
        "execute the public pipeline path with explicit greedy generation settings, and emit the shown machine-readable "
        "outputs in the tested runtime — without the repository being reachable. It does **not** establish benchmark "
        "superiority, captioning or VQA accuracy on any domain, freedom from hallucination, safety for high-consequence "
        "decisions, or production fitness on an unseen domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/smolvlm-500m-instruct/` and rerun Section 3. A `ValueError` naming "
        "`MAX_IMAGE_SIDE`, `MAX_TEXT_CHARS` or the image count in Section 5: fix the BYOD input or the `PROMPTS` field and "
        "rerun from Section 4. `truncated: True` on an answer: raise `MAX_NEW_TOKENS_PER_PROMPT` in Section 4 (up to "
        "`MAX_NEW_TOKENS`). Slow generation on a CPU runtime is expected (about 0.14 s per token on the card's "
        "machine).\n\n"
        "**Next experiments.** Upload a photograph and ask a question whose answer you know, then ask a question whose "
        "answer is not in the image and watch whether the model declines or invents one; rerun the default prompts with "
        "`do_sample=True` to see wording vary while the greedy run stays fixed; run the same prompts on a CUDA runtime and "
        "diff the bfloat16 answers against the CPU float32 ones. None of these turns the sample result into evidence of "
        "production fitness.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/smolvlm-vision-language-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/huggingface/smollm\n"
        "- SmolVLM paper: https://arxiv.org/abs/2504.05299\n"
        "- Text decoder (DIMER language-model profile): https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct"
    ),
}
