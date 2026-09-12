"""Image + text -> text chat generation with the pinned ``HuggingFaceTB/SmolVLM-500M-Instruct`` snapshot.

The class loads weights only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly
allowed, from the Hugging Face Hub at the pinned revision. One image and one user text turn are rendered
through the snapshot's chat template; decoding is greedy unless ``do_sample=True`` is passed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
MODEL_REVISION = "a7da5b986cb59b408707209984f360a5f4ad7e47"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "smolvlm-500m-instruct"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHTS_FILE = "model.safetensors"
CONFIG_FILE = "config.json"

MAX_IMAGE_SIDE = 4096  # pixels; the processor resizes to longest_edge 2048 and splits into 512-px tiles
MAX_IMAGES = 1  # v0.1.0 accepts exactly one image per call
MAX_TEXT_CHARS = 2000  # characters of user prompt accepted
MAX_NEW_TOKENS = 512  # hard ceiling for `max_new_tokens`
DEFAULT_MAX_NEW_TOKENS = 128
DECODING = "greedy"  # do_sample=False by default -> deterministic on a fixed device/dtype


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its manifest; raise naming the first mismatch."""
    root = Path(path or DEFAULT_WEIGHTS_DIR)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"snapshot manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest.get("files", []):
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {"path": str(root), **manifest}


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def build_messages(prompt: str) -> list[dict[str, Any]]:
    """One user turn with one image placeholder followed by the text, in the snapshot chat-template shape."""
    return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]


@dataclass
class SmolVLMPipeline:
    """``_runner(image, prompt, max_new_tokens, do_sample)`` returns ``{"text": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SmolVLMPipeline:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if resolved_device.startswith("cuda") else torch.float32
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        processor = AutoProcessor.from_pretrained(location, **common)
        model = AutoModelForImageTextToText.from_pretrained(location, dtype=dtype, **common)
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, prompt: str, max_new_tokens: int, do_sample: bool) -> dict[str, Any]:
            text = processor.apply_chat_template(build_messages(prompt), add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
            new_ids = generated[0, inputs["input_ids"].shape[1] :]
            decoded = processor.batch_decode(new_ids.unsqueeze(0), skip_special_tokens=True)[0]
            return {"text": decoded, "new_tokens": int(new_ids.shape[0])}

        return cls(runner, resolved_device, str(dtype).removeprefix("torch."), source)

    def _validate(self, images: Any, prompt: Any, max_new_tokens: int) -> list[Image.Image]:
        if isinstance(images, Image.Image):
            images = [images]
        if not isinstance(images, list | tuple):
            raise TypeError("images must be a PIL.Image.Image or a list of them")
        if not 1 <= len(images) <= MAX_IMAGES:
            raise ValueError(f"image count must be between 1 and MAX_IMAGES={MAX_IMAGES}, got {len(images)}")
        for image in images:
            if not isinstance(image, Image.Image):
                raise TypeError(f"each image must be a PIL.Image.Image, got {type(image).__name__}")
            width, height = image.size
            if width < 1 or height < 1 or max(width, height) > MAX_IMAGE_SIDE:
                raise ValueError(f"image side outside 1..MAX_IMAGE_SIDE={MAX_IMAGE_SIDE} px: {image.size}")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a str")
        if not prompt.strip():
            raise ValueError("prompt must not be empty")
        if len(prompt) > MAX_TEXT_CHARS:
            raise ValueError(f"prompt exceeds MAX_TEXT_CHARS={MAX_TEXT_CHARS}: {len(prompt)}")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an int")
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
            raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
        return list(images)

    def generate(
        self,
        images: Image.Image | list[Image.Image],
        prompt: str,
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = False,
    ) -> dict[str, Any]:
        """Answer ``prompt`` about one image; ``text`` is the decoded assistant turn (no special tokens)."""
        batch = self._validate(images, prompt, max_new_tokens)
        raw = self._runner(batch[0].convert("RGB"), prompt, max_new_tokens, bool(do_sample))
        if not isinstance(raw, dict) or "text" not in raw:
            raise RuntimeError("runner must return a dict with 'text'")
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "text": str(raw["text"]).strip(),
            "prompt": prompt,
            "image_size": list(batch[0].size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= max_new_tokens,
            "generation": {
                "max_new_tokens": max_new_tokens,
                "do_sample": bool(do_sample),
                "decoding": "sampling" if do_sample else DECODING,
            },
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
