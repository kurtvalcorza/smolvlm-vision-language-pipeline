import hashlib
import json
import re
from pathlib import Path

import pytest
from PIL import Image

from smolvlm_vision_language_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WEIGHTS_DIR,
    MAX_IMAGE_SIDE,
    MAX_IMAGES,
    MAX_NEW_TOKENS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    SmolVLMPipeline,
    build_messages,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")


def _fake_runner(image, prompt, max_new_tokens, do_sample):
    assert image.mode == "RGB"
    return {"text": f"  Assistant: echo {prompt} ", "new_tokens": min(7, max_new_tokens)}


def _pipeline() -> SmolVLMPipeline:
    return SmolVLMPipeline(_fake_runner, "cpu", "float32", "injected")


def _write_snapshot(root: Path, payload: bytes = b"weights") -> Path:
    (root / "model.safetensors").write_bytes(payload)
    manifest = {
        "modelKey": MODEL_KEY,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "model.safetensors",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    path = root / "dimer-base-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_identity_constants_are_40_hex_and_named():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "HuggingFaceTB/SmolVLM-500M-Instruct"
    assert DEFAULT_WEIGHTS_DIR.name == MODEL_KEY
    assert DEFAULT_WEIGHTS_DIR.parent.name == "weights"
    assert MAX_IMAGES == 1
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS


def test_identity_matches_local_manifest_when_present():
    manifest_path = DEFAULT_WEIGHTS_DIR / "dimer-base-manifest.json"
    if not manifest_path.is_file():
        pytest.skip("local snapshot manifest not staged")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["modelId"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["modelKey"] == MODEL_KEY


def test_verify_snapshot_accepts_matching_manifest(tmp_path: Path):
    _write_snapshot(tmp_path)
    result = verify_snapshot(tmp_path)
    assert result["revision"] == MODEL_REVISION
    assert result["path"] == str(tmp_path)


def test_verify_snapshot_rejects_tampered_digest(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = manifest["files"][0]["sha256"]
    manifest["files"][0]["sha256"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_tampered_bytes_and_missing_file(tmp_path: Path):
    _write_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"weightz")
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"short")
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_identity(tmp_path: Path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["revision"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path / "missing")


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    assert len(verify_snapshot(tmp_path)["files"]) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        SmolVLMPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_loading(tmp_path):
    manifest_path = _write_snapshot(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        SmolVLMPipeline.from_pretrained(device="cpu", weights_dir=tmp_path)


def test_build_messages_shape():
    messages = build_messages("Describe the image.")
    assert messages == [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}
    ]


def test_generate_rejects_bad_inputs():
    pipe = _pipeline()
    image = Image.new("RGB", (32, 32))
    with pytest.raises(TypeError):
        pipe.generate("not-an-image", "hi")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pipe.generate([42], "hi")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="MAX_IMAGES"):
        pipe.generate([], "hi")
    with pytest.raises(ValueError, match="MAX_IMAGES"):
        pipe.generate([image] * (MAX_IMAGES + 1), "hi")
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.generate(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 1)), "hi")
    with pytest.raises(TypeError):
        pipe.generate(image, 5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="empty"):
        pipe.generate(image, "   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        pipe.generate(image, "x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.generate(image, "hi", max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.generate(image, "hi", max_new_tokens=0)
    with pytest.raises(TypeError):
        pipe.generate(image, "hi", max_new_tokens=2.5)  # type: ignore[arg-type]


def test_generate_output_fields():
    pipe = _pipeline()
    result = pipe.generate(Image.new("L", (16, 24)), "Describe the image.")
    assert result["model_id"] == MODEL_ID
    assert result["model_revision"] == MODEL_REVISION
    assert result["text"] == "Assistant: echo Describe the image."
    assert result["prompt"] == "Describe the image."
    assert result["image_size"] == [16, 24]
    assert result["new_tokens"] == 7
    assert result["truncated"] is False
    expected = {"max_new_tokens": DEFAULT_MAX_NEW_TOKENS, "do_sample": False, "decoding": "greedy"}
    assert result["generation"] == expected
    assert result["device"] == "cpu" and result["dtype"] == "float32" and result["source"] == "injected"


def test_generate_reports_truncation_and_sampling_flag():
    pipe = _pipeline()
    result = pipe.generate([Image.new("RGB", (8, 8))], "hi", max_new_tokens=7, do_sample=True)
    assert result["truncated"] is True
    assert result["generation"]["decoding"] == "sampling"


def test_generate_rejects_malformed_runner_output():
    pipe = SmolVLMPipeline(lambda *args: {"tokens": 1}, "cpu")
    with pytest.raises(RuntimeError, match="runner must return"):
        pipe.generate(Image.new("RGB", (8, 8)), "hi")
