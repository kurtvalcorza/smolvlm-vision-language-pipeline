"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import pytest
from PIL import Image

from smolvlm_vision_language_pipeline import (
    DECODING,
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_IMAGES,
    MAX_NEW_TOKENS,
    MAX_TEXT_CHARS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)

PROMPT = "Describe this image in one sentence."


def _image(width: int = 384, height: int = 384) -> Image.Image:
    return Image.new("RGB", (width, height), (255, 255, 255))


def _result(prompt: str = PROMPT, new_tokens: int = 42, max_new_tokens: int = 96) -> dict:
    return {
        "text": "The image depicts a red square and a blue circle.",
        "prompt": prompt,
        "image_size": [384, 384],
        "new_tokens": new_tokens,
        "truncated": new_tokens >= max_new_tokens,
        "generation": {"max_new_tokens": max_new_tokens, "do_sample": False, "decoding": DECODING},
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(
        _image(), PROMPT, max_new_tokens=96, names=["synthetic_square_circle_384"]
    )
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["images"] == [1, MAX_IMAGES]
    assert manifest["schema"]["image_side_px"] == [1, MAX_IMAGE_SIDE]
    assert manifest["schema"]["prompt_chars"] == [1, MAX_TEXT_CHARS]
    assert manifest["schema"]["max_new_tokens"] == [1, MAX_NEW_TOKENS]
    assert manifest["inputs"] == [
        {"id": "synthetic_square_circle_384", "mode": "RGB", "size": [384, 384]}
    ]
    assert manifest["prompt"] == PROMPT
    assert manifest["prompt_chars"] == len(PROMPT)
    assert manifest["generation"] == {"max_new_tokens": 96, "do_sample": False, "decoding": DECODING}
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_sampling_decoding() -> None:
    manifest = validate_inputs([_image()], PROMPT, do_sample=True)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["generation"]["do_sample"] is True
    assert manifest["generation"]["decoding"] == "sampling"
    assert manifest["generation"]["max_new_tokens"] == DEFAULT_MAX_NEW_TOKENS


def test_validate_inputs_rejects_like_generate() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 8), PROMPT)
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image", PROMPT)
    with pytest.raises(ValueError, match="MAX_IMAGES"):
        validate_inputs([_image()] * (MAX_IMAGES + 1), PROMPT)
    with pytest.raises(ValueError, match="prompt must not be empty"):
        validate_inputs(_image(), "   ")
    with pytest.raises(ValueError, match="MAX_TEXT_CHARS"):
        validate_inputs(_image(), "x" * (MAX_TEXT_CHARS + 1))
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), PROMPT, max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError, match="max_new_tokens must be an int"):
        validate_inputs(_image(), PROMPT, max_new_tokens=True)
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs(_image(), PROMPT, names=["a", "b"])


def test_evaluation_report_is_always_not_measurable() -> None:
    report = evaluation_report(_result())
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert report["n_answers"] == 1
    assert "no intrinsic correctness signal" in report["reason"]
    assert "reference captions" in report["needs"]
    assert "no score, no probability" in report["score_semantics"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_accepts_a_sequence_and_records_truncation() -> None:
    results = [_result(new_tokens=42), _result("How many shapes?", new_tokens=96)]
    report = evaluation_report(results, sample_kind="BYOD upload")
    assert report["n_answers"] == 2
    assert report["sample_kind"] == "BYOD upload"
    assert [answer["truncated"] for answer in report["answers"]] == [False, True]
    assert [answer["prompt"] for answer in report["answers"]] == [PROMPT, "How many shapes?"]
    assert report["verdict"] == "not-measurable"


def test_evaluation_report_echoes_references_without_creating_a_metric() -> None:
    report = evaluation_report(_result(), ["a red square and a blue circle"])
    assert report["references_supplied"] == ["a red square and a blue circle"]
    assert report["metrics"] == []
    assert report["verdict"] == "not-measurable"
