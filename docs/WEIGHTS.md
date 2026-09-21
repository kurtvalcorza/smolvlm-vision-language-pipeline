# Weight provenance and DIMER hosting

- Upstream: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Immutable revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`
- Weight format: SafeTensors (`model.safetensors`, 1015025832 bytes, bfloat16 as shipped)
- Upstream weight license: Apache-2.0
- Local snapshot: `weights/smolvlm-500m-instruct/` with `dimer-base-manifest.json` (13 entries, per-file bytes + SHA-256, `totalBytes` 1019893574); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/smolvlm_vision_language_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch; `stage_missing_files()` fetches only manifest-listed files at the pinned revision.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `transformers==4.57.6` native `Idefics3ForConditionalGeneration` via `AutoModelForImageTextToText` and `Idefics3Processor` via `AutoProcessor`, both with `trust_remote_code=False`; the snapshot carries no custom code. Local load uses `local_files_only=True`; Hub download is opt-in and pinned to the revision above.

## Adaptation artifacts and the sample corpus

- Dtype: the pipeline loads the bfloat16 checkpoint in float32 on every device (the adapter is trained in float32 and overlays without a cast).
- Adapter artifacts: `save_artifact` writes the trained tensors (the last eight SmolLM2 decoder layers and the final norm, 73 tensors, about 315 MB in float32) as `adapter.safetensors` beside a `manifest.json` (`org.valcorza.smolvlm-500m-instruct.adapter.v1`) naming the base model id and revision, the base `model.safetensors` SHA-256, the tensor names, the file size and SHA-256, the instruction the adapter was trained on, the training configuration and the epoch history. `from_artifact` re-verifies the base snapshot and checks the manifest, the digest and the exact tensor set before deserialising; an adapter is derived from the lines it was trained on and is not redistributed by this repository.
- Adaptation sample: `Teklia/Belfort-line` (MIT), parquet-conversion revision `c4a74bbd39f2df314752e7e6026649a39d365cbb`, `default/test/0000.parquet` row groups 0–7 (800 lines, about 44 MB) read over HTTPS range requests, each row group pinned by SHA-256 and byte total in `samples.py`; cached git-ignored under `weights/belfort/`, never vendored.
