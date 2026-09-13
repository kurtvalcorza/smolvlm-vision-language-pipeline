# Weight provenance and DIMER hosting

- Upstream: `HuggingFaceTB/SmolVLM-500M-Instruct`
- Immutable revision: `a7da5b986cb59b408707209984f360a5f4ad7e47`
- Weight format: SafeTensors (`model.safetensors`, 1015025832 bytes, bfloat16 as shipped)
- Upstream weight license: Apache-2.0
- Local snapshot: `weights/smolvlm-500m-instruct/` with `dimer-base-manifest.json` (13 entries, per-file bytes + SHA-256, `totalBytes` 1019893574); the Git repository does not vendor the checkpoint.
- Load-time check: `verify_snapshot()` in `src/smolvlm_vision_language_pipeline/pipeline.py` re-hashes every manifest entry and refuses on any mismatch; `stage_missing_files()` fetches only manifest-listed files at the pinned revision.
- DIMER hosting: Apache-2.0 permits use, modification, distribution and commercial use subject to the license and notice requirements; DIMER may mirror the pinned checkpoint in its model store under the upstream license.
- Loader trust boundary: `transformers==4.57.6` native `Idefics3ForConditionalGeneration` via `AutoModelForImageTextToText` and `Idefics3Processor` via `AutoProcessor`, both with `trust_remote_code=False`; the snapshot carries no custom code. Local load uses `local_files_only=True`; Hub download is opt-in and pinned to the revision above.
