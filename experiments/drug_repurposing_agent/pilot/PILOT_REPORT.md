# Development Pilot Report

Date: `2026-06-11`

## Scope

The development pilot used 12 synthetic disease-drug pairs under
`pilot/data/`, outside the frozen 100-pair dataset. Each of the four labels appeared
three times. The same sequential pilot ran under `no_memory`, `raw_memory`, and
`validated_lessons` with seed `42` and real `gemini-3.1-flash-lite` calls.

## Results

- 36 pair-level records completed across three conditions.
- Classification correctness: 36/36.
- Relationship extraction score of 1.0: 36/36.
- Citation correctness score of 1.0: 36/36.
- Unsupported-claim rows: 0/36.
- All citation evidence IDs were traceable to supplied pilot evidence.
- Raw memory stored 12 completed learning-stream trajectories.
- Validated memory stored 2 evaluator-approved procedural lessons.

Detailed pair-level review is in `PILOT_REVIEW.json`; condition artifacts are under
`pilot/runs/`.

## Defects Found And Fixed

1. Structured prompts did not include exact Pydantic JSON schemas.
2. The assessor lacked bounded structured-output repair.
3. Gemini requests lacked free-tier rate pacing and retry.
4. Windows atomic checkpoint replacement needed bounded retry.
5. Resume could encounter an atomic record newer than its session checkpoint.
6. Unsupported-claim scoring incorrectly included relationship predicate mismatch.
7. The assessor prompt omitted canonical relationship predicates.
8. Runtime environment variables were not loaded through an experiment config.

All fixes were completed before freeze. The frozen 100-pair dataset hashes and
leakage policy still validate.

## Freeze

Contract, prompts, evaluator policy, code, labeling guide, and frozen dataset are
frozen as contract version `1.1.0`. Subsequent changes require a protocol deviation
and new contract version. Freeze commit:
`2b673988280043124900d012bf2b2f03b4ece983`.
