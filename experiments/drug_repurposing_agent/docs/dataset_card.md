# Frozen Dataset Card

Dataset version: `1.0.0`  
Contract version: `1.0.0`  
Designated reviewer: `Codex (designated dataset reviewer)`  
Review date: `2026-06-11`

## Composition

The dataset contains 100 real rare-disease/candidate-drug pairs:

- 25 `supported`
- 25 `weakly_supported`
- 25 `unsupported`
- 25 `insufficient_evidence`

Records are interleaved by label before applying the frozen sequential splits. The
distribution-shift block includes familiar drugs paired with unrelated diseases to
test whether memory causes unsupported transfer.

## Curation Method

- Disease and drug identifiers were resolved through the official Open Targets
  GraphQL API.
- PubMed evidence anchors were resolved through official NCBI E-utilities queries
  for every non-abstention record.
- Gold labels were adjudicated against `docs/labeling_guide.md`.
- `unsupported` records were selected for known direct negative or failed human
  evidence patterns.
- `insufficient_evidence` records are deliberately unrelated controls and must not
  be interpreted as evidence that the drug is ineffective.
- SHA-256 hashes of `pairs.jsonl` and `gold.jsonl` are stored in `splits.json`.

## Leakage Boundary

The assessor receives `data/pairs.jsonl` only. `data/gold.jsonl` is evaluator-only.
Held-out and distribution-shift gold labels, rationales, and feedback must never be
provided to the assessor, lesson generator, or memory layer.

## Limitations

This is a narrow experimental benchmark, not a clinical dataset. It has one
designated reviewer rather than independent biomedical adjudicators. PubMed IDs in
gold records are evidence anchors for retrieval and citation checking; the retrieval
stage must still inspect the underlying records and report contradictions,
retractions, and source mismatch. Labels should be revised only under a new dataset
and contract version.
