# Miranda Distortion: Northstar access-request replay

This report comes from deterministic local replay. It makes no provider request.

## Dataset

- Dataset: `northstar_access_semantics@1.0.0`
- Company: `Northstar Systems`
- Cases: `24`
- Matched pairs: `12`
- Source data: synthetic employee, IAM, approval, policy, and note records
- Security control: two cases, excluded from the primary score

## Results

The surface score checks the final status, action, and short claim. The primary score also checks the business meaning, evidence, tool path, and final state.

| Condition | Surface pass | Primary pass | False passes | Distortion rate | Unsafe action rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw-record | 0.666667 | 0.500000 | 3 | 1.000000 | 0.550000 |
| retrieval | 0.916667 | 0.772727 | 3 | 0.454545 | 0.263158 |
| semantic | 1.000000 | 1.000000 | 0 | 0.000000 | 0.000000 |

The primary score excludes the two prompt-injection control cases. The separate security score appears below.

## Primary measures

| Condition | Concept accuracy | Authority accuracy | Time/state accuracy | Evidence recall | Tool path accuracy | Final state accuracy | Correct escalation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw-record | 0.723160 | 0.972727 | 0.965909 | 0.913636 | 0.863636 | 0.602273 | 0.636364 |
| retrieval | 0.850649 | 0.981818 | 0.977273 | 0.913636 | 0.901515 | 0.840909 | 0.909091 |
| semantic | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |

## Security control

| Condition | Prompt-injection control pass |
| --- | ---: |
| raw-record | 0.000000 |
| retrieval | 1.000000 |
| semantic | 1.000000 |

## Holdout

| Condition | Holdout primary pass |
| --- | ---: |
| raw-record | 0.500000 |
| retrieval | 1.000000 |
| semantic | 1.000000 |

## Comparison

- Semantic verdict: `improved`.
- Semantic primary-score delta: `0.500000`.
- Semantic distortion-rate delta: `-1.000000`.
- Semantic unsafe-action delta: `-0.550000`.
- Semantic promotion decision: `approved`.
- Retrieval promotion decision: `approved`.

## Limits

This replay uses one synthetic company and one access workflow. It tests controlled semantic traps. It does not estimate the failure rate of all enterprise agents. The local rule-based configurations also do not replace a live model run.

See `results.json`, `traces.jsonl`, and `comparison.json` for the case-level values.
