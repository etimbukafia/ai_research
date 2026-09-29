# Evals 101 for enterprise agents

This report is generated from the deterministic replay. It makes no provider request.

## Dataset

- Dataset: `enterprise_evals_101@1.0.0`
- Cases: `24`
- Pairs: `12`
- Replay traces: `72`
- Retrieval condition: `semantic_catalog`
- Source data: synthetic Aster Cloud records

## Layered results

| Candidate | Answer-only pass | Enterprise pass | False passes | Evidence recall | Unsafe action rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_false_pass` | 1.000000 | 0.458333 | 13 | 0.500000 | 0.333333 |
| `safe_candidate` | 1.000000 | 0.916667 | 2 | 1.000000 | 0.000000 |
| `fast_answer_candidate` | 0.666667 | 0.583333 | 2 | 1.000000 | 0.666667 |

Answer-only false passes keep the final fields correct while an enterprise check fails.

## Featured billing pair

| Candidate | Clean | Distractor | Sensitivity |
| --- | ---: | ---: | ---: |
| `baseline` | 1.000000 | 1.000000 | 0.000000 |
| `safe_candidate` | 1.000000 | 1.000000 | 0.000000 |

## Candidate comparison

- Primary verdict: `improved`
- Primary target improved: `True`
- Primary holdout checked: `True`
- Primary hard regressions: `0`
- Negative control verdict: `rejected`
- Negative control hard regressions: `21`
- Safe candidate promotion: `approved`
- Fast-answer promotion: `rejected`

## Evaluator fixtures

The fixtures run before the agent result is reported. They show which suite can see each failure.

## Live validation

Live Gemini validation was not invoked. The saved plan uses eight cases, three repeats, one worker, a five-second interval, and a 15 RPM project limit.

## Limits

The records and traces are synthetic. The case set is small. The replay measures this fixture. Passing it does not prove production safety.
