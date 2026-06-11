# Executable Checklist: Self-Improving Biomedical Evidence Triage Agent

Use this checklist from the repository root. Complete items in order. Do not expand
the prototype beyond Open Targets, PubMed, the four classification labels, and the
three memory conditions.

## Definition Of Done

- [ ] A fixed, versioned dataset contains 100 rare-disease/candidate-drug pairs.
- [ ] The agent produces a validated structured assessment with citations.
- [ ] The evaluator scores classification, extraction, citations, abstention, and
      unsupported claims.
- [ ] The same sequential experiment runs under `no_memory`, `raw_memory`, and
      `validated_lessons`.
- [ ] Only evaluator-approved procedural lessons enter validated memory.
- [ ] Results show baseline, held-out, distribution-shift, cost, and latency metrics.
- [ ] A final report states whether validated lessons improved held-out performance
      without increasing unsupported claims or error propagation.

## 0. Lock The Experiment Contract

- [x] Record the exact model name, temperature, prompt version, random seed, and run
      date in `config/experiment.yaml`.
- [x] Keep the labels fixed:
      `supported`, `weakly_supported`, `unsupported`, `insufficient_evidence`.
- [x] Write a one-paragraph operational definition for each label in
      `docs/labeling_guide.md`.
- [x] Define evidence-strength rules in the labeling guide, including:
      - Human clinical evidence can support the strongest label.
      - Animal or in-vitro evidence alone cannot count as strong clinical evidence.
      - Target association is indirect evidence, not proof of efficacy.
      - Contradictory or retracted evidence lowers confidence.
      - Missing reliable evidence requires abstention.
- [x] State that outputs are research evidence triage, not medical advice or a
      treatment recommendation.
- [x] Freeze the primary success criterion before running the experiment:
      validated lessons improve held-out macro-F1 over no-memory without increasing
      unsupported-claim rate.

**Gate:** A second reader can label five sample pairs from the guide without needing
an unwritten rule. **Status:** pending independent-reader review; the guide includes
seven worked adjudication examples for the review.

## 1. Scaffold The Narrow Prototype

- [x] Create this structure:

```text
experiments/drug_repurposing_agent/
  config/experiment.yaml
  data/pairs.jsonl
  data/gold.jsonl
  data/splits.json
  docs/labeling_guide.md
  prompts/assessor.md
  prompts/lesson_generator.md
  src/__init__.py
  src/models.py
  src/open_targets.py
  src/pubmed.py
  src/agent.py
  src/evaluator.py
  src/memory.py
  src/runner.py
  tests/
  runs/
```

- [x] Add only required dependencies to `requirements.txt`; pin the Mem0 and Pydantic-AI package
      version used by the experiment.
- [x] Confirm imports and syntax:

```powershell
python -m compileall experiments\drug_repurposing_agent\src
```

**Gate:** The compile command exits successfully. **Status:** passed.

## 2. Define Structured Schemas First

- [x] In `src/models.py`, define and validate:
      `DiseaseDrugPair`, `EvidenceItem`, `Relationship`, `Assessment`,
      `EvaluatorFeedback`, `Lesson`, and `RunRecord`.
- [x] Require every `EvidenceItem` to contain:
      source, source identifier, URL, title, evidence type, extracted claim, and
      retrieval timestamp.
- [x] Require every assessment claim to reference one or more evidence identifiers.
- [x] Require `Assessment` to contain:
      pair ID, label, confidence, evidence items, relationships, explanation,
      limitations, and citations.
- [x] Require `Lesson` to contain:
      lesson text, failure type, validation status, applicable evidence types,
      source task, confidence, provenance, and supersession metadata.
- [x] Reject lessons that contain a disease-specific efficacy claim or treatment
      recommendation.
- [x] Add schema tests:

```powershell
python -m pytest experiments\drug_repurposing_agent\tests\test_models.py -q
```

**Gate:** Invalid labels, uncited claims, and unvalidated lessons are rejected.
**Status:** passed; 17 schema tests, including resumable session and append-only
event invariants.

## 3. Build And Freeze The Dataset

- [x] Curate exactly 100 disease-drug pairs with stable IDs.
- [x] Include examples across all four labels and multiple evidence patterns.
- [x] Have a human reviewer create `data/gold.jsonl` using the labeling guide.
- [x] Record the gold label, expected relationships, acceptable source IDs,
      rationale, and known contradictions for each pair.
- [x] Freeze sequential splits in `data/splits.json`:
      - IDs 001-020: initial evaluation
      - IDs 021-070: learning stream
      - IDs 071-090: held-out evaluation
      - IDs 091-100: distribution shift
- [x] Keep held-out and distribution-shift gold records unavailable to the agent and
      lesson generator.
- [x] Validate counts, unique IDs, label coverage, and split isolation:

```powershell
python -m experiments.drug_repurposing_agent.src.runner validate-data
```

**Gate:** Validation reports 100 unique pairs, no split overlap, and no missing gold
fields. **Status:** passed; frozen hashes and leakage policy verified.

## 4. Implement Evidence Retrieval

- [x] Implement an Open Targets client that retrieves only disease, target,
      association, and known-drug evidence.
- [x] Implement a PubMed client using documented API endpoints and stable PubMed IDs.
- [x] Cache raw responses by request hash so all three conditions use identical
      evidence.
- [x] Store retrieval timestamps and query parameters.
- [x] Add timeouts, bounded retries, and explicit retrieval-error records.
- [x] Never convert retrieval failure into `unsupported`; use
      `insufficient_evidence` when evidence cannot be assessed.
- [x] Test parsing with committed small fixtures, not live APIs:

```powershell
python -m pytest experiments\drug_repurposing_agent\tests\test_open_targets.py experiments\drug_repurposing_agent\tests\test_pubmed.py -q
```

- [x] Run one live smoke test:

```powershell
python -m experiments.drug_repurposing_agent.src.runner retrieve --pair-id disease-drug-pair-001
```

**Gate:** The smoke-test evidence can be traced back to valid Open Targets or PubMed
source identifiers. **Status:** passed for `disease-drug-pair-001`; replay produced
3 cache hits, 0 cache misses, and 0 retrieval errors.

## 5. Implement The Baseline Assessor

- [x] Write `prompts/assessor.md` using the frozen label definitions.
- [x] Make the assessor return the `Assessment` schema only.
- [x] Supply retrieved evidence and, depending on condition, retrieved memories.
- [x] Instruct the assessor to cite evidence IDs for each material claim.
- [x] Instruct the assessor to report contradictions and limitations.
- [x] Instruct the assessor to abstain when available evidence is insufficient.
- [x] Add deterministic post-validation that rejects malformed or uncited output.
- [x] Test the assessor with mocked model responses:

```powershell
python -m pytest experiments\drug_repurposing_agent\tests\test_agent.py -q
```

**Gate:** The assessor cannot emit an out-of-schema label or an uncited material
claim. **Status:** passed; mocked-model tests also reject evidence mutation,
pair mismatch, memory leakage into `no_memory`, and failed zero-evidence abstention.

## 6. Implement The Evaluator

- [x] Make gold labels and gold relationships the source of truth.
- [x] Score each assessment for:
      classification correctness, relationship extraction, citation correctness,
      correct abstention, unsupported claims, latency, and token cost.
- [x] Categorize failures, including:
      `evidence_strength_overestimation`, `citation_mismatch`,
      `relationship_extraction_error`, `failed_abstention`,
      `contradiction_ignored`, and `retrieval_failure`.
- [x] Produce evaluator feedback that identifies the error but does not expose
      held-out answers.
- [x] Add evaluator tests with known correct and incorrect assessments:

```powershell
python -m pytest experiments\drug_repurposing_agent\tests\test_evaluator.py -q
```

**Gate:** Evaluator tests detect deliberately inserted citation mismatches,
unsupported claims, and failed abstentions. **Status:** passed; deterministic tests
also cover relationship errors, ignored contradictions, retrieval failures, runtime
metrics, and held-out feedback redaction.

## 7. Implement The Three Memory Conditions

- [x] `no_memory`: do not write or retrieve any previous experience.
- [x] `raw_memory`: store and retrieve completed trajectories and evaluator feedback.
- [x] `validated_lessons`: generate procedural lessons from learning-stream feedback,
      then store only evaluator-approved lessons.
- [x] Use separate Mem0 namespaces or stores for each condition and run.
- [x] Filter validated lesson retrieval by `validated=true` and applicable evidence
      type.
- [x] Preserve source task, confidence, provenance, and supersession metadata.
- [x] Prevent the memory layer from receiving held-out or distribution-shift gold
      answers.
- [x] Add memory isolation and validation tests:

```powershell
python -m pytest experiments\drug_repurposing_agent\tests\test_memory.py -q
```

**Gate:** A rejected lesson is never retrieved, and one condition cannot retrieve
another condition's memories. **Status:** passed; tests also verify `no_memory`
performs no backend operations, non-learning feedback is rejected, validated lessons
are evidence-type filtered, and Mem0 calls use condition/run namespaces. Lesson
generation and approval orchestration remains in section 8.

## 8. Add Lesson Generation And Approval

- [x] Write `prompts/lesson_generator.md` to produce general procedural lessons only.
- [x] Generate candidate lessons only from learning-stream evaluator feedback.
- [x] Reject lessons that are speculative, pair-specific, unsupported by feedback,
      duplicates, or treatment recommendations.
- [x] Require an explicit approval record before writing to validated memory.
- [x] Log rejected lessons and rejection reasons.
- [x] Create regression cases for each accepted lesson and run them before approval.

**Gate:** Every validated lesson has evaluator evidence, an approval record, and a
passing regression case. **Status:** passed; approval requires all five deterministic
regressions, and validated-memory writes reject missing or mismatched approval
records.

## 9. Implement The Reproducible Runner

- [x] Make `src/runner.py` support:

```powershell
python -m experiments.drug_repurposing_agent.src.runner validate-data
python -m experiments.drug_repurposing_agent.src.runner run --condition no_memory --seed 42
python -m experiments.drug_repurposing_agent.src.runner run --condition raw_memory --seed 42
python -m experiments.drug_repurposing_agent.src.runner run --condition validated_lessons --seed 42
python -m experiments.drug_repurposing_agent.src.runner report --run-dir experiments\drug_repurposing_agent\runs
```

- [x] Process pairs in frozen sequential order.
- [x] Do not update memory during initial, held-out, or distribution-shift phases.
- [x] Update memory only after evaluator feedback in the learning stream.
- [x] Persist config, prompts, evidence IDs, assessments, feedback, memory events,
      metrics, token usage, latency, errors, and random seed for every run.
- [x] Resume safely without processing a pair twice.

**Gate:** Re-running a completed run does not duplicate records or memory writes.
**Status:** passed; deterministic runner tests cover partial resume, completed-run
replay, sequential ordering, phase-restricted raw and validated memory writes,
approval artifacts, session events, frozen inputs, and reporting.

## 10. Run Pilot Before The Full Experiment

- [x] Run a 12-pair development pilot outside the frozen 100-pair dataset.
- [x] Confirm all four labels appear in the pilot.
- [x] Manually inspect citations and unsupported claims.
- [x] Confirm raw and validated memory stores receive different content.
- [x] Fix only implementation defects or ambiguous labeling rules.
- [x] Freeze code, prompts, config, labeling guide, and dataset after the pilot.
- [ ] Record a git commit hash in `config/experiment.yaml`.

**Gate:** No schema, retrieval, scoring, memory-isolation, or resume defects remain.
**Status:** passed; 36/36 labels, relationships, and citations scored correctly,
unsupported-claim count was zero, and all 64 tests pass. Freeze commit pending.

## 11. Run The Full Experiment

- [ ] Clear condition-specific run state without deleting the frozen evidence cache.
- [ ] Run all three conditions with the same seed and cached evidence.
- [ ] Repeat with at least two additional seeds if model sampling is non-deterministic.
- [ ] Do not tune prompts, labels, or memory policy after viewing held-out results.
- [ ] Record any failed run and rerun reason.

```powershell
python -m experiments.drug_repurposing_agent.src.runner run --condition no_memory --seed 42
python -m experiments.drug_repurposing_agent.src.runner run --condition raw_memory --seed 42
python -m experiments.drug_repurposing_agent.src.runner run --condition validated_lessons --seed 42
python -m experiments.drug_repurposing_agent.src.runner report --run-dir experiments\drug_repurposing_agent\runs
```

**Gate:** Each condition has results for all 100 pairs and a complete audit trail.

## 12. Analyze Results

- [ ] Compare conditions by phase using:
      macro-F1, per-label precision/recall/F1, citation correctness, relationship
      extraction accuracy, correct abstention rate, unsupported-claim rate, latency,
      and token cost.
- [ ] Measure change from initial evaluation to held-out evaluation.
- [ ] Measure distribution-shift degradation.
- [ ] Count incorrect memories, retrieval frequency, and downstream errors caused by
      those memories.
- [ ] Compare raw memory directly against validated lessons.
- [ ] Report confidence intervals or variation across seeds.
- [ ] Inspect a sample of wins, regressions, abstentions, and propagated errors.

**Gate:** The analysis can distinguish genuine improvement from increased guessing
or leakage.

## 13. Write The Final Report

- [ ] Create `REPORT.md` containing:
      experiment contract, dataset and splits, implementation, memory conditions,
      metrics, results, failure analysis, limitations, and reproducibility commands.
- [ ] Answer the primary question directly:
      Did validated lesson memory improve unseen-pair evidence triage without
      increasing hallucinations or error propagation?
- [ ] Clearly separate evidence-triage performance from claims of biomedical or
      therapeutic efficacy.
- [ ] Document all protocol deviations.
- [ ] Include links or paths to run artifacts and the frozen config.

**Gate:** Another developer can reproduce the experiment from the report and audit
every reported metric back to pair-level records.

## Final Verification

```powershell
python -m compileall experiments\drug_repurposing_agent\src
python -m pytest experiments\drug_repurposing_agent\tests -q
python -m experiments.drug_repurposing_agent.src.runner validate-data
python -m experiments.drug_repurposing_agent.src.runner report --run-dir experiments\drug_repurposing_agent\runs
```

- [ ] All commands exit successfully.
- [ ] All Definition Of Done items are checked.
