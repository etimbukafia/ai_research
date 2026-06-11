I want to do an experiment on self improving agents. It's not for production so i don't want it to be broad

# Self-Improving Biomedical Evidence Triage Agent

Given a rare disease and candidate drug pair, the agent determines whether available evidence supports investigating
the drug for that disease.

The agent should:

1. Retrieve evidence from Open Targets and PubMed.
2. Extract disease, drug, target, and mechanism relationships.
3. Classify the candidate as:
    - supported
    - weakly_supported
    - unsupported
    - insufficient_evidence

4. Produce citations and a structured explanation.
5. Receive evaluator feedback.
6. Store validated lessons in Mem0.
7. Use those lessons on subsequent candidate assessments.

This tests whether the agent becomes better at evaluating evidence, rather than attempting to discover scientifically
novel drugs.

## Why This Project For The Experiment

The task has repeatable inputs and measurable outputs.

We can evaluate:

- Classification accuracy or F1
- Citation correctness
- Evidence extraction accuracy
- Correct abstention rate
- Unsupported-claim rate
- Performance improvement over time
- Token and latency costs
- Whether incorrect memories propagate

It also naturally produces reusable lessons:

- “Animal-model evidence alone should not be classified as strong clinical evidence.”
- “A drug targeting a disease-associated protein is indirect evidence, not proof of efficacy.”
- “Retracted or contradictory studies should reduce confidence.”
- “Do not treat pathway similarity as direct drug-disease evidence.”

These procedural lessons are appropriate for Mem0. Storing speculative biomedical claims as learned facts would be
considerably riskier.

## Experiment Design

Use a fixed sequential dataset of approximately 100 disease-drug pairs.

 Phase                  Pairs    Purpose
━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 Initial evaluation        20    Establish baseline
─────────────────────  ───────  ────────────────────────────────────────
 Learning stream           50    Provide feedback and build memory
─────────────────────  ───────  ────────────────────────────────────────
 Held-out evaluation       20    Measure generalization
─────────────────────  ───────  ────────────────────────────────────────
 Distribution shift        10    Test resistance to misleading memories

Compare three conditions:

1. No memory
   The agent receives no previous experiences.

2. Raw Mem0 memory
   Successful and failed trajectories are stored directly.

3. Validated lesson memory
   Only evaluator-approved lessons are stored, with evidence type, confidence, source, and failure category.

The important comparison is raw memory versus validated lessons. Raw memory may eventually reduce performance through
error propagation.

## Mem0’s Role

Mem0 is suitable as the experiment’s experiential-memory layer. Its current open-source system supports fact
extraction, custom instructions, metadata filtering, and hybrid retrieval using semantic, keyword, and entity signals.

However, Mem0 does not make the agent self-improving by itself. You still need to implement:

- An evaluator
- Failure attribution
- Lesson generation
- Memory acceptance or rejection
- Confidence and provenance metadata
- Regression testing

The latest Mem0 open-source algorithm uses ADD-only extraction. New facts accumulate rather than automatically
replacing old facts. This makes provenance, validation status, and supersession metadata particularly important for
the experiment.

Suggested memory structure:

{
  "lesson": "Do not classify target association as direct efficacy evidence.",
  "failure_type": "evidence_strength_overestimation",
  "validated": true,
  "applicable_evidence_types": ["target_association"],
  "source_task": "disease-drug-pair-042",
  "confidence": 0.9
}

## Keep the Prototype Narrow

Use only:

- Open Targets for disease, target, association, and known-drug data
- PubMed for literature evidence

Also avoid calling the output a drug-repurposing recommendation. Frame it as an evidence-triage research experiment,
not medical or therapeutic guidance.

## Final Project Definition

> Build an agent that evaluates evidence for rare-disease and candidate-drug pairs, learns validated evidence-
> assessment lessons from evaluator feedback using Mem0, and measures whether those memories improve performance on
> unseen pairs without increasing hallucinations or error propagation.

This is narrow enough to complete, measurable enough to study, and still directly relevant to a future drug-
repurposing agent.