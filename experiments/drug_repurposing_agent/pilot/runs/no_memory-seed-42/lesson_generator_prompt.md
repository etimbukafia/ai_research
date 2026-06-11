# Procedural Lesson Generator

Prompt version: `lesson-generator-v1.0.0`

Generate exactly one JSON object conforming to the `LessonCandidate` schema from the
supplied learning-stream evaluator feedback. Return JSON only and use the exact field
names and types in `lesson_candidate_json_schema`.

The lesson must be a general evidence-assessment procedure that could improve future
tasks. It may explain how to weigh evidence types, verify citations, extract
relationships, resolve contradictions, or abstain. It must be directly supported by
the supplied evaluator feedback and use one of its failure types.

Never include:

- The source disease name, drug name, identifiers, or a pair-specific answer.
- A claim that a drug, target, mechanism, or therapy is effective, ineffective,
  beneficial, harmful, safe, or unsafe for a disease.
- New biomedical facts or speculation not stated in evaluator feedback.
- Medical advice, dosing, prescribing, or treatment recommendations.
- A duplicate or close paraphrase of an existing validated lesson.

Generated output is an unapproved candidate only. It cannot enter validated memory
until deterministic validation passes, all regression cases pass, and an explicit
approval record is created.
