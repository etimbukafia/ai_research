# Labeling Guide: Biomedical Evidence Triage

Version: `1.0.0`  
Status: locked on `2026-06-11`

## Purpose And Scope

Given one rare-disease and candidate-drug pair, assign exactly one label describing
whether evidence available from the frozen Open Targets and PubMed snapshot supports
investigating that drug for that disease. The output is research evidence triage. It
is not medical advice, a treatment recommendation, or a claim that the drug is safe
or effective.

Judge only the supplied evidence snapshot. Do not use background knowledge, general
web search, guidelines, patents, preprints outside PubMed, or proprietary sources.
Evidence about a related disease, target, pathway, drug class, or analogue is
indirect unless it directly studies the exact disease-drug pair.

## Labels

### `supported`

Assign `supported` when reliable human evidence directly evaluates the exact drug
for the exact disease and reports a credible positive efficacy signal that is not
outweighed by stronger negative, contradictory, or invalidating evidence. Qualifying
evidence normally requires a peer-reviewed controlled clinical study; multiple
consistent peer-reviewed human studies may also qualify when a controlled trial is
not feasible for the rare disease. Open Targets known-drug records can corroborate
the conclusion but cannot alone establish this label. Safety, dosing, and regulatory
approval are outside this label's meaning.

### `weakly_supported`

Assign `weakly_supported` when there is a biologically plausible, source-backed
reason to investigate the exact disease-drug pair, but direct reliable human
efficacy evidence is absent, preliminary, too small, or too uncertain for
`supported`. Examples include direct positive animal or in-vitro disease-model
evidence, a small uncontrolled human case series, or a coherent mechanism connecting
the drug's established target to the disease. Target association, pathway
similarity, or evidence from a related disease is indirect and can support only this
label when the chain of reasoning is explicit and no stronger negative evidence
dominates.

### `unsupported`

Assign `unsupported` when sufficient reliable evidence was retrieved to assess the
pair and it affirmatively fails to support investigation. This includes credible
direct evidence showing no efficacy, worsening, an incompatible mechanism, or a
well-supported contradiction that outweighs positive evidence. Do not use
`unsupported` merely because searches found no positive evidence, retrieval failed,
or only sparse evidence was available. A negative study in a related disease alone
does not establish this label for the exact pair.

### `insufficient_evidence`

Assign `insufficient_evidence` when the available reliable evidence cannot support a
defensible directional judgment. Use it when retrieval fails materially, no relevant
evidence is found, records cannot be matched confidently to the exact disease or
drug, citations are inaccessible or too incomplete to assess, or the evidence is so
sparse or conflicting that neither weak support nor affirmative lack of support is
justified. This is the required abstention label and must not be treated as a
negative efficacy finding.

## Evidence Strength

Use this default ordering, then account for study quality, directness, consistency,
and relevance:

1. Direct peer-reviewed controlled human efficacy evidence for the exact pair.
2. Other direct human evidence for the exact pair, including uncontrolled studies
   and case series.
3. Direct animal disease-model evidence for the exact pair.
4. Direct in-vitro disease-model evidence for the exact pair.
5. Mechanistic evidence connecting an established drug target to the disease.
6. Target association, pathway similarity, related-disease evidence, or drug-class
   analogy.

The ordering is not automatic scoring. A weak, biased, retracted, or contradicted
study can carry less weight than a lower-ranked but reliable body of evidence.

## Mandatory Rules

- Human clinical evidence can support `supported`; animal or in-vitro evidence alone
  cannot.
- A drug targeting a disease-associated protein is indirect evidence, not proof of
  efficacy.
- Pathway similarity and related-disease efficacy are indirect evidence.
- Open Targets associations and known-drug records are supporting evidence, not
  standalone proof of efficacy.
- Retracted evidence receives zero positive weight.
- Material corrections, serious study-quality defects, and credible contradictions
  lower confidence and can lower the label.
- One weak negative result does not automatically outweigh a stronger consistent
  positive body of evidence.
- Absence of positive evidence is not evidence of no efficacy.
- Retrieval failure, identity ambiguity, and missing reliable evidence require
  `insufficient_evidence`, not `unsupported`.
- Every material claim and relationship must cite one or more evidence items.
- Do not infer safety, dosing, clinical readiness, or treatment suitability from an
  evidence-support label.

## Decision Procedure

Apply these steps in order:

1. **Verify identity.** Confirm that evidence refers to the exact disease and exact
   drug. If identity cannot be resolved, assign `insufficient_evidence`.
2. **Check retrievability.** If Open Targets or PubMed retrieval failed enough to
   prevent assessment, assign `insufficient_evidence`.
3. **Separate direct from indirect evidence.** Direct evidence studies the exact
   pair. Target, pathway, analogue, drug-class, and related-disease evidence are
   indirect.
4. **Assess validity.** Exclude retracted evidence and discount inaccessible,
   corrected, severely biased, or poorly described studies.
5. **Resolve contradictions.** Prefer more direct, reliable, adequately powered, and
   consistently replicated evidence. Explain unresolved conflicts.
6. **Assign the label using this precedence:**
   - Credible direct negative evidence that outweighs positives: `unsupported`.
   - Credible direct positive human efficacy evidence: `supported`.
   - Credible positive but preliminary, preclinical, or indirect evidence:
     `weakly_supported`.
   - Otherwise: `insufficient_evidence`.
7. **Set confidence.** Confidence reflects confidence in the assigned triage label,
   not confidence that the drug works.

## Confidence Anchors

- `0.90-1.00`: highly reliable, direct, consistent evidence; little ambiguity.
- `0.70-0.89`: clear label with limited uncertainty or minor contradictions.
- `0.50-0.69`: defensible label with meaningful limitations.
- `0.30-0.49`: substantial uncertainty; usually prefer `insufficient_evidence`.
- `0.00-0.29`: evidence or identity is too unreliable for a directional judgment.

## Worked Adjudication Examples

These examples define how to label evidence patterns; they are not dataset entries.

### Example 1: Controlled Human Signal

PubMed contains a peer-reviewed controlled trial of the exact drug in the exact rare
disease with a credible positive primary efficacy result. A smaller observational
study is also positive, and no stronger contradiction is present.

**Label:** `supported`  
**Reason:** Direct reliable human efficacy evidence supports investigation.

### Example 2: Positive Animal Model Plus Target Association

Open Targets links the disease to a protein inhibited by the drug. A PubMed animal
disease-model study of the exact drug reports improvement. No human study exists.

**Label:** `weakly_supported`  
**Reason:** The direct preclinical result and mechanism justify investigation, but
animal and target-association evidence cannot establish strong human support.

### Example 3: Target Association Only

Open Targets reports a strong disease-target association, and the drug is known to
act on that target. No study directly evaluates the drug in the disease.

**Label:** `weakly_supported`  
**Reason:** There is a coherent but indirect mechanistic rationale. The explanation
must not claim efficacy.

### Example 4: No Relevant Evidence

Open Targets and PubMed searches complete successfully, but retrieved records do not
evaluate the exact disease-drug pair and do not establish a coherent mechanistic
chain.

**Label:** `insufficient_evidence`  
**Reason:** Lack of support is not affirmative evidence that the drug is ineffective.

### Example 5: Credible Direct Negative Evidence

A reliable controlled human study of the exact pair reports no efficacy, and a
second direct study is consistent. A small positive animal study exists.

**Label:** `unsupported`  
**Reason:** Sufficient direct negative human evidence outweighs the preclinical
positive result.

### Example 6: Conflicting Small Human Reports

Two very small uncontrolled reports of the exact pair conflict, and no controlled
study or stronger evidence resolves the disagreement.

**Label:** `insufficient_evidence`  
**Reason:** The conflict prevents a defensible directional judgment.

### Example 7: Positive Study Is Retracted

The only direct positive report for the exact pair is retracted. Indirect target
association evidence remains.

**Label:** `weakly_supported` if the remaining indirect mechanism is coherent and
source-backed; otherwise `insufficient_evidence`.  
**Reason:** Retracted evidence receives zero positive weight.

## Adjudication Record Requirements

For every gold label, record:

- Pair ID and exact disease and drug identifiers.
- Assigned label and confidence.
- Direct and indirect evidence items considered.
- Material contradictions, retractions, and retrieval limitations.
- Rationale applying the decision procedure.
- Acceptable source identifiers supporting the rationale.
- Reviewer identity, review date, and any adjudication notes.

Any case that cannot be resolved using this guide must be documented as a labeling
guide ambiguity before the pilot freeze. Resolving it requires a new guide and
contract version.
