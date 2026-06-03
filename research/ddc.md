# Demand-Driven Context: A Methodology for Building Enterprise Knowledge Bases Through Agent Failure

## Paper Link: <https://arxiv.org/pdf/2603.14057>

## Current Agentic Landscape

Large Language Models possess expert-level reasoning but "zero institutional memory" when applied to specific enterprise environments. Current "bottom-up" agentic improvements like Reflexion or ACE focus on optimizing execution strategies (how an agent uses tools or reasons) but cannot autonomously acquire domain knowledge that exists only as "tribal knowledge" in human heads (e.g., "why a specific batch job retries three times"). Meanwhile, RAG fails when the necessary documentation doesn't exist to be retrieved in the first place

This research impacts the Data (Knowledge Bases) and Inference (Context Engineering) layers. It moves the focus from automated behavioral tuning to human-curated domain context.

As AI agents are used/built for enterprise purposes, an important part of the process becomes knowledge discovery and documenting what knowledge is missing.

## Core Concepts?

A real-world enterprise problem (e.g., an SRE incident report) and an initially empty or partial Knowledge Base (KB) serves as input.

**Demand-Driven Curation (DDC)**: Inspired by Test-Driven Development (TDD), the system follows a 9-step "Red-Green-Refactor" cycle:

- Agent Fails: The agent attempts a problem with current context and fails (The "Red" state).
- Information Checklist: The agent identifies precisely what it doesn't know, generating a checklist of missing terminology, systems, and business rules.
- Human Filling: A domain expert provides the "minimum viable context" to satisfy the checklist.
- Meta-Model Structuring: Answers are graduated into a Typed Entity Meta-Model (entities like system, capability, jargon-tech) stored as version-controlled markdown files with YAML frontmatter for machine navigation

This provides a high-precision, reusable knowledge base that improves agent performance on subsequent problems.

### Agent-Generated Information Checklist

The checklist of missing context/information needed to turn the agent's failure into a structured "demand signal" that directs human curation effort only where it is needed.

## Validity Check: Does it actually generalize?

### Assumptions

- Success relies on the availability of human experts and the agent's ability to accurately identify its own knowledge gaps without hallucinating a false "vocabulary without insight".

### Benchmarks vs. Robust Settings

- Validated through nine cycles in a synthetic retail e-commerce SRE domain, where the agent diagnosed root causes for stuck orders and deployment errors.

### Breaking Points at Scale

- **Manual curation bottleneck**: The authors propose a scaling architecture (unvalidated) using semi-automated curation and PR-based governance to manage parallel team contributions

### Real Capability or Artifact

The observed Convergence Trend, where the reuse of existing entities increased from 0.0 to 0.75 by cycle nine, suggests a real stabilization of the knowledge base rather than an evaluation artifact. However, the "Convergence Hypothesis" (that 20–30 cycles cover a domain role) remains a hypothesis.

## Paper Summary

1. **TDD for Context**: Treat knowledge engineering like software testing. Don't document anything until an agent "fails" and proves the demand for that specific fact.

2. **Failure as Signal**: Shift from viewing agent failure as a performance bug to viewing it as a diagnostic tool for institutional knowledge gaps.

3. **Domain vs. Strategy**: Distinguish between how an agent thinks (Strategy) and what it knows (Domain). DDC fixes the latter so the former has high-quality facts to process

4. **Typed Meta-Model Architecture**: Don't just append text, structure domain facts into typed entities (system, api, decision) so they can be navigated by both humans and LLM tools.

5. **The Convergence asymptote**: Knowledge bases for specific roles don't need infinite growth. They stabilize once the foundational "tribal knowledge" is captured through initial problem-solving cycles.
