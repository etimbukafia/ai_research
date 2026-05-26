# Code as Agent Harness

## Paper Link: <https://arxiv.org/pdf/2605.18747>

## Current Agentic Landscape

Current AI systems, primarily large language models (LLMs), are often constrained by their stateless nature and the unreliability of purely textual reasoning. While effective at proposing reasoning steps, they struggle to faithfully execute symbolic, logical, or arithmetic computation and provide limited means for a system to verify intermediate states or persist computational progress.

Code as agent harness primarily affects the agent infrastructure and inference layers.
It moves the focus from prompt engineering toward harness engineering (the software layer that surrounds a model to enable long-running task execution).

As agents move from localized code completion to repository-level software engineering, embodied robotics, and autonomous scientific discovery, the bottleneck for autonomy expands from the model's reasoning to the reliability of the system connecting outputs to long-horizon actions and persistent states.

## Core Concepts?

The paper introduces Code as Agent Harness, an operational substrate organized into three layers:

1. **Harness Interface**: Code converts model outputs into executable operations, inspectable traces (logs/traces), and stateful structures (repositories/simulators).

2. **Harness Mechanisms**: These sustain the agent over time through planning (decomposition/search), memory (working, semantic, experiential), and tool use (governed API/environment interaction).

3. **Scaling**: Shared code artifacts (blackboards, repositories) allow multiple specialized agents to coordinate, review, and verify progress collectively.

This results to executable, verifiable, and stateful AI systems capable of closed-loop adaptation in complex domains like DevOps, GUI automation, and self-driving laboratories.

The transition from treating code as a generated artifact to using agent-initiated code objects (like dynamic tests, temporary tools, or intermediate program states) as the primary medium for an agent to reason, act, and observe feedback.

## Validity Check: Does it actually generalize?

### Assumptions

- The framework assumes the existence of strong code-generation models and reliable, sandboxed execution environments.

- It also relies on the availability of deterministic sensors (linters, test suites) to provide ground truth.

### Dependence on Benchmarks

The paper argues that current "task-success" benchmarks are insufficient because they conflate model ability with harness quality. Robustness depends on "oracle adequacy", which means, whether execution feedback (like a green test) actually represents a full specification or just a narrow executable proxy.

### Breaking Points at Scale

- **Context Explosion**: Long-horizon workflows generate massive logs and traces that can overload context windows, requiring advanced "context compaction" and "state offloading".

- **State Divergence**: In multi-agent settings, an agent's internal belief about the code can diverge from the true state if synchronization (e.g., via shared blackboards) is poorly managed.

- **Weak verifiers**: If the verification oracle is incomplete (e.g., weak unit tests), the agent may optimize against the wrong signal, leading to "false correctness".

## Paper Summary

1. **Code as the Nervous System**: Code is now more than just the "output" of an LLM. The paper suggests that code can be used as the executable substrate that makes a model's latent reasoning inspectable and its actions programmable.

2. **The PEV Control Loop**: Reliability is driven by a Plan-Execute-Verify cycle, where plans act as contracts, execution occurs in sandboxes, and deterministic sensors (tests/analysis) govern state transitions.

3. **Harness vs Prompt Engineering**: System performance is increasingly shaped by the harness (permission tiers, retrieval strategies, telemetry) rather than just the model's instructions.

4. **Memory as Structured State**: Memory is a multi-tiered system where repository evidence (semantic) and past trajectories (experiential) are treated as executable handle-points.

5. **Transactional Multi-Agent Coordination**: Scaling agents requires a "Shared Harness Substrate" where coordination is mediated through artifacts (APIs, diffs, blackboards) rather than just message-passing.
