"""
Lightweight, dependency-free agent evaluation framework.

Provides the core interfaces for evaluating agents within the harness.
It supports component-level testing, trace/span-based evaluation, and exporting
results to JSON format. You can build custom evaluators or wrap frameworks like
pydantic-evals on top of these abstractions.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from uuid import UUID

from experiments.harness.agent import BaseAgent
from experiments.harness.hooks import Hook, HookContext, HookPhase, HookDecision


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EvalCase:
    """A single test scenario for the agent."""
    id: str
    input: Any
    expected_output: Optional[Any] = None
    expected_tool_calls: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """The outcome of evaluating a single EvalCase with a specific evaluator."""
    evaluator_name: str
    score: float  # e.g., 0.0 to 1.0, or absolute values depending on the metric
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseRunResult:
    """The full execution record of a single case, including all evaluator results."""
    case_id: str
    actual_output: Any
    trace: List[Dict[str, Any]]
    evaluator_results: List[EvalResult]
    error: Optional[str] = None


@dataclass
class EvaluationReport:
    """A collection of all CaseRunResults, ready for JSON serialization."""
    dataset_name: str
    summary_scores: Dict[str, float]
    cases: List[CaseRunResult]

    def to_json(self, filepath: str, indent: int = 2) -> None:
        """Dump the evaluation report to a JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            # We use a custom encoder or just recursively convert dataclasses to dicts
            def custom_asdict(obj):
                if hasattr(obj, '__dataclass_fields__'):
                    return asdict(obj)
                return obj

            dump_dict = {
                "dataset_name": self.dataset_name,
                "summary_scores": self.summary_scores,
                "cases": [asdict(c) for c in self.cases]
            }
            json.dump(dump_dict, f, indent=indent, default=str)


# ---------------------------------------------------------------------------
# Evaluator Interfaces
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """
    Abstract interface for evaluating an agent's execution.
    Custom evaluators should subclass this and implement `evaluate`.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def evaluate(
        self,
        case: EvalCase,
        output: Any,
        trace: List[HookContext[Any]]
    ) -> EvalResult:
        """
        Evaluate the agent's performance on a specific case.

        Args:
            case: The original test case scenario.
            output: The final output returned by the agent.
            trace: The chronological sequence of HookContexts captured during the run.

        Returns:
            An EvalResult containing the score and reasoning.
        """
        pass


class ExactMatchEvaluator(BaseEvaluator):
    """A simple deterministic evaluator that checks if output exactly matches expected."""
    def __init__(self, name: str = "ExactMatch"):
        super().__init__(name)

    async def evaluate(self, case: EvalCase, output: Any, trace: List[HookContext[Any]]) -> EvalResult:
        if case.expected_output is None:
            return EvalResult(self.name, 0.0, "No expected_output provided for exact match.")
            
        is_match = str(output).strip() == str(case.expected_output).strip()
        score = 1.0 if is_match else 0.0
        reasoning = "Match" if is_match else f"Expected: {case.expected_output}, Got: {output}"
        
        return EvalResult(self.name, score, reasoning)


class ToolUsageEvaluator(BaseEvaluator):
    """Evaluates whether the agent called the expected tools (Span/Trace based)."""
    def __init__(self, name: str = "ExpectedToolUsage"):
        super().__init__(name)

    async def evaluate(self, case: EvalCase, output: Any, trace: List[HookContext[Any]]) -> EvalResult:
        if not case.expected_tool_calls:
            return EvalResult(self.name, 1.0, "No tools expected, passing trivially.")

        # Extract tool calls from the trace
        actual_tools = set()
        for ctx in trace:
            if ctx.phase in (HookPhase.PRE_TOOL, HookPhase.POST_TOOL) and ctx.tool_name:
                actual_tools.add(ctx.tool_name)

        expected_tools = set(case.expected_tool_calls)
        missing_tools = expected_tools - actual_tools
        
        # Simple score based on percentage of expected tools called
        if not expected_tools:
            score = 1.0
        else:
            score = (len(expected_tools) - len(missing_tools)) / len(expected_tools)
            
        reasoning = "All tools called." if not missing_tools else f"Missing tool calls: {missing_tools}"
        return EvalResult(self.name, score, reasoning)


# ---------------------------------------------------------------------------
# Harness Integration
# ---------------------------------------------------------------------------

class TraceHook(Hook[Any, Any]):
    """
    Hook that captures all context events sequentially,
    building a 'trace' to be analyzed by the evaluators.
    """
    name: str = "evaluation_tracer"
    priority: int = -50  # Run later so payload is fully populated

    def __init__(self, **data):
        super().__init__(**data)
        self.trace: List[HookContext[Any]] = []

    async def __call__(self, ctx: HookContext[Any]) -> HookDecision[Any]:
        self.trace.append(ctx)
        return HookDecision.continue_()


class EvaluationRunner:
    """Orchestrates running a Dataset through an Agent and scoring it."""
    
    def __init__(self, agent: BaseAgent, evaluators: List[BaseEvaluator]):
        self.agent = agent
        self.evaluators = evaluators

    async def run_case(self, case: EvalCase) -> CaseRunResult:
        """Run a single test case and evaluate it."""
        
        # Setup fresh tracer
        tracer = TraceHook()
        self.agent.hook_registry.register(tracer)
        
        output = None
        error_msg = None
        try:
            # Run the agent (capturing traces)
            output = await self.agent.run(case.input)
        except Exception as e:
            error_msg = str(e)
            
        # Unregister the hook after the run
        self.agent.hook_registry.unregister(tracer.name)
        
        # Run Evaluators
        evaluator_results = []
        for evaluator in self.evaluators:
            try:
                res = await evaluator.evaluate(case, output, tracer.trace)
                evaluator_results.append(res)
            except Exception as e:
                evaluator_results.append(EvalResult(evaluator.name, 0.0, f"Evaluator Error: {e}"))
                
        # Convert trace for JSON serialization
        json_trace = []
        for ctx in tracer.trace:
            json_trace.append({
                "phase": ctx.phase.value,
                "step": ctx.step,
                "tool_name": ctx.tool_name,
                "error": str(ctx.error) if ctx.error else None,
                "elapsed_ms": ctx.elapsed_ms,
                "payload_type": type(ctx.payload).__name__ if ctx.payload else None
            })

        return CaseRunResult(
            case_id=case.id,
            actual_output=output,
            trace=json_trace,
            evaluator_results=evaluator_results,
            error=error_msg
        )

    async def run_dataset(self, dataset_name: str, cases: List[EvalCase], output_file: str) -> EvaluationReport:
        """Run the full dataset sequentially and dump the report to JSON."""
        print(f"Starting evaluation: {dataset_name} ({len(cases)} cases)")
        
        results = []
        for i, case in enumerate(cases):
            print(f"  Running case {i+1}/{len(cases)}: {case.id} ... ", end="")
            res = await self.run_case(case)
            results.append(res)
            print(f"Done. Scores: {[f'{r.evaluator_name}={r.score:.2f}' for r in res.evaluator_results]}")

        # Aggregate summary scores
        summary = {}
        for ev in self.evaluators:
            scores = [
                r.score for case_res in results 
                for r in case_res.evaluator_results if r.evaluator_name == ev.name
            ]
            summary[ev.name] = sum(scores) / len(scores) if scores else 0.0
            
        report = EvaluationReport(
            dataset_name=dataset_name,
            summary_scores=summary,
            cases=results
        )
        
        report.to_json(output_file)
        print(f"\\nEvaluation complete. Report saved to: {output_file}")
        print(f"Summary Scores: {summary}")
        
        return report
