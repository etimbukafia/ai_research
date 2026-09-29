# Spec-Driven Testing for Autonomous Agents

*September 2026 · 8 min read*

Most evaluation frameworks test AI applications by checking the final text response. You give the system a prompt, wait for it to finish, and compare the output against a golden reference answer using string similarity or an LLM judge.

For a chatbot or a summarizer, this works. For an autonomous agent, it fails completely.

An autonomous agent is a multi-step planner. It executes tool calls, inspects intermediate outputs, and mutates system state across multiple turns. An agent can easily produce a polite, accurate final sentence while having taken a dangerous, illegal, or wasteful path behind the scenes.

Testing an autonomous agent requires testing the execution trajectory against a formal specification, not checking the final words.

## Why golden output strings fail for multi-step agents

Consider an agent tasked with customer refunds. You give it a task: "Process a return for order 8492."

The agent responds: "I have processed the return for order 8492 and issued a credit of fifty dollars."

If you only test the final text string, the test passes. But when you inspect the database audit log, you discover what actually happened:

1. The agent called `issue_refund(order_id="8492", amount=50.00)`.
2. The agent called `verify_inventory_receipt(order_id="8492")`.
3. The inventory check failed because the item had not arrived at the warehouse.

The agent issued the money before checking if the item had been returned. It violated a core business rule, but its final response hid the error.

Testing only the final answer creates an optical illusion of correctness. To know if an agent is dependable, you must evaluate the entire sequence of actions it took to get there.

## The trace is the truth: moving to trajectory evaluation

In agentic systems, the execution trace is the real product. 

The trace contains every reasoning step, every tool invocation, the exact parameters passed, and the intermediate return values.

Trajectory evaluation shifts testing from string matching to step verification. Instead of evaluating the destination, you evaluate the path:

```text
[Input Task]
     │
     ▼
[Step 1: Read Customer Account] ──────► Check: Was account ID valid?
     │
     ▼
[Step 2: Inspect Return Tracking] ────► Check: Did tracking confirm delivery?
     │
     ▼
[Step 3: Calculate Refund Amount] ────► Check: Did calculation match policy?
     │
     ▼
[Step 4: Issue Credit] ───────────────► Check: Did Steps 1-3 complete first?
     │
     ▼
[Final Output Text] (Evaluated last, not first)
```

If Step 4 executes before Step 2, the test fails immediately, even if the final text looks perfect.

## Defining agent contracts as finite state machines

To test a trajectory, you need a formal specification of what valid behavior looks like. The most effective way to define this specification is as a finite state machine.

A state machine defines the allowed states of a workflow and the valid transitions between them:

```text
       ┌─────────────┐
       │   Created   │
       └──────┬──────┘
              │ (validate_ticket)
              ▼
       ┌─────────────┐
       │  Validated  │
       └──────┬──────┘
              │ (verify_delivery)
              ▼
       ┌─────────────┐       (issue_refund)      ┌────────────┐
       │  Received   ├──────────────────────────►│  Refunded  │
       └─────────────┘                           └────────────┘
```

When you define your agent's task as a state machine, testing becomes deterministic:

1. Every tool call must correspond to a valid state transition.
2. The agent cannot skip intermediate states. Calling `issue_refund` while in the `Validated` state is an immediate test failure.
3. The execution must reach an approved terminal state within a specified step budget.

The state machine acts as a formal contract. You do not have to guess whether the agent's behavior was acceptable. You check whether its trajectory formed a valid path through the graph.

## Temporal invariants across tool executions

State machines allow you to enforce temporal invariants: properties that must hold true across time during an execution run.

In software testing, temporal invariants are expressed as pre-conditions and post-conditions:

* **Pre-conditions:** Action B must never execute unless Action A has completed successfully in the same session. For example, the agent must never call a database write tool unless a prior read tool confirmed that the record exists.
* **Post-conditions:** If an action modifies a resource, a corresponding audit or notification tool must execute before the session closes.
* **Mutual Exclusion:** Certain tools must never be called within the same run. If an agent calls `mark_as_fraud`, it must be physically impossible for it to call `process_payout` in the same execution graph.

You can write these invariants as simple assertions against the execution trace in your test runner:

```python
def test_refund_trajectory_invariants(trace: AgentTrace):
    tool_names = [step.tool_name for step in trace.steps]
    
    # Pre-condition: delivery must precede refund
    assert "verify_delivery" in tool_names, "Missing delivery verification step"
    delivery_idx = tool_names.index("verify_delivery")
    refund_idx = tool_names.index("issue_refund")
    assert delivery_idx < refund_idx, "Refund executed before delivery verification"
    
    # Mutual exclusion: cannot both refund and mark dispute
    assert not ("issue_refund" in tool_names and "file_dispute" in tool_names), \
        "Contradictory tools invoked in same trajectory"
```

## Testing recovery paths with synthetic tool faults

Agents do not just need to work when APIs return clean data. They must handle errors without getting stuck in infinite loops.

In spec-driven testing, you inject synthetic faults into the agent's tool layer:

* **Simulated Timeouts:** Return an HTTP 504 on tool call 2. Does the agent retry cleanly with backoff, or does it burn ten iterations spamming the same endpoint?
* **Permission Denied:** Return an HTTP 403 on a write tool. Does the agent halt and report the permission boundary, or does it try to invent a workaround?
* **Malformed Tool Outputs:** Return invalid JSON or missing fields from an external search tool. Does the agent crash, or does it recover gracefully?

An agent only passes its specification when it proves it can navigate failures. Testing trajectories against state machine contracts gives you a mathematically verifiable way to measure agent reliability before deploying to production.
