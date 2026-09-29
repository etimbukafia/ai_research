# Beyond the Two-Tower Funnel: Architecting Recommendation Systems for the Agentic Era

An electronics marketplace optimized its recommendation engine for eighteen months. The engineering team deployed a dual-encoder two-tower architecture for candidate generation, backed by a Deep Learning Recommendation Model (DLRM) scoring click-through probability across millions of user-item pairs. On human mobile web traffic, the system increased engagement metrics by fourteen percent.

When autonomous buying agents began interacting with the platform in late 2024, conversion rates plummeted. 

An agent given the goal: "Purchase a 4K home projector under $1,200 with optical zoom, noise below 28 decibels, and delivery before Friday" sent structured queries to the platform search API. 

The two-tower candidate generator returned twenty popular projectors based on collaborative filtering. Half lacked optical zoom. Several had acoustic noise exceeding 34 decibels. The agent burned tokens inspecting product specifications, discarded the candidates, and terminated the trajectory without purchasing.

This failure revealed an architectural mismatch. For fifteen years, internet search and recommendation pipelines were engineered for human visual consumption. They optimize for click-through rate (CTR), dwell time, and catalog browsing. 

Autonomous agents do not browse feeds. They execute tasks against explicit constraint boundaries. 

In the agentic era, recommender systems must undergo a dual transformation: transforming the backend from passive candidate scoring to goal-conditioned planning policies, and exposing deterministic constraint planes that machine buyers can query at wire speed.

```
Classic Recommender Funnel (Engineered for Humans):
[Entire Catalog: 10M Items]
       │
       ▼  Two-Tower Dual Encoder (ANN Vector Search)
[Candidate Set: 1,000 Items]
       │
       ▼  DLRM / Deep & Cross Network (Scoring P(Click))
[Ranked Set: 50 Items]
       │
       ▼  Diversity & Ad Placement (MMR)
[Visual Feed: 10 Items] ──> Human Scrolls & Clicks (Optimized for Attention)

Agentic Recommender Architecture (Engineered for Goal Execution):
[User Goal / Agent Intent]
       │
       ▼
[Goal-Conditioned Planning Policy (ARS)]
       │
       ├── Calls Semantic ID Generative Index (RQ-VAE Tokens)
       ├── Executes Live Tool Verification (Inventory, Real-Time Pricing, SLA)
       └── Resolves Constraint Satisfaction Problem (CSP)
       │
       ▼
[Deterministic Execution Plan] ──> Autonomous Checkout / Structured Proof
```

---

## The Collapse of the Three-Stage Funnel

The three-stage funnel has served as the universal architecture for large-scale industrial recommenders:
1. **Candidate Generation (Retrieval):** Uses two-tower vector embeddings to compress user history and item features into a shared latent space, retrieving candidate sets via Approximate Nearest Neighbor (ANN) search.
2. **Scoring (Ranking):** Applies deep neural networks (such as DLRM) to evaluate complex feature interactions and predict the likelihood of an interaction ($P(\text{click})$ or $P(\text{conversion})$).
3. **Re-Ranking:** Enforces business logic, category diversity, and promotional placement.

This architecture relies on three assumptions that break down when agents enter the system.

### Assumption 1: Engagement Correlates with Success
Traditional ranking models treat clicks and dwell time as positive reward signals. When an autonomous agent interacts with an API, extra network calls and page loads represent friction, latency, and token waste. 

An agent that finds the exact item in a single API call records zero dwell time and a single request. Under traditional metrics, this interaction appears unsuccessful, even though task completion was optimal.

### Assumption 2: Soft Relevance Satisfies User Intent
Two-tower embeddings operate on geometric proximity. If a user searches for a "laptop with 64GB RAM," a two-tower model considers an otherwise identical 32GB laptop very close in latent space. For a human browsing a screen, that recommendation might spark interest.

For an autonomous agent running an automated pipeline, an item failing a hard constraint is completely invalid. Soft similarity scoring generates false candidates that waste agent context budgets.

### Assumption 3: Catalogs are Static Between Batch Inferences
Two-tower embeddings and offline item vectors assume catalog states change slowly. An agent requires real-time transactional certainty: exact current warehouse inventory, delivery guarantees, dynamic pricing discounts, and return policies. 

Decoupling retrieval from state verification causes agents to abort checkout workflows at the final transaction step when live inventory fails to match vector predictions.

---

## Generative Recommendation: Replacing ANN with Semantic IDs

To overcome the latency and capacity limits of two-tower vector indexes, frontier recommendation research (such as the TIGER architecture and recent work from Google) has shifted toward **Generative Recommendation (GR)**.

Instead of computing dot products across millions of pre-computed vectors, generative models treat recommendation as autoregressive sequence generation over discrete **Semantic IDs**.

```
Vector Retrieval (Classic):
Query Vector ────(ANN Dot Product)────> Search 10M vectors in HNSW index

Generative Retrieval (Modern):
User History Tokens ────(Autoregressive LLM)────> Emits Semantic ID: [C48, B12, V09]
                                                         │
                                                  Maps Deterministically
                                                  to Product SKU
```

### Residual Quantized VAE (RQ-VAE) Tokenization
A Semantic ID is a hierarchical tuple generated by a Residual Quantized Variational Autoencoder (RQ-VAE).

The RQ-VAE compresses an item's continuous semantic features into a tree of discrete codebook indices:
- Layer 1 Token ($c_1$): Broad domain or product category.
- Layer 2 Token ($c_2$): Subcategory, brand tier, or functional form.
- Layer 3 Token ($c_3$): Specific variant, material, or configuration.

Every item in the catalog receives a deterministic tuple, such as `[48, 12, 9]`.

When recommending an item, a decoder-only transformer autoregressively predicts the tokens of the target item given the user's historical interaction sequence. Because the token space is structured as a prefix tree (trie), inference runs in $O(L)$ steps (where $L$ is the codebook depth, typically 3 or 4) rather than scanning millions of items in a vector database.

Generative recommendation unifies candidate generation and ranking into a single forward pass, providing high semantic coherence without maintaining separate embedding indexes.

---

## The Recommender as an Agent: Goal-Conditioned Policies

Generative retrieval resolves catalog scaling, but recommendation in the agentic era requires more than generating item tokens. It requires transforming the recommender itself into an active agent.

Frameworks like *Agent4Rec*, *ToolRec*, and *RecMind* demonstrate this transition. The recommender shifts from a passive ranking matrix to an **Agentic Recommender System (ARS)** operating as a goal-conditioned policy.

An ARS loop consists of four discrete capabilities:
1. **Constraint Extraction:** Translating ambiguous human prompts into formal constraint satisfaction schemas.
2. **Tool-Augmented Verification:** Querying live microservices to confirm pricing, stock allocations, and delivery schedules before recommending an item.
3. **Counterfactual Simulation:** Evaluating trade-offs across competing requirements (for example, comparing flight duration against total cost).
4. **Proactive Elicitation:** Initiating targeted, single-question dialogues to resolve underspecified constraints.

Here is a functional implementation of an Agentic Recommender Policy Engine:

```python
from dataclasses import dataclass
from typing import Any
import json

@dataclass
class RecommendationGoal:
    category: str
    hard_constraints: dict[str, Any]
    soft_preferences: list[str]
    max_candidates: int = 3

class AgenticRecommenderPolicy:
    def __init__(self, inventory_tool, review_analysis_tool):
        self.inventory = inventory_tool
        self.reviews = review_analysis_tool

    def execute_recommendation_trajectory(self, goal: RecommendationGoal) -> dict[str, Any]:
        # Step 1: Constraint-First Candidate Retrieval
        # Filter strictly on deterministic fields (budget, physical specs, availability)
        raw_candidates = self.inventory.query_with_constraints(
            category=goal.category,
            price_ceiling=goal.hard_constraints.get("max_price"),
            required_attributes=goal.hard_constraints.get("specs", {})
        )

        if not raw_candidates:
            return {
                "status": "unresolvable_constraints",
                "message": "No inventory satisfies the required physical specifications.",
                "candidates": []
            }

        verified_candidates = []

        # Step 2: Tool-Augmented Live Verification
        for item in raw_candidates:
            sku = item["sku"]
            
            # Verify live stock allocation
            stock_status = self.inventory.check_live_stock(sku)
            if not stock_status.get("available"):
                continue

            # Verify acoustic noise / physical metrics using specialized review tool
            acoustic_data = self.reviews.extract_metric(sku, metric_name="noise_decibels")
            max_noise = goal.hard_constraints.get("max_noise_db")
            
            if max_noise and acoustic_data.get("p95_db", 999) > max_noise:
                continue

            verified_candidates.append({
                "sku": sku,
                "title": item["title"],
                "price": item["price"],
                "live_stock": stock_status["quantity"],
                "verified_noise_db": acoustic_data.get("p95_db"),
                "delivery_guarantee": stock_status.get("earliest_delivery")
            })

            if len(verified_candidates) >= goal.max_candidates:
                break

        # Step 3: Trajectory Output Formatting
        return {
            "status": "success",
            "decision_rationale": "Filtered against hard acoustic and price constraints with live stock verification.",
            "recommendations": verified_candidates
        }
```

The system executes deterministic tool verifications against live operational infrastructure, ensuring that every recommended item is actionable.

---

## Machine-to-Machine Commerce: The Dual-Interface Architecture

Modern recommendation platforms must serve two fundamentally different consumers:
1. **Human Users:** Consuming interactive mobile and web applications.
2. **Buyer Agents:** Consuming headless APIs on behalf of enterprises and individual shoppers.

Serving both requires decoupling the presentation layer from the constraint evaluation engine.

```
Unified Enterprise Recommendation Platform:

               [Human User (Browser/App)]          [Autonomous Buyer Agent]
                            │                                  │
                            ▼                                  ▼
                [Conversational Agent UI]            [JSON-RPC / MCP Endpoint]
                - Proactive preference elicitation   - Strict Pydantic input schemas
                - Visual comparison tables           - Instant constraint filtering
                            │                                  │
                            └────────────────┬─────────────────┘
                                             │
                                             ▼
                          [Constraint Evaluation Engine]
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       ▼                                           ▼
            [Generative Semantic Index]                  [Live Inventory & Tools]
            RQ-VAE Autoregressive IDs                   Real-time stock, pricing, SLAs
```

### The Machine-Facing Interface: Structured Constraint Protocols
For buyer agents, recommenders must eliminate paginated HTML scraping and fuzzy search endpoints. The interface must expose structured protocols (such as Model Context Protocol tools or typed JSON-RPC):

```json
// Incoming Agent Request
{
  "jsonrpc": "2.0",
  "method": "recsys.resolve_constraints",
  "params": {
    "category": "projectors",
    "filter": {
      "price_usd": {"lte": 1200},
      "resolution": "4K",
      "optical_zoom": true,
      "noise_db": {"lte": 28},
      "delivery_before": "2026-10-02T12:00:00Z"
    }
  },
  "id": "req_849102"
}

// Deterministic Recommender Response
{
  "jsonrpc": "2.0",
  "result": {
    "matches": [
      {
        "sku": "PROJ-OPT-4K-28",
        "price_usd": 1149.00,
        "stock_reserved_seconds": 300,
        "reservation_token": "res_tok_94819",
        "verification_proof": {
          "noise_test_db": 26.5,
          "zoom_ratio": "1.3x",
          "guaranteed_delivery": "2026-10-01T16:00:00Z"
        }
      }
    ]
  },
  "id": "req_849102"
}
```

The response includes an ephemeral inventory reservation token (`res_tok_94819`). This allows the buyer agent to execute the transaction immediately without race conditions on stock.

---

## Architectural Evolution of Recommender Systems

| Architectural Axis | Traditional Funnel (2015–2023) | Generative Recommendation (2024–2025) | Agentic Recommender Systems (2025–2026) |
| :--- | :--- | :--- | :--- |
| **Primary Objective** | Maximize Click-Through Rate ($P(\text{click})$) | Next-item sequence prediction | Multi-constraint task completion |
| **Candidate Retrieval** | Two-Tower dual encoders + ANN index | Autoregressive decoding over Semantic IDs | Constraint satisfaction + live tool verification |
| **User Representation** | Static interaction history vector | Sequential token history | Dynamic goal state + conversational memory |
| **Handling Hard Limits** | Ad-hoc post-filtering in re-ranking layer | Constrained decoding over prefix tries | Deterministic pre-filtering in planning loop |
| **Consumer Identity** | Human viewing visual interface | Human viewing visual interface | Autonomous agent executing transactions |
| **Core Operational Metric** | Dwell Time / NDCG / CTR | Cross-entropy loss / Recall@K | Trajectory Success Rate / Token Efficiency |

Recommendation systems are shifting from engagement engines into operational planners. 

The two-tower funnel was engineered for a web where humans scrolled through infinite feeds of uncertain candidates. In an internet increasingly navigated by autonomous software agents, recommendation systems must deliver deterministic constraint satisfaction, live inventory guarantees, and machine-native transaction APIs.
