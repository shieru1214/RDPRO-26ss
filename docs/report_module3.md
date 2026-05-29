# Module 3 — Model Knowledge Base & Retrieval

**Status.** Core prototype is functional: a typed directed graph (14 backbones, 22 HuggingFace checkpoints, 7 heads, 6 losses, 3 optimizers) with semantically distinct edge types (`compatible_with`, `requires` for non-swappable architectural constraints, `preferred_when` with structured boolean conditions). Retrieval runs as a four-stage funnel — scale-band filtering, tier-based gating, hybrid scoring (structured 60% + `all-MiniLM-L6-v2` vector similarity 40%), graph traversal — and emits full model configurations with training strategy recommendations as structured task lists for Module 4. Covered by a 29-test suite.

**Biggest challenge.** Whether to keep deterministic hybrid scoring or replace it with an LLM reasoning agent. The current scorer is consistent but fails when constraints conflict or fall outside the schema; an agent handles ambiguity naturally but introduces non-determinism. Planned resolution: retain hard-filter stages as a pre-filter and introduce the agent only over the reduced candidate set, preserving upstream correctness guarantees.

**Remaining plan.** Near-term: activate `domain_transfer` scoring, complete `preferred_when` evaluation for loss/pretrained nodes, implement Module 2 keyword mapping. Mid-term: agent integration, end-to-end pipeline connection.

**Work division.** Module 3 (KB, retrieval, Module 4 interface) — [name].

**Delay mitigation.** Module 4's interface is already defined and tested, keeping downstream development unblocked. The deterministic pipeline serves as a fallback if agent integration is delayed.
