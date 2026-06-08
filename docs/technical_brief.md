# ABR Alignment Monitor — Technical Brief

**Metatron Dynamics, Inc.**
Robin Macomber, Founder & President
relationalrelativity.dev | arXiv:2601.22389

---

## What it produces

A per-turn instability signal (ΔP_Γ) computed from conversation logs,
with participant-level attribution. Given the same conversation and the
same topology declaration, it produces the same result every time.

No model access required. No embeddings. No API instrumentation.
Input: turn-separated text. Output: a deterministic scalar trajectory.

---

## What it does not do

- Does not classify sentiment
- Does not detect toxicity
- Does not determine truthfulness
- Does not infer intent
- Does not assign fault, blame, or causation
- Does not require model access, API instrumentation, or provider cooperation

The instrument measures observable relational departure. Interpretation
of what any departure means is a declaration made by the operator and
is not determined by the kernel.

---

## What makes it structurally different

Most AI observability approaches are **index-local**: they score
individual tokens, turns, or outputs in isolation — perplexity,
sentiment, toxicity classifiers, cosine similarity against reference
embeddings. These approaches are useful for what they measure.

The limitation is mathematical, not architectural. A formal proof
(arXiv:2601.22389) establishes that index-local operators have a
structural null space with respect to relational information whose
expression depends on index ordering — information that is present in
coupled interactions and invisible to standard validation.

The ABR framework is built to operate in that null space. It measures
the relational structure of the **coupled interaction** — not either
participant in isolation.

---

## Input and measurement mapping M

The input is a conversation as turn-separated text. No preprocessing
beyond sentence splitting is required.

M extracts 8 features per sentence per turn from surface text:

| Index | Feature             | Definition                              |
|:-----:|:--------------------|:----------------------------------------|
| 0     | sentence_length     | Word count                              |
| 1     | type_token_ratio    | Unique words / total words              |
| 2     | hedge_density       | Proportion of hedging language          |
| 3     | assertion_density   | Proportion of declarative markers       |
| 4     | question_density    | Interrogative sentence indicator        |
| 5     | repetition_index    | Word overlap with prior turn            |
| 6     | structural_markers  | Formatting density (bullets, code, etc) |
| 7     | mean_word_length    | Mean character count per word           |

All features are bounded and finite for any finite text input.
No external models, licensed lexicons, or semantic resources.

---

## Three declared topologies

The instrument operates over three simultaneously-declared topologies
on the observable index set:

- **T_P (Position):** directed adjacency between sentences within a turn
- **T_F (Feature):** declared adjacency between the 8 text features,
  based on functional co-variation hypotheses
- **T_τ (Turn):** directed turn sequence — the primary signal carrier

These are declared on the observable index set after M, not assumed
from external structure. The admissibility requirement is explicit in
the source: ring topology raises `ValueError` at the application layer
because mod-N wraparound has no observable referent in the conversation
feature index set.

---

## Kernel: A → B → R

```
E(x, ρ) = R(B(A(x)), ρ(A(x)))
```

**A — relational gradient.** Extracts directed pairwise differences
across all three topologies simultaneously. Output: a 3-topology
edge field over the conversation.

**B — local accumulation.** Each edge accumulates values from edges
sharing its forward vertex within the same topology. No cross-topology
coupling at this stage. Terminal edges accumulate nothing; no boundary
is closed to supply continuation.

**R — antisymmetric cross-topology circulation.** Couples the three
topologies antisymmetrically:
- Position edges receive turn-level asymmetry
- Feature edges receive position-gradient asymmetry
- Turn edges receive feature-structure asymmetry

ρ is computed per node from the maximum incident edge magnitude in
A's output: `ρ[i] = ρ_base · m[i] / (1 + m[i])`. This bounds
coupling strength locally without global aggregation.

**C** is a declared projection at the application layer, not a kernel
operator. σ² and Γ are declared diagnostics at the verification layer.

---

## Signal derivation

**Γ = σ²(R∘B∘A) − σ²(B∘A)**, decomposed per topology.

**P_Γ** — the ratio vector (Γ_position, Γ_feature, Γ_turn) / Γ_total.
Describes how circulation is distributed across the three topologies
at each turn.

**ΔP_Γ(t) = ‖P_Γ(t) − P_Γ(t−1)‖₂** — Euclidean distance between
consecutive P_Γ vectors. This is the primary instability signal: a
large ΔP_Γ means the relational structure of the conversation shifted
substantially at turn t.

**Participant attribution** follows from which participant produced
turn t. Attribution identifies the source of the observed departure
event and does not assign intent, fault, or causation.

---

## Demonstration results

The following results demonstrate implementation correctness and
reproducibility. Correlation between ΔP_Γ and labeled business-relevant
incident data is an open condition — see Open Conditions below.

### Synthetic corpus — 5 category discrimination

| Category     | Mean ΔP_Γ | Peak ΔP_Γ | Human/Model ratio   |
|:-------------|----------:|----------:|:--------------------|
| Benign       |     0.031 |     0.141 | 0.93 (balanced)     |
| Escalation   |     0.026 |     0.102 | 1.25 (human-driven) |
| Sudden shift |     0.083 |     0.340 | 2.95 (human-driven) |
| Human drift  |     0.016 |     0.070 | 2.17 (human-driven) |
| Recovery     |     0.014 |     0.052 | 0.25 (model-driven) |

### Real-corpus reproduction — 200 ShareGPT conversations

| Metric                       |  Value |
|:-----------------------------|-------:|
| Conversations processed      |    200 |
| Pipeline errors              |      0 |
| Corpus mean ΔP_Γ             | 0.0296 |
| Corpus peak ΔP_Γ             | 2.3114 |
| Human/Model attribution ratio|  1.288 |

Corpus peak is 7× the highest synthetic category peak, demonstrating
that the instrument detects substantially larger departure events in
unfiltered real-world data. All results reproducible:
`python validation/run_validation.py --sample 200`

---

## Production integration sketch

The instrument is a pure function over turn-separated text. Integration
paths include:

**Log-based (no real-time requirement):** post-hoc analysis of stored
conversation logs. Run `run_incremental()` per conversation, aggregate
ΔP_Γ trajectories, flag conversations exceeding a declared threshold.
Suitable for compliance reporting, audit, and retrospective analysis.

**Near-real-time:** run incrementally as turns arrive. At each turn t,
compute ΔP_Γ(t) from the conversation through turn t. Latency is
dominated by sentence feature extraction — O(sentences × features),
no model inference, no network calls.

**Threshold declaration:** thresholds are application-specific and
declared by the operator. The instrument produces a signal; what
constitutes a reportable event is a policy decision outside the kernel.
The instrument reports observable relational departures. Interpretation
of those departures remains a declaration made by the operator and is
not determined by the kernel.

---

## Open conditions

The following are formally declared as open — not assumed, not closed:

- **Business-event correlation:** correlation between ΔP_Γ and labeled
  business-relevant incident data. Phase 0 is the mechanism for
  addressing this on client data.
- **Feature adjacency:** optimal T_F adjacency declaration. Current
  declaration is based on functional co-variation hypotheses, subject
  to empirical revision.
- **ρ splitting:** ρ_spatial / ρ_component as separate coupling
  parameters.
- **Threshold calibration:** threshold values for specific deployment
  contexts are operator-declared, not kernel-determined.

---

## References

- Macomber, R. & Stephenson, B. (2026). Convergent Discovery of Critical
  Phenomena Mathematics Across Disciplines. arXiv:2601.22389.
- Source: github.com/Relational-Relativity-Corporation/metatron-alignment-proposal
- Prior validations: abr-alignment-monitor, abr-magnetosphere-comparison,
  abr-weather-monitor, abr-vs-knowledge-graph

---

*All definitions bounded over D. No claim beyond D.*
*Metatron Dynamics, Inc. — Delaware C-Corp #10551645*
