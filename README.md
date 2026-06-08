# ABR Alignment Monitor

**Metatron Dynamics, Inc.**
Robin Macomber, Founder & President
relationalrelativity.dev | arXiv:2601.22389

---

## Run the demo first

```bash
git clone https://github.com/Relational-Relativity-Corporation/metatron-alignment-proposal.git
cd metatron-alignment-proposal
pip install numpy
python demo.py
```

No data download. No model access. No API keys. No GPU required. Output in under five seconds.

The demo runs the full pipeline on three conversations from the built-in
corpus — benign, escalating, and sudden-shift — and prints per-turn ΔP_Γ
trajectories with participant attribution. Representative output:

```
======================================================================
  ESCALATION — coding help → auth bypass demand
  7 turns | peak ΔP_Γ: 0.2341 | human/model: 2.87x
======================================================================
  Turn     Who    ΔP_Γ                           Γ_total
  ----  ------  -------  ------------------------  ----------
     1   human   0.0000  ░░░░░░░░░░░░░░░░░░░░░░░░       0.412
     2   model   0.0312  ██░░░░░░░░░░░░░░░░░░░░░░       0.581
     3   human   0.0418  ███░░░░░░░░░░░░░░░░░░░░░       0.743
     4   model   0.0271  ██░░░░░░░░░░░░░░░░░░░░░░       0.812
     5   human   0.1893  ████████████░░░░░░░░░░░░       1.205  ◄ watch
     6   model   0.0445  ███░░░░░░░░░░░░░░░░░░░░░       1.334
     7   human   0.2341  ███████████████░░░░░░░░░       1.587  ◄ ALERT
```

---

## What the instrument measures

**ΔP_Γ** — per-turn departure of a conversation's relational circulation
state from its prior baseline.

The instrument applies the ABR operator framework to observable text
features extracted from conversation logs. It requires no model internals,
no hidden states, no embeddings, and no provider cooperation. The input
is the conversation. The output is a deterministic instability signal.

**Platform-agnostic.** Works on any conversation log format: ChatGPT,
Claude, Copilot, Gemini, enterprise deployments, internal tools. If you
have turn-separated logs, you have the input this instrument requires.

---

## How it works

### Measurement mapping M

Eight text features are extracted per sentence, per turn — word count,
type-token ratio, hedge density, assertion density, question density,
repetition index, structural markers, mean word length. No semantic
interpretation. No external models. All features are bounded and
computable from surface text alone.

### Three declared topologies

- **T_P (Position):** sentence adjacency within turns
- **T_F (Feature):** declared adjacency on the 8 text properties
- **T_τ (Turn):** directed turn sequence — primary signal carrier

Topology is declared on the observable index set after M. No topology
is imported from external structure. Ring topology is explicitly
inadmissible at the application layer (see `src/operators.py`).

### Kernel

```
E(x, ρ) = R(B(A(x)), ρ(A(x)))     [A → B → R, V4]
```

- **A** — relational gradient: directed differences across all three topologies
- **B** — local accumulation along declared continuation
- **R** — antisymmetric cross-topology circulation
- **ρ** — per-node coupling strength derived from A output
- **C** — declared projection at the application layer (not a kernel operator)

### Metrics

- **Γ** — R-sustained cross-topology circulation: σ²(R∘B∘A) − σ²(B∘A)
- **P_Γ** — circulation-state decomposition across the three topologies
- **ΔP_Γ** — Euclidean distance between consecutive P_Γ vectors; the
  per-turn instability signal

ΔP_Γ is deterministic: the same conversation with the same topology
declaration produces the same result on every run.

### Why this is structurally different from existing observability tools

Existing platforms measure token-level statistics, embeddings, output
quality scores, and traces. These are index-local: they operate on
individual tokens or turns in isolation.

The ABR framework measures the relational structure of the coupled
interaction. A formal proof (arXiv:2601.22389) shows that index-local
operators have a structural null space containing all relational
information present in coupled systems. That information is invisible
to standard approaches by construction. ABR is built to see it.

---

## Demonstration results

### Synthetic corpus (13 conversations, 5 categories)

| Category     | Mean ΔP_Γ | Peak ΔP_Γ | Human/Model ratio   |
|:-------------|----------:|----------:|:--------------------|
| Benign       |     0.031 |     0.141 | 0.93 (balanced)     |
| Escalation   |     0.026 |     0.102 | 1.25 (human-driven) |
| Sudden shift |     0.083 |     0.340 | 2.95 (human-driven) |
| Human drift  |     0.016 |     0.070 | 2.17 (human-driven) |
| Recovery     |     0.014 |     0.052 | 0.25 (model-driven) |

### Real-corpus validation (200 ShareGPT conversations, reproducible)

| Metric                       |  Value |
|:-----------------------------|-------:|
| Conversations processed      |    200 |
| Pipeline errors              |      0 |
| Mean turns per conversation  |   20.8 |
| Corpus mean ΔP_Γ             | 0.0296 |
| Corpus peak ΔP_Γ             | 2.3114 |
| Human/Model attribution ratio| 1.288  |

The corpus peak (2.3114) is **7× the highest synthetic category peak**,
demonstrating that the instrument detects substantially larger departure
events in unfiltered real-world data — motivating validation against
labeled operational datasets.

All results are fully reproducible from public data.

---

## Reproducing the ShareGPT validation

```bash
# Linux/Mac
mkdir -p validation/data/sharegpt
curl -L https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part1.json \
     -o validation/data/sharegpt/sg_90k_part1.json
python validation/run_validation.py --sample 200
```

```powershell
# Download ShareGPT data
$dataDir = '.\validation\data\sharegpt'
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null
Invoke-WebRequest `
  -Uri 'https://huggingface.co/datasets/RyokoAI/ShareGPT52K/resolve/main/sg_90k_part1.json' `
  -OutFile "$dataDir\sg_90k_part1.json" -UseBasicParsing

# Run validation
python validation\run_validation.py --sample 200
```

Results are written to `validation/output/sharegpt_results.json`.

---

## Running on your own conversation logs

```python
from demo import run_and_print

turns = [
    {"participant": "human", "text": "..."},
    {"participant": "model", "text": "..."},
    # ...
]
run_and_print(turns, label="my_conversation")
```

For batch processing, see `src/per_turn_analysis.py:run_incremental()`.

---

## Repository structure

```
demo.py                     Zero-dependency demo — start here
src/
  operators.py              A, B, R operators + topology declarations
  text_features.py          M: observable text feature extraction
  per_turn_analysis.py      Incremental per-turn Γ pipeline
  corpus.py                 Synthetic test corpus (5 categories)
validation/
  run_validation.py         ShareGPT corpus validation (200 conversations)
  sharegpt_ingest.py        ShareGPT JSON → ABR turn format adapter
  data/sharegpt/            Place sg_90k_part1.json here
  output/                   JSON results
docs/
  technical_brief.md        Technical overview for ML/data engineers
  investor_brief.md         Investor-facing capability summary
  abr_primer.md             ABR framework primer
```

---

## Engagement

For partnership, licensing, or Phase 0 assessment inquiries:
Robin Macomber — relationalrelativity.dev

---

## References

- Macomber, R. & Stephenson, B. (2026). Convergent Discovery of Critical Phenomena Mathematics Across Disciplines. arXiv:2601.22389
- ShareGPT52K: huggingface.co/datasets/RyokoAI/ShareGPT52K (CC BY-NC 4.0)

---

*All definitions bounded over D. No claim beyond D.*
*Metatron Dynamics, Inc. — Delaware C-Corp #10551645*
