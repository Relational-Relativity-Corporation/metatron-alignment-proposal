#!/usr/bin/env python3
"""
demo.py — ABR Alignment Monitor: Live Detection Demo
Metatron Dynamics, Inc.

Runs the full ABR pipeline on two conversations from the built-in
corpus — one benign, one escalating — and prints the per-turn
ΔP_Γ signal side by side.

No external data. No model access. No API. Requires only numpy.

Usage:
    python demo.py

To run on your own conversation log:
    from demo import run_and_print
    run_and_print(your_turns, label="my_conversation")

Turn format: [{"participant": "human"|"model", "text": str}, ...]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from corpus import make_corpus
from per_turn_analysis import run_incremental

# ===================================================================
# Display
# ===================================================================

W = 70

def bar(val, max_val=0.4, width=24):
    filled = int(round(val / max_val * width)) if max_val > 0 else 0
    filled = min(filled, width)
    return "█" * filled + "░" * (width - filled)

def flag(val, warn=0.15, alert=0.30):
    if val >= alert: return "  ◄ ALERT"
    if val >= warn:  return "  ◄ watch"
    return ""

def run_and_print(turns, label="conversation", warn=0.15, alert=0.30):
    results = run_incremental(turns)
    peak    = max((r.delta_p_gamma for r in results), default=0.0)
    h_vals  = [r.delta_p_gamma for r in results if r.participant == "human"]
    m_vals  = [r.delta_p_gamma for r in results if r.participant == "model"]
    h_mean  = sum(h_vals) / len(h_vals) if h_vals else 0.0
    m_mean  = sum(m_vals) / len(m_vals) if m_vals else 0.0
    ratio   = h_mean / m_mean if m_mean > 0 else float("inf")

    print(f"\n{'='*W}")
    print(f"  {label}")
    print(f"  {len(turns)} turns | peak ΔP_Γ: {peak:.4f} | "
          f"human/model: {ratio:.2f}x")
    print(f"{'='*W}")
    print(f"  {'Turn':>4}  {'Who':>6}  {'ΔP_Γ':>7}  {'':24}  {'Γ_total':>10}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*7}  {'-'*24}  {'-'*10}")

    for r in results:
        who  = r.participant[:5]
        dv   = r.delta_p_gamma
        gv   = r.gamma_total
        b    = bar(dv)
        f    = flag(dv, warn, alert)
        print(f"  {r.turn_index:4d}  {who:>6}  {dv:7.4f}  {b}  {gv:10.3f}{f}")

    print(f"\n  Summary — human mean ΔP_Γ: {h_mean:.4f} | "
          f"model mean ΔP_Γ: {m_mean:.4f}")
    print(f"           peak ΔP_Γ: {peak:.4f} | "
          f"human/model ratio: {ratio:.2f}x")
    return results

# ===================================================================
# Main demo: benign vs escalation
# ===================================================================

def main():
    corpus = make_corpus()
    by_id  = {c["id"]: c for c in corpus}

    print("\n" + "=" * W)
    print("  ABR Alignment Monitor — Detection Demo")
    print("  Metatron Dynamics, Inc. | relationalrelativity.dev")
    print("=" * W)
    print("""
  What this measures: ΔP_Γ — the per-turn departure of a
  conversation's relational circulation state from its prior
  baseline. Computed from observable text features only.
  No model access. No embeddings. Deterministic.

  Topology: T_P (position) × T_F (feature) × T_τ (turn)
  Kernel:   E(x,ρ) = R(B(A(x)), ρ(A(x)))  [A→B→R, V4]
""")

    # --- Demo 1: benign ---
    benign = by_id["benign_01"]
    print(f'  Conversation 1: "{benign["description"]}"')
    run_and_print(benign["turns"], label="BENIGN — sorting algorithms")

    # --- Demo 2: escalation ---
    esc = by_id["escalation_01"]
    print(f'\n  Conversation 2: "{esc["description"]}"')
    run_and_print(esc["turns"], label="ESCALATION — coding help → auth bypass demand")

    # --- Demo 3: sudden shift ---
    shift = by_id["sudden_01"]
    print(f'\n  Conversation 3: "{shift["description"]}"')
    run_and_print(shift["turns"], label="SUDDEN SHIFT — weather → hostile attack")

    # --- Cross-conversation summary ---
    print(f"\n{'='*W}")
    print("  CROSS-CONVERSATION COMPARISON")
    print(f"{'='*W}")
    print(f"  {'Conversation':35}  {'Peak ΔP_Γ':>10}  {'H/M ratio':>10}")
    print(f"  {'-'*35}  {'-'*10}  {'-'*10}")

    rows = [
        ("benign_01",     "Benign (sorting)"),
        ("escalation_01", "Escalation (auth bypass)"),
        ("sudden_01",     "Sudden shift (hostile)"),
        ("recovery_01",   "Recovery (hostile→cooperative)"),
        ("human_drift_01","Human drift (terse/dismissive)"),
    ]
    for conv_id, label in rows:
        c = by_id[conv_id]
        rs     = run_incremental(c["turns"])
        peak   = max((r.delta_p_gamma for r in rs), default=0.0)
        h_v    = [r.delta_p_gamma for r in rs if r.participant == "human"]
        m_v    = [r.delta_p_gamma for r in rs if r.participant == "model"]
        h_mean = sum(h_v) / len(h_v) if h_v else 0.0
        m_mean = sum(m_v) / len(m_v) if m_v else 0.0
        ratio  = h_mean / m_mean if m_mean > 0 else float("inf")
        print(f"  {label:35}  {peak:10.4f}  {ratio:10.2f}x")

    print(f"\n  ΔP_Γ is deterministic: same conversation → same result.")
    print(f"  Proof of structural basis: arXiv:2601.22389")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
