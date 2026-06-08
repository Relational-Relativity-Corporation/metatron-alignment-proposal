#!/usr/bin/env python3
"""
per_turn_analysis.py — Per-Turn Incremental Γ Pipeline
ABR Alignment Monitor
Metatron Dynamics, Inc.

Computes Γ and P_Γ incrementally as each turn arrives.
At each turn t, the edge field includes all turns 0..t.
The departure ΔP_Γ(t) measures how the circulation state
changes with each new turn.

Participant attribution: at human turns, Γ_τ is driven
by the human's relational departure from the model's prior
response. At model turns, Γ_τ is driven by the model's
relational departure from the human's prior input.

DECLARATION:
  Per-turn Γ is a different declared projection from
  whole-conversation Γ. It answers: "how is the relational
  structure evolving?" rather than "what is the total
  relational coupling?" Both are valid projections of the
  same edge field.

  drift := P_Γ(t) ≉ P_Γ,0
  The meaning of any departure is application-dependent,
  not operator-native.
"""

import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass

import numpy as np

SRC_DIR = Path(__file__).parent
sys.path.insert(0, str(SRC_DIR))

from text_features import (
    extract_conversation_features,
    extract_turn_features,
    ConversationFeatures,
    TurnFeatures,
    FEATURE_NAMES,
    N_FEATURES,
)
from operators import (
    operator_a,
    compute_rho,
    operator_b,
    operator_r,
    compute_gamma,
    compute_p_gamma,
    declare_feature_adjacency_functional,
    run_abr,
)
from corpus import make_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("per_turn")

REPO_ROOT = SRC_DIR.parent
OUTPUT_DIR = REPO_ROOT / "output"
FIGURES_DIR = REPO_ROOT / "figures"


# ===================================================================
# Per-Turn Incremental Analysis
# ===================================================================

@dataclass
class TurnResult:
    """Results for a single turn in the incremental analysis."""
    turn_index: int
    participant: str
    n_turns_so_far: int
    gamma_total: float
    gamma_position: float
    gamma_feature: float
    gamma_turn: float
    p_gamma: dict
    delta_p_gamma: float  # ||P_Γ(t) - P_Γ(t-1)||
    delta_turn_ratio: float  # change in turn ratio
    sigma_r: float
    sigma_ba: float


def p_gamma_distance(pg1: dict, pg2: dict) -> float:
    """Euclidean distance between two P_Γ vectors (ratios only)."""
    v1 = np.array([pg1["ratio_position"], pg1["ratio_feature"], pg1["ratio_turn"]])
    v2 = np.array([pg2["ratio_position"], pg2["ratio_feature"], pg2["ratio_turn"]])
    return float(np.linalg.norm(v1 - v2))


def run_incremental(
    conversation: list[dict],
    feature_adjacency: list[tuple[int, int]] = None,
    rho_base: float = 0.3,
) -> list[TurnResult]:
    """
    Run ABR incrementally, adding one turn at a time.

    At each turn t, computes Γ and P_Γ using the conversation
    through turn t. Returns a TurnResult for each turn from
    turn 1 onward (turn 0 alone has no turn edges).

    Parameters
    ----------
    conversation : list of dict
        Each dict: {"participant": str, "text": str}
    feature_adjacency : list of (int, int) or None
        Feature topology pairs. Default: functional adjacency.
    rho_base : float

    Returns
    -------
    list of TurnResult, one per turn from turn 1 onward.
    """
    if feature_adjacency is None:
        feature_adjacency = declare_feature_adjacency_functional()

    results = []
    prev_p_gamma = None

    for t in range(1, len(conversation)):
        # Build conversation through turn t
        partial = conversation[:t + 1]
        cf = extract_conversation_features(partial)

        # Run ABR on partial conversation
        abr = run_abr(cf, feature_adjacency, rho_base)
        diag = abr["diagnostics"]
        pg = compute_p_gamma(diag)

        # Compute ΔP_Γ
        if prev_p_gamma is not None:
            delta = p_gamma_distance(pg, prev_p_gamma)
            delta_turn = pg["ratio_turn"] - prev_p_gamma["ratio_turn"]
        else:
            delta = 0.0
            delta_turn = 0.0

        tr = TurnResult(
            turn_index=t,
            participant=conversation[t]["participant"],
            n_turns_so_far=t + 1,
            gamma_total=diag["gamma_total"],
            gamma_position=diag["gamma_position"],
            gamma_feature=diag["gamma_feature"],
            gamma_turn=diag["gamma_turn"],
            p_gamma=pg,
            delta_p_gamma=delta,
            delta_turn_ratio=delta_turn,
            sigma_r=diag["sigma_r_total"],
            sigma_ba=diag["sigma_ba_total"],
        )
        results.append(tr)
        prev_p_gamma = pg

    return results


# ===================================================================
# Corpus-Level Per-Turn Analysis
# ===================================================================

def analyze_corpus():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    corpus = make_corpus()
    feature_adj = declare_feature_adjacency_functional()

    all_results = {}

    for conv in corpus:
        conv_id = conv["id"]
        category = conv["category"]

        log.info(f"\n{'='*50}")
        log.info(f"  {conv_id} [{category}]")
        log.info(f"{'='*50}")

        turn_results = run_incremental(conv["turns"], feature_adj)

        log.info(f"\n  {'Turn':>4s} {'Who':>6s} {'Γ_total':>10s} "
                 f"{'r_turn':>7s} {'ΔP_Γ':>8s} {'Δr_turn':>8s}")
        log.info(f"  {'-'*4} {'-'*6} {'-'*10} {'-'*7} {'-'*8} {'-'*8}")

        for tr in turn_results:
            log.info(f"  {tr.turn_index:4d} {tr.participant:>6s} "
                     f"{tr.gamma_total:10.2f} {tr.p_gamma['ratio_turn']:7.3f} "
                     f"{tr.delta_p_gamma:8.4f} {tr.delta_turn_ratio:+8.4f}")

        all_results[conv_id] = {
            "category": category,
            "turn_results": turn_results,
        }

    # =================================================================
    # Category comparison: ΔP_Γ profiles
    # =================================================================

    log.info(f"\n{'='*60}")
    log.info(f"  ΔP_Γ SUMMARY BY CATEGORY")
    log.info(f"{'='*60}")

    categories = sorted(set(c["category"] for c in corpus))

    cat_delta_stats = {}
    for cat in categories:
        cat_convs = {k: v for k, v in all_results.items() if v["category"] == cat}
        all_deltas = []
        max_deltas = []
        late_deltas = []  # deltas in second half of conversation

        for conv_id, data in cat_convs.items():
            trs = data["turn_results"]
            deltas = [tr.delta_p_gamma for tr in trs]
            all_deltas.extend(deltas)
            max_deltas.append(max(deltas) if deltas else 0)

            # Late-conversation deltas (second half)
            mid = len(trs) // 2
            if mid > 0:
                late_deltas.extend([tr.delta_p_gamma for tr in trs[mid:]])

        stats = {
            "n_conversations": len(cat_convs),
            "mean_delta": float(np.mean(all_deltas)) if all_deltas else 0,
            "std_delta": float(np.std(all_deltas)) if all_deltas else 0,
            "max_delta": float(np.max(all_deltas)) if all_deltas else 0,
            "mean_max_delta": float(np.mean(max_deltas)) if max_deltas else 0,
            "mean_late_delta": float(np.mean(late_deltas)) if late_deltas else 0,
        }
        cat_delta_stats[cat] = stats

        log.info(f"\n  {cat} (n={stats['n_conversations']}):")
        log.info(f"    ΔP_Γ mean: {stats['mean_delta']:.4f}, "
                 f"std: {stats['std_delta']:.4f}, "
                 f"max: {stats['max_delta']:.4f}")
        log.info(f"    Mean peak ΔP_Γ per conversation: {stats['mean_max_delta']:.4f}")
        log.info(f"    Mean late-conversation ΔP_Γ: {stats['mean_late_delta']:.4f}")

    # =================================================================
    # Participant attribution
    # =================================================================

    log.info(f"\n{'='*60}")
    log.info(f"  PARTICIPANT ATTRIBUTION")
    log.info(f"{'='*60}")

    for cat in categories:
        cat_convs = {k: v for k, v in all_results.items() if v["category"] == cat}
        human_deltas = []
        model_deltas = []

        for conv_id, data in cat_convs.items():
            for tr in data["turn_results"]:
                if tr.participant == "human":
                    human_deltas.append(tr.delta_p_gamma)
                else:
                    model_deltas.append(tr.delta_p_gamma)

        h_mean = float(np.mean(human_deltas)) if human_deltas else 0
        m_mean = float(np.mean(model_deltas)) if model_deltas else 0

        log.info(f"\n  {cat}:")
        log.info(f"    Human turn mean ΔP_Γ: {h_mean:.4f} (n={len(human_deltas)})")
        log.info(f"    Model turn mean ΔP_Γ: {m_mean:.4f} (n={len(model_deltas)})")
        if h_mean > 0 and m_mean > 0:
            ratio = h_mean / m_mean
            log.info(f"    Human/Model ratio: {ratio:.2f}")

    # =================================================================
    # Save
    # =================================================================

    summary = {
        "analysis": "per_turn_incremental",
        "corpus_size": len(corpus),
        "category_delta_stats": cat_delta_stats,
        "conversations": {},
    }

    for conv_id, data in all_results.items():
        summary["conversations"][conv_id] = {
            "category": data["category"],
            "turns": [{
                "turn_index": tr.turn_index,
                "participant": tr.participant,
                "gamma_total": tr.gamma_total,
                "gamma_position": tr.gamma_position,
                "gamma_feature": tr.gamma_feature,
                "gamma_turn": tr.gamma_turn,
                "p_gamma": tr.p_gamma,
                "delta_p_gamma": tr.delta_p_gamma,
                "delta_turn_ratio": tr.delta_turn_ratio,
            } for tr in data["turn_results"]],
        }

    out_path = OUTPUT_DIR / "per_turn_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSaved: {out_path}")

    # =================================================================
    # Plot: ΔP_Γ trajectories by conversation
    # =================================================================

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {
            "benign": "steelblue",
            "escalation": "firebrick",
            "sudden_shift": "darkorange",
            "human_drift": "mediumpurple",
            "recovery": "seagreen",
        }

        # --- Figure 1: ΔP_Γ over turns for all conversations ---
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        ax = axes[0]
        for conv_id, data in all_results.items():
            cat = data["category"]
            trs = data["turn_results"]
            turns = [tr.turn_index for tr in trs]
            deltas = [tr.delta_p_gamma for tr in trs]
            label = conv_id if conv_id.endswith("01") else None
            ax.plot(turns, deltas, color=colors.get(cat, "gray"),
                    alpha=0.7, linewidth=1.5, label=label,
                    marker="o", markersize=3)

        ax.set_ylabel("ΔP_Γ")
        ax.set_xlabel("Turn index")
        ax.set_title("Per-Turn Circulation State Departure (ΔP_Γ)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

        # --- Figure 2: Turn ratio trajectory ---
        ax = axes[1]
        for conv_id, data in all_results.items():
            cat = data["category"]
            trs = data["turn_results"]
            turns = [tr.turn_index for tr in trs]
            ratios = [tr.p_gamma["ratio_turn"] for tr in trs]
            ax.plot(turns, ratios, color=colors.get(cat, "gray"),
                    alpha=0.7, linewidth=1.5,
                    marker="o", markersize=3)

        ax.set_ylabel("P_Γ turn ratio")
        ax.set_xlabel("Turn index")
        ax.set_title("Turn Topology Fraction Over Conversation")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = FIGURES_DIR / "per_turn_delta_pgamma.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved: {plot_path}")

        # --- Figure 3: Participant attribution by category ---
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))

        cat_list = sorted(categories)
        x = np.arange(len(cat_list))
        width = 0.35

        human_means = []
        model_means = []
        for cat in cat_list:
            cat_convs = {k: v for k, v in all_results.items() if v["category"] == cat}
            h_d = [tr.delta_p_gamma for cid, d in cat_convs.items()
                   for tr in d["turn_results"] if tr.participant == "human"]
            m_d = [tr.delta_p_gamma for cid, d in cat_convs.items()
                   for tr in d["turn_results"] if tr.participant == "model"]
            human_means.append(np.mean(h_d) if h_d else 0)
            model_means.append(np.mean(m_d) if m_d else 0)

        ax.bar(x - width/2, human_means, width, label="Human turns",
               color="mediumpurple", alpha=0.7)
        ax.bar(x + width/2, model_means, width, label="Model turns",
               color="steelblue", alpha=0.7)
        ax.set_ylabel("Mean ΔP_Γ")
        ax.set_xlabel("Category")
        ax.set_title("Participant Attribution: Who Drives ΔP_Γ?")
        ax.set_xticks(x)
        ax.set_xticklabels(cat_list, rotation=20)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()
        plot_path = FIGURES_DIR / "participant_attribution.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        log.info(f"Saved: {plot_path}")

    except ImportError:
        log.warning("matplotlib not available — plots skipped.")

    return all_results, summary


if __name__ == "__main__":
    all_results, summary = analyze_corpus()
    log.info("\nPer-turn analysis complete.")
