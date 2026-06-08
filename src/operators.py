#!/usr/bin/env python3
"""
operators.py — ABR Operators for Conversation Edge Fields
ABR Alignment Monitor
Metatron Dynamics, Inc.

Operators A, B, R on a 3-topology edge field over conversation
observables. Dict-based implementation matching Phase 2.1 of the
magnetosphere comparison — same structural pattern, different
observable index set.

Three topologies:
  T_P — Position topology: sentence adjacency within turns
  T_F — Feature topology: declared adjacency on text features
  T_tau — Turn topology: directed turn sequence (both participants)

Operator A: NodeField → EdgeField (3-topology)
Operator B: EdgeField → EdgeField (accumulation within each topology)
Operator R: EdgeField → EdgeField (cross-topology circulation)

DECLARATION:
  No semantic interpretation. No embedding. No model access.
  Operators act on bounded measurable text features extracted by M.
  Topology is declared on the observable index set after M,
  not on presumed entities prior to M.
"""

import logging
from typing import Optional

import numpy as np

from text_features import (
    ConversationFeatures,
    TurnFeatures,
    N_FEATURES,
    FEATURE_NAMES,
)

log = logging.getLogger("operators")

# ===================================================================
# Feature Topology Declaration
# ===================================================================

def declare_feature_adjacency_ring() -> list[tuple[int, int]]:
    """
    INADMISSIBLE IN APPLICATION LAYER.

    Ring topology is the V3 proof topology used in Object Error
    theorems (Theorems 5-6). It is NOT an admissible declared
    topology for application-layer operator instances.

    Under V4 kernel discipline, every application topology must
    be declared by Origin on the observable index set after M.
    Ring topology is a mathematical convenience for proof
    structure, not a declaration about observable structure.

    Importing ring topology into an application instance
    constitutes undeclared topology — a V4 admissibility
    violation.

    Reference: Topological Admissibility criterion, two gates:
      Gate 1 — Evaluability: topology must be declared on D
      Gate 2 — Provenance: topology must not import structure
                           not present in the observable index set

    Ring fails Gate 2: mod N wrap-around has no observable
    referent in the conversation feature index set.

    DO NOT CALL THIS FUNCTION IN APPLICATION CODE.
    It is retained here as a documented inadmissibility record
    only. Any call in application context raises ValueError.
    """
    raise ValueError(
        "declare_feature_adjacency_ring() is inadmissible in "
        "application layer. Ring topology is the V3 proof topology "
        "(Object Error Theorems 5-6) and must not be used as a "
        "declared application topology. Use "
        "declare_feature_adjacency_functional() or declare your "
        "own topology on the observable index set after M. "
        "See V4 Topological Admissibility criterion."
    )


def declare_feature_adjacency_functional() -> list[tuple[int, int]]:
    """
    Declared feature adjacency based on observed functional
    co-variation hypotheses. Not semantic structure — declared
    structural commitment subject to empirical validation.

    Adjacency pairs are hypotheses about which features
    co-vary under relational change. The specific adjacency
    declaration is an open condition (build plan §7.1).

    sentence_length (0) — mean_word_length (7)
    type_token_ratio (1) — repetition_index (5)
    hedge_density (2) — assertion_density (3)
    question_density (4) — assertion_density (3)
    structural_markers (6) — sentence_length (0)

    Plus ring connections for full connectivity.
    """
    pairs = set()
    # Functional connections
    functional = [
        (0, 7),  # sentence_length — mean_word_length
        (1, 5),  # type_token_ratio — repetition_index
        (2, 3),  # hedge_density — assertion_density
        (3, 4),  # assertion_density — question_density
        (0, 6),  # sentence_length — structural_markers
    ]
    for p in functional:
        pairs.add(p)

    # Ring for connectivity
    for i in range(N_FEATURES):
        j = (i + 1) % N_FEATURES
        pairs.add((i, j))

    return sorted(pairs)


def declare_feature_adjacency_allpairs() -> list[tuple[int, int]]:
    """
    All-pairs adjacency on features. Matches magnetosphere
    component topology (all-pairs on {N, E, Z}).
    """
    pairs = []
    for i in range(N_FEATURES):
        for j in range(i + 1, N_FEATURES):
            pairs.append((i, j))
    return pairs


# Default: functional adjacency
DEFAULT_FEATURE_ADJACENCY = declare_feature_adjacency_functional


# ===================================================================
# Operator A — 3-Topology Edge Extraction
# ===================================================================

def operator_a(
    cf: ConversationFeatures,
    feature_pairs: list[tuple[int, int]],
) -> tuple[dict, dict, dict]:
    """
    A : NodeField → 3-topology EdgeField

    Extracts directed pairwise differences over all three
    topologies simultaneously.

    Returns
    -------
    position_ef : dict
        Key: (turn_idx, sent_i, sent_j, feat_idx) → float
        Directed position edges within each turn.
        Both directions: (i,j) and (j,i).

    feature_ef : dict
        Key: (turn_idx, sent_idx, feat_a, feat_b) → float
        Feature edges at each sentence position.

    turn_ef : dict
        Key: (turn_t, turn_t1, feat_idx) → float
        Directed turn edges between adjacent turns.
        Computed on turn means. Both directions.
    """
    position_ef = {}
    feature_ef = {}
    turn_ef = {}

    n_turns = cf.n_turns

    for turn in cf.turns:
        t = turn.turn_index
        n_sent = turn.n_sentences

        # --- Position edges: adjacent sentences within this turn ---
        for i in range(n_sent):
            for j in [i - 1, i + 1]:
                if 0 <= j < n_sent:
                    for f in range(N_FEATURES):
                        val = float(turn.features[i, f] - turn.features[j, f])
                        position_ef[(t, i, j, f)] = val

        # --- Feature edges: between adjacent features at each position ---
        for i in range(n_sent):
            for (fa, fb) in feature_pairs:
                val = float(turn.features[i, fa] - turn.features[i, fb])
                feature_ef[(t, i, fa, fb)] = val

    # --- Turn edges: between adjacent turns on turn means ---
    for t in range(n_turns):
        for t2 in [t - 1, t + 1]:
            if 0 <= t2 < n_turns:
                # Only forward for temporal topology? No — both directions
                # for spatial analog. Turn topology is directed (forward only)
                # in the magnetosphere. Here we use directed forward only.
                pass

    # Rewrite: forward-only turn edges matching magnetosphere temporal topology
    for t in range(n_turns - 1):
        t1 = t
        t2 = t + 1
        for f in range(N_FEATURES):
            val = float(cf.turn_means[t2, f] - cf.turn_means[t1, f])
            turn_ef[(t1, t2, f)] = val

    log.info(f"  A: {len(position_ef)} position, "
             f"{len(feature_ef)} feature, "
             f"{len(turn_ef)} turn edges")

    return position_ef, feature_ef, turn_ef


# ===================================================================
# Compute ρ from A output
# ===================================================================

def compute_rho(
    position_ef: dict,
    feature_ef: dict,
    turn_ef: dict,
    cf: ConversationFeatures,
    rho_base: float = 0.3,
) -> dict:
    """
    Per station-step ρ: max absolute edge value incident to
    (turn, sentence) across all three topologies.

    Key: (turn_idx, sent_idx) → float
    """
    rho = {}

    # Initialize all positions
    for turn in cf.turns:
        for i in range(turn.n_sentences):
            rho[(turn.turn_index, i)] = 0.0

    # Position edges
    for (t, i, j, f), val in position_ef.items():
        key = (t, i)
        rho[key] = max(rho.get(key, 0.0), abs(val))

    # Feature edges
    for (t, i, fa, fb), val in feature_ef.items():
        key = (t, i)
        rho[key] = max(rho.get(key, 0.0), abs(val))

    # Turn edges — incident to all sentences in both turns
    for (t1, t2, f), val in turn_ef.items():
        aval = abs(val)
        for turn in cf.turns:
            if turn.turn_index in (t1, t2):
                for i in range(turn.n_sentences):
                    key = (turn.turn_index, i)
                    rho[key] = max(rho.get(key, 0.0), aval)

    # Apply bounding: ρ_base * m / (1 + m)
    for key in rho:
        m = rho[key]
        rho[key] = rho_base * m / (1.0 + m)

    return rho


# ===================================================================
# Operator B — 3-Topology Accumulation
# ===================================================================

def operator_b(
    position_ef: dict,
    feature_ef: dict,
    turn_ef: dict,
    cf: ConversationFeatures,
    feature_pairs: list[tuple[int, int]],
) -> tuple[dict, dict, dict]:
    """
    B : EdgeField → EdgeField

    Accumulates each edge with edges sharing its forward vertex
    within the same topology. No cross-topology coupling.
    """

    # --- Position B: accumulate with adjacent position edges at forward vertex ---
    b_position = {}
    for (t, i, j, f), val in position_ef.items():
        acc = val
        # Forward vertex is j. Accumulate with edges leaving j at same turn, same feature.
        for (t2, i2, j2, f2), val2 in position_ef.items():
            if t2 == t and i2 == j and f2 == f and j2 != i:
                acc += val2
        b_position[(t, i, j, f)] = acc

    # --- Feature B: accumulate with feature edges at adjacent positions ---
    b_feature = {}
    for (t, i, fa, fb), val in feature_ef.items():
        acc = val
        # Accumulate with same feature pair at adjacent sentence positions
        turn = cf.turns[t]
        for di in [-1, 1]:
            ni = i + di
            if 0 <= ni < turn.n_sentences:
                key_nb = (t, ni, fa, fb)
                if key_nb in feature_ef:
                    acc += feature_ef[key_nb]
        b_feature[(t, i, fa, fb)] = acc

    # --- Turn B: accumulate with next temporal edge (forward only) ---
    b_turn = {}
    for (t1, t2, f), val in turn_ef.items():
        acc = val
        # Forward temporal neighbor: (t2, t2+1, f)
        next_key = (t2, t2 + 1, f)
        if next_key in turn_ef:
            acc += turn_ef[next_key]
        b_turn[(t1, t2, f)] = acc

    log.info(f"  B: {len(b_position)} position, "
             f"{len(b_feature)} feature, "
             f"{len(b_turn)} turn edges")

    return b_position, b_feature, b_turn


# ===================================================================
# Operator R — Cross-Topology Circulation
# ===================================================================

def operator_r(
    b_position: dict,
    b_feature: dict,
    b_turn: dict,
    rho: dict,
    cf: ConversationFeatures,
    feature_pairs: list[tuple[int, int]],
) -> tuple[dict, dict, dict]:
    """
    R : EdgeField → EdgeField

    Cross-couples between the three topologies antisymmetrically.

    Position edges receive turn asymmetry:
      How is the turn-level evolution different at the two ends
      of this position edge?

    Feature edges receive position asymmetry:
      How does the position gradient structure differ between
      the two features?

    Turn edges receive feature asymmetry:
      How is the feature relationship structure changing
      between the two turns?

    Circulation cycle:
      T_P → T_tau → T_F → T_P
    """

    r_position = {}
    r_feature = {}
    r_turn = {}

    # --- Position edges receive turn asymmetry ---
    for (t, i, j, f), bv in b_position.items():
        rv = bv
        rh = rho.get((t, i), 0.0)

        if rh > 0.0:
            # Turn edge at forward vertex's turn vs current turn
            # For within-turn position edges, both endpoints share
            # the same turn. Use turn edges incident to this turn.
            t_fwd = (t, t + 1, f)
            t_bwd = (t - 1, t, f)

            te_fwd = b_turn.get(t_fwd)
            te_bwd = b_turn.get(t_bwd)

            if te_fwd is not None and te_bwd is not None:
                turn_asym = te_fwd - te_bwd
                rv += rh * turn_asym
            elif te_fwd is not None:
                rv += rh * te_fwd
            elif te_bwd is not None:
                rv -= rh * te_bwd

        r_position[(t, i, j, f)] = rv

    # --- Feature edges receive position asymmetry ---
    for (t, i, fa, fb), bv in b_feature.items():
        rv = bv
        rh = rho.get((t, i), 0.0)

        if rh > 0.0:
            # Position gradient for feature fa vs feature fb
            # at this sentence position
            pos_asym_vals = []
            for (t2, i2, j2, f2), pv in b_position.items():
                if t2 == t and i2 == i:
                    if f2 == fa:
                        # Find matching edge for fb
                        key_fb = (t, i, j2, fb)
                        if key_fb in b_position:
                            pos_asym_vals.append(pv - b_position[key_fb])

            if pos_asym_vals:
                pos_asym = float(np.mean(pos_asym_vals))
                rv += rh * pos_asym

        r_feature[(t, i, fa, fb)] = rv

    # --- Turn edges receive feature asymmetry ---
    for (t1, t2, f), bv in b_turn.items():
        rv = bv
        # Use rho at midpoint — average across sentences in t1
        turn_t1 = cf.turns[t1]
        rh_vals = [rho.get((t1, i), 0.0) for i in range(turn_t1.n_sentences)]
        rh = float(np.mean(rh_vals)) if rh_vals else 0.0

        if rh > 0.0:
            # Feature asymmetry at t2 vs t1
            # For each feature pair containing f, compute
            # the feature edge difference between turns
            for (fa, fb) in feature_pairs:
                if fa == f or fb == f:
                    # Mean feature edge at t2 vs t1
                    fe_t2_vals = [v for (tt, ii, ffa, ffb), v in b_feature.items()
                                  if tt == t2 and ffa == fa and ffb == fb]
                    fe_t1_vals = [v for (tt, ii, ffa, ffb), v in b_feature.items()
                                  if tt == t1 and ffa == fa and ffb == fb]

                    if fe_t2_vals and fe_t1_vals:
                        feat_asym = float(np.mean(fe_t2_vals) - np.mean(fe_t1_vals))
                        if fa == f:
                            rv += rh * feat_asym
                        else:
                            rv -= rh * feat_asym

        r_turn[(t1, t2, f)] = rv

    log.info(f"  R: {len(r_position)} position, "
             f"{len(r_feature)} feature, "
             f"{len(r_turn)} turn edges")

    return r_position, r_feature, r_turn


# ===================================================================
# Diagnostics — σ² and Γ
# ===================================================================

def sigma_sq_scalar(ef: dict) -> float:
    """σ² over all scalar-valued edges."""
    vals = list(ef.values())
    if not vals:
        return 0.0
    return float(np.var(vals))


def compute_gamma(
    r_position: dict, r_feature: dict, r_turn: dict,
    b_position: dict, b_feature: dict, b_turn: dict,
) -> dict:
    """
    Γ = σ²(R∘B∘A) − σ²(B∘A), total and per topology.
    """
    s2_r_pos = sigma_sq_scalar(r_position)
    s2_r_feat = sigma_sq_scalar(r_feature)
    s2_r_turn = sigma_sq_scalar(r_turn)
    s2_r_total = s2_r_pos + s2_r_feat + s2_r_turn

    s2_ba_pos = sigma_sq_scalar(b_position)
    s2_ba_feat = sigma_sq_scalar(b_feature)
    s2_ba_turn = sigma_sq_scalar(b_turn)
    s2_ba_total = s2_ba_pos + s2_ba_feat + s2_ba_turn

    gamma_pos = s2_r_pos - s2_ba_pos
    gamma_feat = s2_r_feat - s2_ba_feat
    gamma_turn = s2_r_turn - s2_ba_turn
    gamma_total = s2_r_total - s2_ba_total

    return {
        "gamma_total": gamma_total,
        "gamma_position": gamma_pos,
        "gamma_feature": gamma_feat,
        "gamma_turn": gamma_turn,
        "sigma_r_total": s2_r_total,
        "sigma_ba_total": s2_ba_total,
        "sigma_r_position": s2_r_pos,
        "sigma_r_feature": s2_r_feat,
        "sigma_r_turn": s2_r_turn,
        "sigma_ba_position": s2_ba_pos,
        "sigma_ba_feature": s2_ba_feat,
        "sigma_ba_turn": s2_ba_turn,
    }


def compute_p_gamma(diag: dict) -> dict:
    """
    Circulation-state projection P_Γ.

    Returns sign structure and ratio decomposition.
    """
    g_p = diag["gamma_position"]
    g_f = diag["gamma_feature"]
    g_t = diag["gamma_turn"]
    g_total = diag["gamma_total"]

    if abs(g_total) > 0:
        ratio_p = g_p / g_total
        ratio_f = g_f / g_total
        ratio_t = g_t / g_total
    else:
        ratio_p = ratio_f = ratio_t = 0.0

    return {
        "sgn_position": int(np.sign(g_p)),
        "sgn_feature": int(np.sign(g_f)),
        "sgn_turn": int(np.sign(g_t)),
        "ratio_position": ratio_p,
        "ratio_feature": ratio_f,
        "ratio_turn": ratio_t,
    }


# ===================================================================
# Full Pipeline: A → B → R → Γ
# ===================================================================

def run_abr(
    cf: ConversationFeatures,
    feature_adjacency: Optional[list[tuple[int, int]]] = None,
    rho_base: float = 0.3,
) -> dict:
    """
    Run full ABR pipeline on a conversation.

    Returns dict with Γ decomposition, P_Γ, and edge counts.
    """
    if feature_adjacency is None:
        feature_adjacency = DEFAULT_FEATURE_ADJACENCY()

    log.info("Running operator A...")
    pos_ef, feat_ef, turn_ef = operator_a(cf, feature_adjacency)

    log.info("Computing ρ...")
    rho = compute_rho(pos_ef, feat_ef, turn_ef, cf, rho_base)

    log.info("Running operator B...")
    b_pos, b_feat, b_turn = operator_b(
        pos_ef, feat_ef, turn_ef, cf, feature_adjacency
    )

    log.info("Running operator R...")
    r_pos, r_feat, r_turn = operator_r(
        b_pos, b_feat, b_turn, rho, cf, feature_adjacency
    )

    log.info("Computing Γ...")
    diag = compute_gamma(r_pos, r_feat, r_turn, b_pos, b_feat, b_turn)
    p_gamma = compute_p_gamma(diag)

    return {
        "diagnostics": diag,
        "p_gamma": p_gamma,
        "edge_counts": {
            "position": len(pos_ef),
            "feature": len(feat_ef),
            "turn": len(turn_ef),
        },
        "parameters": {
            "rho_base": rho_base,
            "n_feature_pairs": len(feature_adjacency),
            "feature_adjacency_type": "functional",
        },
    }


# ===================================================================
# Self-Test
# ===================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

    from text_features import extract_conversation_features

    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # Same test conversation as text_features.py
    test_conversation = [
        {
            "participant": "human",
            "text": "Can you help me understand how transformers work? I've been reading about attention mechanisms but I'm a bit confused about the multi-head part.",
        },
        {
            "participant": "model",
            "text": "Multi-head attention runs several attention computations in parallel, each with different learned projections. Each head can attend to different aspects of the input. The outputs are concatenated and projected back to the model dimension. This allows the model to capture different types of relationships simultaneously.",
        },
        {
            "participant": "human",
            "text": "That makes sense. But why do we need multiple heads? Couldn't one head learn all the patterns?",
        },
        {
            "participant": "model",
            "text": "A single head computes one set of attention weights, which means it can only represent one relationship pattern per position. Multiple heads let the model simultaneously track syntactic dependencies, semantic relationships, and positional patterns. It's similar to having multiple sensor types at the same location, each measuring a different quantity.",
        },
        {
            "participant": "human",
            "text": "I think I understand now. So each head is like a different measurement of the same input?",
        },
        {
            "participant": "model",
            "text": "Yes, that's a reasonable analogy. Each head projects the input into a different subspace and computes attention there. The information each head captures is determined by the learned projection matrices, not by any predetermined function.",
        },
    ]

    print("=" * 60)
    print("  ABR Alignment Monitor — Operator Test")
    print("=" * 60)

    # Extract features
    cf = extract_conversation_features(test_conversation)
    print(f"\nConversation: {cf.n_turns} turns, {cf.total_sentences} sentences")

    # Run ABR
    result = run_abr(cf)
    diag = result["diagnostics"]
    pg = result["p_gamma"]

    print(f"\n--- Γ Decomposition ---")
    print(f"  Γ_total:    {diag['gamma_total']:.6f}")
    print(f"  Γ_position: {diag['gamma_position']:.6f}")
    print(f"  Γ_feature:  {diag['gamma_feature']:.6f}")
    print(f"  Γ_turn:     {diag['gamma_turn']:.6f}")
    print(f"  σ²(E):      {diag['sigma_r_total']:.6f}")
    print(f"  σ²(BA):     {diag['sigma_ba_total']:.6f}")

    print(f"\n--- P_Γ (Circulation State) ---")
    print(f"  sgn:   pos={pg['sgn_position']:+d}, "
          f"feat={pg['sgn_feature']:+d}, turn={pg['sgn_turn']:+d}")
    print(f"  ratio: pos={pg['ratio_position']:.3f}, "
          f"feat={pg['ratio_feature']:.3f}, turn={pg['ratio_turn']:.3f}")

    print(f"\n--- Edge Counts ---")
    ec = result["edge_counts"]
    print(f"  Position: {ec['position']}")
    print(f"  Feature:  {ec['feature']}")
    print(f"  Turn:     {ec['turn']}")

    # Verify Γ > 0 on structured conversation
    if diag["gamma_total"] > 0:
        print(f"\n  ✓ Γ > 0 on structured conversation")
    else:
        print(f"\n  ✗ Γ ≤ 0 — investigate")

    print("\n" + "=" * 60)
    print("  Operator test complete.")
    print("=" * 60)
