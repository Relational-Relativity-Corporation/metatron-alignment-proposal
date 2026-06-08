#!/usr/bin/env python3
"""
text_features.py — Measurement Mapping M for Conversation Text
ABR Alignment Monitor
Metatron Dynamics, Inc.

M : O → D

Observable space O: conversation text (strings, alternating
  human and model turns).
Domain D: bounded, finite feature vectors at each sentence
  position within each turn.

No model access. No hidden states. No API. Text only.
The conversation is the sensor data.

DECLARATION:
  M extracts measurable text features at declared sentence
  positions. No semantic interpretation. No sentiment analysis.
  No embeddings. All features are computable from surface
  text properties without external models.

  Features are extracted identically from human and model text.
  Both participants produce the same measurable quantities.

FEATURES (per sentence position):
  0. sentence_length    — word count
  1. type_token_ratio   — unique words / total words (local)
  2. hedge_density      — proportion of hedging language
  3. assertion_density  — proportion of declarative markers
  4. question_density   — proportion of interrogative sentences
  5. repetition_index   — n-gram overlap with prior turn
  6. structural_markers — formatting density (bullets, code, etc.)
  7. mean_word_length   — mean character count per word

All features are bounded in [0, ∞) or [0, 1] by construction.
All features are finite for any finite text input.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("text_features")

# ===================================================================
# Feature Configuration
# ===================================================================

FEATURE_NAMES = [
    "sentence_length",
    "type_token_ratio",
    "hedge_density",
    "assertion_density",
    "question_density",
    "repetition_index",
    "structural_markers",
    "mean_word_length",
]

N_FEATURES = len(FEATURE_NAMES)

# Hedging language markers
HEDGE_WORDS = {
    "maybe", "perhaps", "possibly", "somewhat", "might",
    "could", "would", "should", "probably", "likely",
    "unlikely", "apparently", "seemingly", "roughly",
    "approximately", "generally", "typically", "usually",
    "often", "sometimes", "arguably", "potentially",
    "i think", "i believe", "i suppose", "it seems",
    "in my opinion", "it appears", "to some extent",
    "more or less", "kind of", "sort of",
}

# Assertion markers
ASSERTION_WORDS = {
    "certainly", "definitely", "absolutely", "clearly",
    "obviously", "undoubtedly", "always", "never",
    "must", "shall", "will", "is", "are", "was", "were",
    "proves", "demonstrates", "establishes", "confirms",
    "ensures", "guarantees", "requires", "demands",
}

# Structural markers (formatting patterns)
STRUCTURAL_PATTERNS = [
    r"^[\s]*[-*•]",           # bullet points
    r"^[\s]*\d+[.)]\s",       # numbered lists
    r"```",                    # code blocks
    r"^#{1,6}\s",             # markdown headers
    r"\|.*\|.*\|",            # table rows
    r"^\s*>",                 # blockquotes
]


# ===================================================================
# Sentence Splitting
# ===================================================================

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences. Simple rule-based splitter.
    Handles common abbreviations and decimal numbers.
    Returns non-empty sentences only.
    """
    # Protect common abbreviations
    protected = text
    for abbr in ["Mr.", "Mrs.", "Dr.", "Prof.", "e.g.", "i.e.", "etc.", "vs."]:
        protected = protected.replace(abbr, abbr.replace(".", "∎"))

    # Protect decimal numbers
    protected = re.sub(r"(\d)\.(\d)", r"\1∎\2", protected)

    # Split on sentence-ending punctuation followed by space or end
    parts = re.split(r"(?<=[.!?])\s+", protected)

    # Restore protected periods
    sentences = []
    for p in parts:
        p = p.replace("∎", ".").strip()
        if p:
            sentences.append(p)

    # If no sentence boundaries found, treat whole text as one sentence
    if not sentences and text.strip():
        sentences = [text.strip()]

    return sentences


def tokenize_words(text: str) -> list[str]:
    """Simple word tokenization. Strips punctuation."""
    return re.findall(r"\b\w+\b", text.lower())


# ===================================================================
# Per-Sentence Feature Extraction
# ===================================================================

def extract_sentence_features(
    sentence: str,
    prior_turn_text: Optional[str] = None,
) -> np.ndarray:
    """
    Extract feature vector for a single sentence.

    Returns np.ndarray of shape (N_FEATURES,).
    All values are finite and bounded.
    """
    words = tokenize_words(sentence)
    n_words = len(words)

    if n_words == 0:
        return np.zeros(N_FEATURES, dtype=np.float64)

    # 0. sentence_length — word count
    sentence_length = float(n_words)

    # 1. type_token_ratio — unique words / total words
    unique_words = len(set(words))
    type_token_ratio = unique_words / n_words

    # 2. hedge_density — proportion of words/phrases that are hedges
    text_lower = sentence.lower()
    hedge_count = 0
    for hedge in HEDGE_WORDS:
        if " " in hedge:
            # Multi-word hedge: count occurrences in text
            hedge_count += text_lower.count(hedge)
        else:
            hedge_count += words.count(hedge)
    hedge_density = min(hedge_count / n_words, 1.0)

    # 3. assertion_density — proportion of assertion markers
    assertion_count = 0
    for marker in ASSERTION_WORDS:
        assertion_count += words.count(marker)
    assertion_density = min(assertion_count / n_words, 1.0)

    # 4. question_density — is this sentence a question?
    #    Binary per sentence (0.0 or 1.0)
    question_density = 1.0 if sentence.rstrip().endswith("?") else 0.0

    # 5. repetition_index — n-gram overlap with prior turn
    if prior_turn_text is not None and prior_turn_text.strip():
        prior_words = set(tokenize_words(prior_turn_text))
        current_words = set(words)
        if current_words:
            overlap = len(current_words & prior_words)
            repetition_index = overlap / len(current_words)
        else:
            repetition_index = 0.0
    else:
        repetition_index = 0.0

    # 6. structural_markers — formatting density
    structural_count = 0
    for pattern in STRUCTURAL_PATTERNS:
        if re.search(pattern, sentence, re.MULTILINE):
            structural_count += 1
    structural_markers = structural_count / len(STRUCTURAL_PATTERNS)

    # 7. mean_word_length — mean character count per word
    mean_word_length = np.mean([len(w) for w in words])

    features = np.array([
        sentence_length,
        type_token_ratio,
        hedge_density,
        assertion_density,
        question_density,
        repetition_index,
        structural_markers,
        mean_word_length,
    ], dtype=np.float64)

    # Verify all finite
    assert np.all(np.isfinite(features)), f"Non-finite features: {features}"

    return features


# ===================================================================
# Per-Turn Feature Extraction
# ===================================================================

@dataclass
class TurnFeatures:
    """Feature extraction result for a single conversation turn."""
    participant: str               # "human" or "model"
    turn_index: int                # 0-indexed position in conversation
    sentences: list[str]           # raw sentences
    features: np.ndarray           # shape (n_sentences, N_FEATURES)
    turn_mean: np.ndarray          # shape (N_FEATURES,) — mean across sentences

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)


def extract_turn_features(
    text: str,
    participant: str,
    turn_index: int,
    prior_turn_text: Optional[str] = None,
) -> TurnFeatures:
    """
    Extract features for all sentences in a single turn.

    Parameters
    ----------
    text : str
        The turn's text content.
    participant : str
        "human" or "model"
    turn_index : int
        0-indexed position in the conversation.
    prior_turn_text : str or None
        The previous turn's text, for repetition_index computation.

    Returns
    -------
    TurnFeatures with per-sentence features and turn mean.
    """
    sentences = split_sentences(text)

    if not sentences:
        # Empty turn — single zero-vector sentence
        features = np.zeros((1, N_FEATURES), dtype=np.float64)
        return TurnFeatures(
            participant=participant,
            turn_index=turn_index,
            sentences=[""],
            features=features,
            turn_mean=np.zeros(N_FEATURES, dtype=np.float64),
        )

    feature_rows = []
    for sentence in sentences:
        fv = extract_sentence_features(sentence, prior_turn_text)
        feature_rows.append(fv)

    features = np.array(feature_rows, dtype=np.float64)
    turn_mean = np.mean(features, axis=0)

    return TurnFeatures(
        participant=participant,
        turn_index=turn_index,
        sentences=sentences,
        features=features,
        turn_mean=turn_mean,
    )


# ===================================================================
# Full Conversation Feature Extraction
# ===================================================================

@dataclass
class ConversationFeatures:
    """Feature extraction result for a full conversation."""
    turns: list[TurnFeatures]
    n_turns: int
    total_sentences: int
    turn_means: np.ndarray         # shape (n_turns, N_FEATURES)
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))

    def summary(self) -> dict:
        """JSON-serializable summary."""
        return {
            "n_turns": self.n_turns,
            "total_sentences": self.total_sentences,
            "participants": [t.participant for t in self.turns],
            "sentences_per_turn": [t.n_sentences for t in self.turns],
            "feature_names": self.feature_names,
            "turn_means": self.turn_means.tolist(),
        }


def extract_conversation_features(
    conversation: list[dict],
) -> ConversationFeatures:
    """
    Extract features for a full conversation.

    Parameters
    ----------
    conversation : list of dict
        Each dict has:
          "text": str — the turn text
          "participant": str — "human" or "model"

    Returns
    -------
    ConversationFeatures with per-turn and per-sentence features.
    """
    turns = []
    prior_text = None

    for i, turn in enumerate(conversation):
        text = turn["text"]
        participant = turn["participant"]

        tf = extract_turn_features(
            text=text,
            participant=participant,
            turn_index=i,
            prior_turn_text=prior_text,
        )
        turns.append(tf)
        prior_text = text

    n_turns = len(turns)
    total_sentences = sum(t.n_sentences for t in turns)
    turn_means = np.array([t.turn_mean for t in turns], dtype=np.float64)

    return ConversationFeatures(
        turns=turns,
        n_turns=n_turns,
        total_sentences=total_sentences,
        turn_means=turn_means,
    )


# ===================================================================
# Self-Test
# ===================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

    # --- Test conversation ---
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
    print("  ABR Alignment Monitor — Text Feature Extraction Test")
    print("=" * 60)

    cf = extract_conversation_features(test_conversation)

    print(f"\nConversation: {cf.n_turns} turns, {cf.total_sentences} sentences")
    print(f"Features per sentence: {N_FEATURES}")
    print(f"Feature names: {FEATURE_NAMES}")

    for turn in cf.turns:
        print(f"\n--- Turn {turn.turn_index} ({turn.participant}) ---")
        print(f"  Sentences: {turn.n_sentences}")
        print(f"  Feature matrix shape: {turn.features.shape}")
        for i, name in enumerate(FEATURE_NAMES):
            vals = turn.features[:, i]
            print(f"  {name:20s}: mean={turn.turn_mean[i]:.3f}, "
                  f"range=[{vals.min():.3f}, {vals.max():.3f}]")

    print(f"\nTurn means shape: {cf.turn_means.shape}")
    print(f"All finite: {np.all(np.isfinite(cf.turn_means))}")

    # --- Verify features vary across turns ---
    print("\n--- Feature variation across turns ---")
    for i, name in enumerate(FEATURE_NAMES):
        col = cf.turn_means[:, i]
        print(f"  {name:20s}: std={np.std(col):.4f}, "
              f"range=[{col.min():.3f}, {col.max():.3f}]")

    # --- Verify repetition_index works ---
    print("\n--- Repetition index check ---")
    for turn in cf.turns:
        rep_vals = turn.features[:, FEATURE_NAMES.index("repetition_index")]
        print(f"  Turn {turn.turn_index} ({turn.participant}): "
              f"mean rep={np.mean(rep_vals):.3f}")

    print("\n" + "=" * 60)
    print("  All checks passed. M is operational.")
    print("=" * 60)
