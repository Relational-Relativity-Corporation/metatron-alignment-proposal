"""
Run ABR per-turn analysis on ShareGPT real conversation data.
Usage: python validation/run_validation.py [--sample N]
"""
import sys, json, argparse
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(Path(__file__).parent))

from sharegpt_ingest import load_all
from text_features import extract_conversation_features
from operators import run_abr, declare_feature_adjacency_functional
from per_turn_analysis import run_incremental, p_gamma_distance

import numpy as np

def to_abr_format(turns):
    """Map ShareGPT role names to ABR participant names."""
    return [{'participant': t['role'], 'text': t['text']} for t in turns]

def summarize_incremental(turn_results):
    if not turn_results:
        return {}
    deltas = [tr.delta_p_gamma for tr in turn_results]
    human_d = [tr.delta_p_gamma for tr in turn_results if tr.participant == 'human']
    model_d = [tr.delta_p_gamma for tr in turn_results if tr.participant == 'model']
    return {
        'mean_delta':        round(float(np.mean(deltas)), 4),
        'peak_delta':        round(float(np.max(deltas)), 4),
        'late_delta':        round(float(deltas[-1]), 4),
        'n_turns':           len(turn_results),
        'human_mean_delta':  round(float(np.mean(human_d)), 4) if human_d else 0.0,
        'model_mean_delta':  round(float(np.mean(model_d)), 4) if model_d else 0.0,
        'human_model_ratio': round(float(np.mean(human_d) / np.mean(model_d)), 3)
                             if human_d and model_d and np.mean(model_d) > 0 else None,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sample', type=int, default=200)
    ap.add_argument('--out', default='output/sharegpt_results.json')
    args = ap.parse_args()

    out_path = Path(__file__).parent / args.out
    out_path.parent.mkdir(exist_ok=True)

    feature_adj = declare_feature_adjacency_functional()

    print(f"Loading {args.sample} ShareGPT conversations...")
    convs = load_all(max_convs=args.sample)
    print(f"Loaded {len(convs)} conversations (min 4 turns)")

    results = []
    errors  = 0
    for i, c in enumerate(convs):
        if i % 50 == 0:
            print(f"  [{i}/{len(convs)}] processing...")
        try:
            abr_turns = to_abr_format(c['turns'])
            cf        = extract_conversation_features(abr_turns)
            trs       = run_incremental(abr_turns, feature_adj)
            summary   = summarize_incremental(trs)
            results.append({'id': c['id'], **summary})
        except Exception as e:
            errors += 1
            results.append({'id': c['id'], 'error': str(e)})

    valid = [r for r in results if 'mean_delta' in r]
    agg = {
        'n_conversations':      len(valid),
        'n_errors':             errors,
        'corpus_mean_delta':    round(float(np.mean([r['mean_delta'] for r in valid])), 4),
        'corpus_peak_delta':    round(float(np.max([r['peak_delta']  for r in valid])), 4),
        'corpus_mean_turns':    round(float(np.mean([r['n_turns']    for r in valid])), 1),
        'human_mean_delta':     round(float(np.mean([r['human_mean_delta'] for r in valid])), 4),
        'model_mean_delta':     round(float(np.mean([r['model_mean_delta'] for r in valid])), 4),
    }
    if agg['model_mean_delta'] > 0:
        agg['human_model_ratio'] = round(agg['human_mean_delta'] / agg['model_mean_delta'], 3)

    output = {'aggregate': agg, 'per_conversation': results}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*50}")
    print(f"  ShareGPT ABR Validation Results")
    print(f"{'='*50}")
    print(f"  Conversations:       {agg['n_conversations']}")
    print(f"  Errors:              {agg['n_errors']}")
    print(f"  Mean DeltaP_Gamma:   {agg['corpus_mean_delta']}")
    print(f"  Peak DeltaP_Gamma:   {agg['corpus_peak_delta']}")
    print(f"  Mean turns/conv:     {agg['corpus_mean_turns']}")
    print(f"  Human mean delta:    {agg['human_mean_delta']}")
    print(f"  Model mean delta:    {agg['model_mean_delta']}")
    if 'human_model_ratio' in agg:
        print(f"  Human/Model ratio:   {agg['human_model_ratio']}")
    print(f"\n  Results -> {out_path}")

if __name__ == '__main__':
    main()