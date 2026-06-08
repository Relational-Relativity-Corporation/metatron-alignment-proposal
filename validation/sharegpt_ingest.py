"""
ShareGPT -> ABR turn sequence adapter.
Schema: [{"id": str, "conversations": [{"from": "human"|"gpt", "value": str}, ...]}]
"""
import json
from pathlib import Path

DATA_FILE = Path(__file__).parent / 'data' / 'sharegpt' / 'sg_90k_part1.json'

def load_all(path=DATA_FILE, min_turns=4, max_convs=None):
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    results = []
    for conv in raw:
        turns = [
            {'role': 'human' if t['from'] == 'human' else 'model',
             'text': t['value'].strip()}
            for t in conv.get('conversations', [])
            if t.get('value', '').strip()
        ]
        if len(turns) >= min_turns:
            results.append({'id': conv.get('id', ''), 'turns': turns})
        if max_convs and len(results) >= max_convs:
            break
    return results

if __name__ == '__main__':
    convs = load_all(max_convs=5)
    print(f"Sample load: {len(convs)} conversations")
    c = convs[0]
    print(f"  id={c['id']}  turns={len(c['turns'])}")
    print(f"  turn[0]: role={c['turns'][0]['role']}  text={c['turns'][0]['text'][:80]!r}")
    print(f"  turn[1]: role={c['turns'][1]['role']}  text={c['turns'][1]['text'][:80]!r}")