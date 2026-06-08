"""
Inspect ShareChat schema.
If you have parquet files, first convert:
  pip install pandas pyarrow
  python -c "import pandas as pd; pd.read_parquet('validation/data/sharechat/sharechat-train-00000.parquet').head(3).to_json('validation/data/sharechat/sample.json', orient='records', indent=2)"
Then run this script.
"""
import json
from pathlib import Path

data_dir = Path(r'D:\Metatron_Dynamics\GitHub_Repos\metatron-alignment-proposal\validation\data\sharechat')

# Try JSON sample first, then JSONL
candidates = list(data_dir.glob('sample.json')) + list(data_dir.glob('*.jsonl'))
if not candidates:
    print("No JSON/JSONL files found yet. Convert parquet first (see docstring above).")
    print("Files present:")
    for f in data_dir.rglob('*'): print(f"  {f}")
else:
    path = candidates[0]
    print(f"Inspecting: {path}")
    with open(path, encoding='utf-8') as f:
        raw = json.load(f) if path.suffix == '.json' else [json.loads(l) for l in f][:5]
    first = raw[0]
    print(f"\nKeys: {list(first.keys())}")
    for key in ('conversations','messages','turns','dialogue'):
        if key in first:
            turns = first[key]
            print(f"\nTurn key: '{key}', count: {len(turns)}")
            print(f"First turn keys: {list(turns[0].keys())}")
            print(json.dumps(turns[0], ensure_ascii=False, indent=2)[:400])
            break
    print(f"\nFull first record (truncated):")
    print(json.dumps(first, ensure_ascii=False, indent=2)[:600])