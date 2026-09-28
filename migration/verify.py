"""Verify copied source artifacts against the migration inventory (offline)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
rows = json.loads((root / 'migration/included.json').read_text())
failures = []
for row in rows:
    path = root / row['path']
    if not path.is_file():
        failures.append((row['path'], 'missing'))
    elif path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        failures.append((row['path'], 'changed'))
for path, reason in failures:
    print(f'{reason}: {path}')
print(f'{len(rows) - len(failures)}/{len(rows)} source artifacts match the inventory')
raise SystemExit(bool(failures))
