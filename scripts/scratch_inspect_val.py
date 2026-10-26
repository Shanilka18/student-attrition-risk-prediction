import io, requests, numpy as np, pandas as pd, json

ITEMS = ['S1','S2','S3','M1','M2','M3','B1','B2','B3','SE1','SE2','SE3','PS1','PS2','PS3']
RNG = np.random.default_rng(42)
rows = []
for i in range(150):
    row = {'study_id': f'SLIIT_TEST_{i:05d}'}
    for col in ITEMS:
        row[col] = int(RNG.integers(1, 6))
    rows.append(row)
df = pd.DataFrame(rows)
for col in ['S1', 'M2', 'B3']:
    df.loc[RNG.choice(df.index, 10, replace=False), col] = float('nan')
df.loc[RNG.choice(df.index, 5, replace=False), 'SE1'] = 7
df.loc[RNG.choice(df.index, 5, replace=False), 'PS2'] = 0
csv_bytes = df.to_csv(index=False).encode()
r = requests.post('http://127.0.0.1:8000/api/upload', files={'file': ('dirty.csv', io.BytesIO(csv_bytes), 'text/csv')}, timeout=30)
body = r.json()
val = body.get('validation_report', {})
print('Validation report keys:', list(val.keys()))
print(json.dumps(val, indent=2)[:2000])
