import time, requests, json
time.sleep(3)

r = requests.get('http://127.0.0.1:8000/api/leaderboard?track=regression', timeout=5)
data = r.json()
print('REGRESSION LEADERBOARD (after fix):')
print('  Rank | Model                    | Val R2  | Val RMSE | Champion')
print('  ' + '-'*65)
for e in data['entries']:
    champ = '<-- CHAMPION' if e['is_champion'] else ''
    rmse = str(e['secondary_metric']) if e['secondary_metric'] else '---'
    rank = e['rank']
    name = e['model_name']
    pm = e['primary_metric']
    print(f'  #{rank}    {name:24s}  {pm:.4f}   {rmse:8}  {champ}')

print()
r2 = requests.get('http://127.0.0.1:8000/api/leaderboard?track=classification', timeout=5)
data2 = r2.json()
print('CLASSIFICATION LEADERBOARD (after fix):')
print('  Rank | Model                    | Macro-F1 | MCC    | Champion')
print('  ' + '-'*65)
for e in data2['entries']:
    champ = '<-- CHAMPION' if e['is_champion'] else ''
    mcc = str(e['secondary_metric']) if e['secondary_metric'] else '---'
    rank = e['rank']
    name = e['model_name']
    pm = e['primary_metric']
    print(f'  #{rank}    {name:24s}  {pm:.4f}   {mcc:8}  {champ}')
