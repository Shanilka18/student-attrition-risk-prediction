import requests

base = 'http://127.0.0.1:8000'
all_ok = True

def check(method, path, body=None, note=""):
    global all_ok
    try:
        if method == 'GET':
            r = requests.get(base + path, timeout=5)
        else:
            r = requests.post(base + path, json=body, timeout=5)
        ok = r.status_code < 400
        if not ok:
            all_ok = False
        tag = 'OK  ' if ok else 'FAIL'
        print(f'  [{tag}] {method:4} {path:50} {r.status_code}  {note}')
        return r
    except Exception as e:
        all_ok = False
        print(f'  [ERR ] {method:4} {path:50} ERROR: {e}')
        return None

print('=== LIVE API SMOKE TEST ===')
print(f'Backend  -> {base}')
print(f'Frontend -> http://localhost:5173/')
print()

check('GET',  '/health')
check('GET',  '/api/meta')
check('GET',  '/api/champion')
check('GET',  '/api/reliability')
check('GET',  '/api/shap/global?track=regression')
check('GET',  '/api/shap/global?track=classification')
check('GET',  '/api/leaderboard?track=regression')
check('GET',  '/api/leaderboard?track=classification')
check('GET',  '/api/eda')
check('GET',  '/api/docs')

student = {
    'study_id': 'TEST_001',
    'S1':5,'S2':4,'S3':5,
    'M1':1,'M2':2,'M3':1,
    'B1':1,'B2':2,'B3':1,
    'SE1':2,'SE2':1,'SE3':2,
    'PS1':1,'PS2':2,'PS3':1,
}
r = check('POST', '/predict', student)
if r and r.status_code == 200:
    d = r.json()
    pri = d.get('pri_score', '?')
    risk = d.get('risk_category', '?')
    print(f'         -> PRI={pri}  risk_category={risk}')

print()
print('Result:', 'ALL OK' if all_ok else 'SOME FAILURES - check above')
