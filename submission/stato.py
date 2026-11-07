
"""Stato della run di uno short notebook: python stato.py 1  (oppure 2).
    per vedere lo stato della run, qunaod si avviano i notebook tramire linea di comando
        -> Usata in particolare su Azure ML
"""

import re, subprocess, sys, shutil
N = sys.argv[1] if len(sys.argv) > 1 else '1'
ORDER = ['vesle_kjela', 'songa', 'byrte', 'haukeli', 'kjela', 'lio', 'vinje', 'tokke', 'hogga']
sh = lambda c: subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
pid = sh("pgrep -f 'ipykernel_launcher -f /tmp/tmp' | head -1")
log = sh(f'cat run{N}.log 2>/dev/null')
if 'end ' in log:
    print(f'Short {N}: FINITO.', *[l for l in log.splitlines() if l.startswith(('start', 'end'))], sep='\n'); sys.exit()

if not pid: print(f'Short {N}: nessun kernel attivo. Controlla run{N}.log:\n{log}'); sys.exit()

dump = sh(f'sudo {shutil.which("py-spy")} dump --pid {pid} --locals')

dump = dump.split('"MainThread"', 1)[-1].split('\nThread ', 1)[0]

F = []  # [(funzione, file, {variabili})] dal frame più interno

for l in dump.splitlines():
    m = re.match(r'^ {4}(\S+) \((.+):\d+\)$', l)
    if m: F.append((m[1], m[2], {})); continue
    m = re.match(r'^ {8,}(\w+): (.*)$', l)
    if m and F: F[-1][2][m[1]] = m[2].strip('"')

get = lambda f, k: next((v[k] for n, _, v in F if n == f and k in v), None)

has = lambda f: any(n == f for n, _, _ in F)

plant = get('run', 'plant') or get('base', 'plant')

n = get('fit_tcn', 'n') or get('fit_group', 'n')

fase = {'2337': 'primo training (80% dei casi)', '2922': 'training su tutti i casi'}.get(n, '')

if N == '2': fase = ''

if has('fit_tcn'):
    cosa = f"rete temporale: seed {int(get('fit_tcn', 'seed') or 0) + 1}/3, epoca {int(get('fit_tcn', 'ep') or 0) + 1}/10"

elif has('solve') or has('linprog'):
    cosa = f'LP della cascata (feature di {plant})'

elif has('fit_group'):
    mod = next((m for k, m in [('lgb', 'LightGBM'), ('catboost', 'CatBoost'), ('rf_model', 'ExtraTrees'), ('xt', 'ExtraTrees'), ('xgboost', 'XGBoost')] if any(k in f for _, f, _ in F)), 'alberi')
    it = get('update', 'iteration')
    cosa = f'{mod} su {plant} (centrale {ORDER.index(plant) + 1}/9)' + (f', albero {int(it) + 1}/200 di uno dei 18 fit della centrale' if it else '')

elif plant:
    cosa = f'feature di {plant} (centrale {ORDER.index(plant) + 1}/9)'

else:
    cosa = 'meta-modello / submission' if has('fit') or has('decision_function') else 'altro: ' + ' < '.join(n for n, _, _ in F[:4])


print(f"Short {N} | in esecuzione da {sh(f'ps -o etime= -p {pid}')} | {fase + ' | ' if fase else ''}{cosa}")
