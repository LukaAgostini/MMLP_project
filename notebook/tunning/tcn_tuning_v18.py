"""Tuning della TCN di v18, fuori dai notebook (nei notebook vanno solo i valori migliori).

Riusa il codice di model_v18_meta.ipynb (feature L, LP, fit_tcn) eseguendone le celle di definizione,
quindi va lanciato dalla cartella del notebook, con ./data come in v18.

1) screening su validazione interna al train (ultimo 20% del train, gap 7), 1 seed per variante:
       python tcn_tuning_v18.py screen
2) config scelta, tutti i seed: test locale + Kaggle (pesi 2021+ x2, come in v18), salvati come
   data/preds_v18b_tcn_test.npz e data/preds_v18b_tcn_kaggle.npz:
       python tcn_tuning_v18.py final EPOCHS=15 DROP=0.2 TSEEDS=5
   poi: python meta_tuning_v18.py ./data v18b
"""
import sys
import json
import numpy as np
import pandas as pd

NB = 'model_v18_meta.ipynb'
VARIANTS = {  # rispetto a v18: EPOCHS=10, CH=128, LR=1e-3, DROP=0.2
    'v18': {},
    'ep15': {'EPOCHS': 15},
    'ep20': {'EPOCHS': 20},
    'drop0.1': {'DROP': 0.1},
    'drop0.3': {'DROP': 0.3},
    'ch192': {'CH': 192},
    'lr2e-3': {'LR': 2e-3},
}

# definizioni del notebook, fino alla TCN esclusa (salta XGBoost e tutto ciò che allena)
g = {}
for cell in json.load(open(NB))['cells']:
    src = ''.join(cell['source'])
    if cell['cell_type'] != 'code' or "cached('xgb_test'" in src:
        continue
    if 'fit_tcn(x_train' in src:
        break
    exec(src, g)
auc, fit_tcn = g['auc'], g['fit_tcn']
DEFAULTS = {k: g[k] for k in ('EPOCHS', 'CH', 'LR', 'DROP', 'TSEEDS')}


def run(cfg, *args, **kw):
    g.update({**DEFAULTS, **cfg})  # fit_tcn legge gli iperparametri dai globali del notebook
    return fit_tcn(*args, **kw)


if sys.argv[1] == 'screen':
    xtr, Ytr = g['x_train'], g['Y3']
    v = int(len(xtr) * 0.8)
    xa, Ya, xv, Yv = xtr.iloc[:v - 7], Ytr[:v - 7], xtr.iloc[v:], Ytr[v:]
    print(f'screening: train {len(xa)} casi, validazione {len(xv)} casi', flush=True)
    res = {}
    for name, cfg in VARIANTS.items():
        res[name] = auc(Yv, run({**cfg, 'TSEEDS': 1}, xa, Ya, xv, Yp=Yv))
        print(f'>>> {name:<8} val {res[name]:.4f} ({res[name] - res["v18"]:+.4f})', flush=True)
        pd.Series(res, name='val_auc').to_csv('tcn_tuning_results.csv')

elif sys.argv[1] == 'final':
    cfg = {k: type(DEFAULTS[k])(float(v)) for k, v in (a.split('=') for a in sys.argv[2:])}
    print('config', {**DEFAULTS, **cfg}, flush=True)
    P = run(cfg, g['x_train'], g['Y3'], g['x_test'], Yp=g['Y3te'])
    print(f'TCN test {auc(g["Y3te"], P):.4f} (v18 0.9723)', flush=True)
    np.savez_compressed(f'{g["DATA"]}/preds_v18b_tcn_test.npz', P=P)

    # Kaggle, come nella cella di submission di v18
    DATA, X_all, Y_all, build_cases = g['DATA'], g['X_all'], g['Y_all'], g['build_cases']
    mp = pd.read_csv(f'{DATA}/prediction_mapping.csv'); mp['starttime'] = pd.to_datetime(mp['starttime'], utc=True)
    dummy = pd.DataFrame(0, index=pd.DatetimeIndex(mp['starttime']),
                         columns=['Run No', 'total_calculation_time', *g['y_train'].columns])
    x_all, y_all = build_cases(X_all, Y_all)
    x_kag, _ = build_cases(X_all, dummy)
    cw = np.where(x_all.index.year >= 2021, 2.0, 1.0)
    Pk = run(cfg, x_all, g['stack3'](y_all), x_kag, cw)
    np.savez_compressed(f'{DATA}/preds_v18b_tcn_kaggle.npz', P=Pk)
    print('salvate preds_v18b_tcn_test.npz e preds_v18b_tcn_kaggle.npz')
