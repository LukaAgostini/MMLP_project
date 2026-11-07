"""Tuning del meta-modello di v18 (model_v18_meta.ipynb).

Riparte esattamente dal meta di v18: logistica su logit XGBoost e TCN, pesi globali + scostamenti per unità,
stessi split ('cv' a 5 blocchi con gap 7 e 'forward'). Prova C e due feature in più:
  prod: logit_xgb * logit_tcn (accordo tra i modelli)
  stat: logit medio della centrale per XGBoost e TCN (il "quante unità" di Tokke/Vinje)

File letti da DATA: preds_v18_xgb_test.npz, preds_v18_tcn_test.npz (chiave 'P', forma casi x 14 x 168),
Unit_commitment_decisions.csv; per la submission anche preds_v18_xgb_kaggle.npz, la TCN Kaggle
(preds_v18_tcn_kaggle.npz, oppure preds_v19_tcn_kaggle.npz se la TCN di v19 coincide con quella di v18),
prediction_mapping.csv e sample_submission.csv.
Uso: python meta_tuning_v18.py [cartella_dati] [tag_tcn, default v18]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else './data')
TCN = sys.argv[2] if len(sys.argv) > 2 else 'v18'  # 'v18b' = TCN da tcn_tuning_v18.py
T, GAP, MARGIN = 168, 7, 5e-4
BASE = (1e-4, '')                       # configurazione di v18_meta
GRID_C = [1e-4, 1e-3, 1e-2, 1e-1]
GRID_KIND = ['', 'prod', 'stat', 'prod+stat']

load = lambda name: np.load(DATA / f'preds_{name}.npz')['P']
lg = lambda p: np.log(np.clip(p, 1e-5, 1 - 1e-5) / (1 - np.clip(p, 1e-5, 1 - 1e-5)))
auc = lambda Y, P: roc_auc_score(Y.ravel(), P.ravel())

# label del test locale: stessa split di v18 (80% train, gap 7, casi ordinati per starttime)
UCD = next(p for p in (DATA / 'kernel' / 'Unit_commitment_decisions.csv', DATA / 'Unit_commitment_decisions.csv') if p.exists())
ucd = pd.read_csv(UCD, parse_dates=['starttime']).sort_values('starttime')
ycols = [c for c in ucd.columns if c.startswith('result_committed')]
norm = lambda c: c.lower().replace(' ', '_')
units = sorted({norm(c)[len('result_committed_'):].rsplit('_t', 1)[0] for c in ycols})
col_of = {norm(c): c for c in ycols}
ucols = [col_of[f'result_committed_{u}_t{h}'] for u in units for h in range(T)]
split = int(len(ucd) * 0.8)
Y = ucd.iloc[split + 7:][ucols].to_numpy(np.uint8).reshape(-1, len(units), T)

Px, Pt = load('v18_xgb_test'), load(f'{TCN}_tcn_test')
assert Px.shape == Pt.shape == Y.shape, (Px.shape, Pt.shape, Y.shape)
L = np.stack([lg(Px), lg(Pt)], -1)
n, nu = L.shape[:2]

# media di centrale: matrice (unità x unità) che fa la media sulle unità della stessa centrale
plant = [u.rsplit('_g', 1)[0] for u in units]
M = np.array([[float(p == q) for q in plant] for p in plant]); M /= M.sum(1, keepdims=True)

b = np.linspace(0, n, 6, dtype=int)
SPLITS = {'cv': [(np.r_[0:max(a - GAP, 0), min(c + GAP, n):n], np.arange(a, c)) for a, c in zip(b[:-1], b[1:])],
          'forward': [(np.arange(n // 2 - GAP), np.arange(n // 2, n))]}


def mx(L, kind):  # pesi globali (+ feature extra) + intercetta e pendenze per unità, come in v18
    a, c = L[..., 0], L[..., 1]
    G = [a, c]
    if 'prod' in kind: G.append(a * c)
    if 'stat' in kind: G += [np.einsum('uv,nvt->nut', M, a), np.einsum('uv,nvt->nut', M, c)]
    u = np.broadcast_to(np.arange(nu)[None, :, None], a.shape).ravel()
    U = lambda v: sp.csr_matrix((v, (np.arange(len(u)), u)), shape=(len(u), nu))
    return sp.hstack([sp.csr_matrix(np.column_stack([g.ravel() for g in G])),
                      U(np.ones(len(u))), U(a.ravel()), U(c.ravel())]).tocsr()


def fit(L_, Y_, C, kind):
    return LogisticRegression(C=C, max_iter=500).fit(mx(L_, kind), Y_.ravel())


def ev(C, kind):  # AUC per split + predizioni fuori fold della cv (per l'AUC per unità)
    out, oof = {}, np.zeros(Y.shape)
    for k, s in SPLITS.items():
        ys, zs = [], []
        for tr, va in s:
            z = fit(L[tr], Y[tr], C, kind).decision_function(mx(L[va], kind)).reshape(len(va), nu, T)
            ys.append(Y[va].ravel()); zs.append(z.ravel())
            if k == 'cv': oof[va] = z
        out[k] = roc_auc_score(np.concatenate(ys), np.concatenate(zs))
    return out, oof


if __name__ == '__main__':
    print(f'{n} casi, {nu} unità | XGB {auc(Y, Px):.4f} | TCN {auc(Y, Pt):.4f} | blend 30% {auc(Y, .7 * L[..., 0] + .3 * L[..., 1]):.4f}')
    res, oofs = {}, {}
    for C in GRID_C:
        for kind in GRID_KIND:
            res[(C, kind)], oofs[(C, kind)] = ev(C, kind)
            r = res[(C, kind)]
            print(f'C={C:<7g} {kind or "base":<10} cv {r["cv"]:.4f}  forward {r["forward"]:.4f}', flush=True)

    base = res[BASE]
    # cambia solo se migliora la cv di almeno MARGIN senza peggiorare il forward
    ok = {k: r for k, r in res.items() if r['cv'] > base['cv'] + MARGIN and r['forward'] >= base['forward']}
    choice = max(ok, key=lambda k: ok[k]['cv']) if ok else BASE
    print(f'\nv18 {BASE}: cv {base["cv"]:.4f} forward {base["forward"]:.4f} -> scelta {choice}: '
          f'cv {res[choice]["cv"]:.4f} forward {res[choice]["forward"]:.4f}')

    print(pd.DataFrame({'v18': [auc(Y[:, i], oofs[BASE][:, i]) for i in range(nu)],
                        'scelta': [auc(Y[:, i], oofs[choice][:, i]) for i in range(nu)]}, index=units).round(4))

    # submission: meta della config scelta su tutto il test locale, applicato alle predizioni Kaggle
    if choice == BASE and TCN == 'v18':  # con una TCN nuova la submission serve comunque
        sys.exit('\nnessuna configurazione batte v18 con margine: resta submission_v18_meta')
    tk = DATA / f'preds_{TCN}_tcn_kaggle.npz'
    if not tk.exists() and TCN == 'v18':  # la TCN di v19 è la stessa di v18? si verifica sul test locale
        d = np.abs(load('v19_tcn_test') - Pt).max() if (DATA / 'preds_v19_tcn_test.npz').exists() else np.inf
        print(f'\nTCN v19 vs v18 sul test locale: max |diff| = {d:.2e}')
        if d > 1e-3:
            sys.exit('TCN Kaggle di v18 non disponibile: salvarla da model_v18_meta (Pk_t) come preds_v18_tcn_kaggle.npz, chiave P')
        tk = DATA / 'preds_v19_tcn_kaggle.npz'
    Lk = np.stack([lg(load('v18_xgb_kaggle')), lg(np.load(tk)['P'])], -1)
    Pk = 1 / (1 + np.exp(-fit(L, Y, *choice).decision_function(mx(Lk, choice[1]))))
    mp, ss = pd.read_csv(DATA / 'prediction_mapping.csv'), pd.read_csv(DATA / 'sample_submission.csv')
    sub = pd.DataFrame(Pk.reshape(len(Lk), -1), columns=ucols)
    nm = {norm(c): c for c in ss.columns}
    sub = sub.rename(columns=lambda c: nm[norm(c)])
    sub.insert(0, 'Run No', mp['Run No'].to_numpy())
    sub = sub.set_index('Run No').loc[ss['Run No']].reset_index()[list(ss.columns)]
    name = DATA / f'submission_{TCN}_meta_C{choice[0]:g}_{choice[1].replace("+", "_") or "base"}.csv'
    sub.to_csv(name, index=False)
    print(f'\nsalvata {name} {sub.shape}')
