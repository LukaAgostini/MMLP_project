"""
Unit commitment model (TDT4173 / SINTEF) - standalone script.

  1. loads the kernel data
  2. builds one row per (case, unit, hour) with a small feature set
  3. blocked leave-one-year-out CV (pooled micro ROC-AUC)
  4. fits a final model on all cases and saves it

Run:  python uc_model.py            (UC_DATA=path/to/kernel to override the data folder)
"""
import os
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------
DATA = Path(os.environ.get("UC_DATA", "../sintef_project/data/kernel"))
H = 168                                   # hours per case
GAP = pd.Timedelta(days=8)                # keep overlapping cases out of train when validating
TRAIN_STRIDE = 2                          # CV only: train on every n-th case (speed)
N_ROUNDS = int(os.environ.get("UC_ROUNDS", 400))

LGB_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
    feature_fraction=0.8, bagging_fraction=0.7, bagging_freq=1, lambda_l2=1.0,
    num_threads=-1, verbose=-1, seed=0,
)

PLANT_TO_RES = {
    "Hogga": ["Bandak"], "Tokke": ["Vinjevatn"], "Lio": ["Byrtevatn"], "Byrte": ["Botnedalsvatn"],
    "Vinje": ["Vaamarvatn"], "Haukeli": ["Vatjern"], "Songa": ["Bitdalsvatn", "Songavatn"],
    "Kjela": ["Foersvatn"], "Vesle Kjela": ["Kjelavatn"],
}

FEATURES = [
    "unit", "hour_of_day", "day_of_week", "hour_in_case",
    "price", "price_rank_in_case", "price_z_in_case", "price_minus_value",
    "storage_anom", "storage_anom_rel", "value_anom", "value_anom_rel",
]

COL_RE = re.compile(r"^result_committed_(?P<plant>.+)_(?P<unit>[^_]+)_t(?P<hour>\d+)$")


# ----------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------
def read_ts(path):
    """CSV with a 'date' column -> DataFrame indexed by UTC-naive timestamps."""
    d = pd.read_csv(path)
    d["date"] = pd.to_datetime(d["date"], utc=True).dt.tz_localize(None)
    return d.set_index("date").sort_index()


def load_uc(path):
    """Wide decisions CSV -> Y[case, unit, hour] (float32 0/1), case start times, unit ids."""
    df = pd.read_csv(path)
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True).dt.tz_localize(None)
    df = df.sort_values("starttime").reset_index(drop=True)

    recs = [(c, m["plant"], m["unit"], int(m["hour"]))
            for c in df.columns if (m := COL_RE.match(c))]
    meta = pd.DataFrame(recs, columns=["col", "plant", "unit", "hour"])
    meta["unit_id"] = meta["plant"] + "_" + meta["unit"]
    unit_ids = list(dict.fromkeys(meta["unit_id"]))

    Y = np.full((len(df), len(unit_ids), H), np.nan, dtype=np.float32)
    for u, uid in enumerate(unit_ids):
        sub = meta[meta["unit_id"] == uid].sort_values("hour")
        Y[:, u, sub["hour"].to_numpy()] = df[sub["col"]].to_numpy(dtype=np.float32)
    return Y, pd.DatetimeIndex(df["starttime"]), unit_ids


# ----------------------------------------------------------------------------------
# Unit-level aggregation of reservoir data and anomalies
# ----------------------------------------------------------------------------------
def unit_wide(df, unit_ids, agg):
    """Reservoir columns -> one column per unit (aggregating the unit's plant reservoirs)."""
    out = {}
    for uid in unit_ids:
        plant = uid.rsplit("_", 1)[0]
        res = PLANT_TO_RES.get(plant) or PLANT_TO_RES.get(plant.replace("_", " ")) or []
        cols = [c for c in res if c in df.columns]
        out[uid] = getattr(df[cols], agg)(axis=1) if cols else pd.Series(np.nan, index=df.index)
    return pd.DataFrame(out).astype(np.float32)


def rel(daily, clim):
    """Relative deviation from climatology (NaN where the climatology is ~0)."""
    denom = clim.abs().where(clim.abs() > 1e-6)
    return (daily - clim) / denom


def anomaly(daily):
    """Deviation from the day-of-year mean over the full history."""
    clim = daily.groupby(daily.index.dayofyear).transform("mean")
    return daily - clim, rel(daily, clim)


def load_context(unit_ids):
    price = read_ts(DATA / "Historical_day_ahead_price_2015_2025.csv")["Day-ahead price (EUR/MWh)"]
    sources = {
        "storage": (read_ts(DATA / "Historical_volume_2015_2024.csv"), "sum"),
        "value": (read_ts(DATA / "Synthetic_water_value_2015_2024.csv"), "mean"),
    }
    daily, value_daily = {}, None
    for base, (raw, agg) in sources.items():
        d = unit_wide(raw.astype(np.float32), unit_ids, agg).resample("D").mean()
        a_abs, a_rel = anomaly(d)
        daily[f"{base}_anom"], daily[f"{base}_anom_rel"] = a_abs, a_rel
        if base == "value":
            value_daily = d
    return dict(price=price.astype(np.float32), daily=daily, value=value_daily)


# ----------------------------------------------------------------------------------
# Feature table: one row per (case, unit, hour), C-order = (case, unit, hour)
# ----------------------------------------------------------------------------------
def build_table(starts, unit_ids, ctx, Y=None):
    n, U = len(starts), len(unit_ids)

    hours = starts.to_numpy()[:, None] + np.arange(H).astype("timedelta64[h]")[None, :]   # (n, H)
    ts = pd.DatetimeIndex(hours.ravel())

    def case_hour(M):                      # (n, H) -> (n*U*H)
        return np.repeat(np.asarray(M).reshape(n, 1, H), U, axis=1).ravel()

    def case_unit(M):                      # (n, U) -> (n*U*H)
        return np.repeat(np.asarray(M).reshape(n, U, 1), H, axis=2).ravel()

    def lookup(wide, keys):                # rows of `wide` at `keys`; NaN where missing -> (len(keys), U)
        idx = wide.index.get_indexer(keys)
        vals = wide[unit_ids].to_numpy(np.float32)[idx]
        vals[idx < 0] = np.nan
        return vals

    # price features (same for all units in a case)
    P = ctx["price"].reindex(ts).to_numpy(np.float32).reshape(n, H)
    rank = pd.DataFrame(P).rank(axis=1, pct=True).to_numpy(np.float32)
    z = (P - np.nanmean(P, axis=1, keepdims=True)) / (np.nanstd(P, axis=1, keepdims=True) + 1e-6)

    # end-of-case water value per unit (value at start of day 8), price minus that value
    v_end = lookup(ctx["value"], starts.floor("D") + pd.Timedelta(days=7))               # (n, U)

    tab = pd.DataFrame({
        "case_idx": np.repeat(np.arange(n, dtype=np.int32), U * H),
        "unit": pd.Categorical.from_codes(np.tile(np.repeat(np.arange(U), H), n), categories=unit_ids),
        "hour_in_case": np.tile(np.arange(H, dtype=np.int16), n * U),
        "hour_of_day": case_hour(ts.hour.to_numpy().reshape(n, H)).astype(np.int8),
        "day_of_week": case_hour(ts.dayofweek.to_numpy().reshape(n, H)).astype(np.int8),
        "price": case_hour(P),
        "price_rank_in_case": case_hour(rank),
        "price_z_in_case": case_hour(z.astype(np.float32)),
        "price_minus_value": case_hour(P) / case_unit(v_end),
    })

    # daily per-unit anomaly features, taken from the calendar day of each hour
    day_keys = ts.floor("D")
    for name, wide in ctx["daily"].items():
        vals = lookup(wide, day_keys).reshape(n, H, U)
        tab[name] = vals.transpose(0, 2, 1).reshape(-1)

    if Y is not None:
        tab["y"] = Y.reshape(-1).astype(np.int8)
    return tab


# ----------------------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------------------
def year_folds(starts):
    """Leave-one-year-out at case level with a gap so no train case overlaps a validation case."""
    years = starts.year.to_numpy()
    for vy in sorted(set(years)):
        val = years == vy
        vs, ve = starts[val].min(), starts[val].max()
        tr = ~val & np.asarray((starts < vs - GAP) | (starts > ve + GAP))
        yield vy, tr, val


def fit(X, y, rounds=N_ROUNDS):
    ds = lgb.Dataset(X, label=y, categorical_feature=["unit"], free_raw_data=False)
    return lgb.train(LGB_PARAMS, ds, num_boost_round=rounds)


def cross_validate(tab, starts):
    case_idx = tab["case_idx"].to_numpy()
    X, y = tab[FEATURES], tab["y"].to_numpy()
    oof = np.full(len(tab), np.nan, dtype=np.float32)
    fold_auc = {}

    for vy, tr_c, va_c in year_folds(starts):
        tr_rows = tr_c[case_idx] & (case_idx % TRAIN_STRIDE == 0)
        va_rows = va_c[case_idx]
        model = fit(X[tr_rows], y[tr_rows])
        oof[va_rows] = model.predict(X[va_rows])
        fold_auc[vy] = roc_auc_score(y[va_rows], oof[va_rows])
        print(f"  fold {vy}: micro AUC = {fold_auc[vy]:.4f}")

    a = np.array(list(fold_auc.values()))
    print(f"CV micro AUC: mean {a.mean():.4f} +- {a.std():.4f}")

    ok = ~np.isnan(oof)
    print(f"Pooled OOF micro AUC: {roc_auc_score(y[ok], oof[ok]):.4f}")

    per_unit = {}
    for u, g in tab[ok].groupby("unit", observed=True).indices.items():
        yy, pp = y[ok][g], oof[ok][g]
        if yy.min() != yy.max():
            per_unit[u] = roc_auc_score(yy, pp)
    print("Per-unit AUC:\n", pd.Series(per_unit).round(4).to_string())
    return oof


# ----------------------------------------------------------------------------------
# Final fit / prediction
# ----------------------------------------------------------------------------------
def predict_cases(model, starts, unit_ids, ctx):
    """Probabilities for new cases (given their start times) as Y-shaped array (case, unit, hour)."""
    tab = build_table(pd.DatetimeIndex(starts), unit_ids, ctx)
    return model.predict(tab[FEATURES]).reshape(len(starts), len(unit_ids), H)


def main():
    Y, starts, unit_ids = load_uc(DATA / "Unit_commitment_decisions.csv")
    print(f"{Y.shape[0]} cases, {len(unit_ids)} units: {unit_ids}")
    ctx = load_context(unit_ids)
    tab = build_table(starts, unit_ids, ctx, Y)
    print(f"Feature table: {tab.shape}, missing share per feature:")
    print(tab[FEATURES].isna().mean().round(4).to_string())

    print("Cross-validation (leave-one-year-out):")
    cross_validate(tab, starts)

    print("Final fit on all cases...")
    model = fit(tab[FEATURES], tab["y"].to_numpy())
    model.save_model("uc_lgbm.txt")
    imp = pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("Gain importance:\n", (imp / imp.sum()).round(4).to_string())

    # Test-time usage (test case start times from the Kaggle data):
    #   P = predict_cases(model, test_starts, unit_ids, ctx)   # shape (n_test, n_units, 168)


if __name__ == "__main__":
    main()
