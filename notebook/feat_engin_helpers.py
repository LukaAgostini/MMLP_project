import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

COL_RE = re.compile(r"^result_committed_(?P<plant>.+)_(?P<unit>[^_]+)_t(?P<hour>\d+)$")


def _utc_naive(s):
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def parse_uc_columns(columns):
    recs = []
    for c in columns:
        m = COL_RE.match(c)
        if m:
            recs.append((c, m["plant"], m["unit"], f'{m["plant"]}_{m["unit"]}', int(m["hour"])))
    meta = pd.DataFrame(recs, columns=["col", "plant", "unit", "unit_id", "hour"])
    return meta


def load_uc():
    df = pd.read_csv("../sintef_project/data/kernel/Unit_commitment_decisions.csv")

    run_col = next(c for c in df.columns if c.lower().replace(" ", "_") == "run_no")
    df["starttime"] = _utc_naive(df["starttime"])
    df = df.sort_values("starttime").reset_index(drop=True)

    meta = parse_uc_columns(df.columns)
    unit_ids = list(dict.fromkeys(meta["unit_id"]))          # keeps file order
    n_hours = meta["hour"].max() + 1

    Y = np.full((len(df), len(unit_ids), n_hours), np.nan, dtype=np.float32)
    for u, uid in enumerate(unit_ids):
        sub = meta[meta["unit_id"] == uid].sort_values("hour")
        Y[:, u, sub["hour"].to_numpy()] = df[sub["col"]].to_numpy(dtype=np.float32)
    bad = ~np.isin(Y[~np.isnan(Y)], [0.0, 1.0])

    cases = pd.DataFrame({
        "run_no": df[run_col].to_numpy(),
        "starttime": df["starttime"],
        "case_idx": np.arange(len(df)),
        "year": df["starttime"].dt.year.to_numpy(),
    })
    return Y, cases, unit_ids


def to_long(Y, cases, unit_ids):
    n_cases, n_units, n_hours = Y.shape
    per_case = n_units * n_hours

    case_idx = np.repeat(cases["case_idx"].to_numpy(), per_case)
    unit_code = np.tile(np.repeat(np.arange(n_units), n_hours), n_cases)
    hour = np.tile(np.arange(n_hours, dtype=np.int16), n_cases * n_units)
    start = np.repeat(cases["starttime"].to_numpy(), per_case)

    long = pd.DataFrame({
        "case_idx": case_idx.astype(np.int32),
        "unit": pd.Categorical.from_codes(unit_code, categories=unit_ids),
        "hour": hour,
        "y": Y.reshape(-1).astype(np.int8),
    })
    long["timestamp"] = start + hour.astype("timedelta64[h]")
    long["hour_of_day"] = long["timestamp"].dt.hour.astype(np.int8)
    long["day_in_case"] = (hour // 24).astype(np.int8)  # 0..6
    long["year"] = long["case_idx"].map(cases.set_index("case_idx")["year"]).astype(np.int16)
    return long


# Helpers to add features

def attach_hourly(long, series, name):
    long[name] = series.reindex(long["timestamp"]).to_numpy()
    return long


def attach_daily(long, series, name):
    long[name] = series.reindex(long["timestamp"].dt.floor("D")).to_numpy()
    return long


def attach_case_level(long, cases, series, name, offset_days=0):
    key = cases["starttime"].dt.floor("D") + pd.Timedelta(days=offset_days)
    vals = pd.Series(series.reindex(key).to_numpy(), index=cases["case_idx"].to_numpy())
    long[name] = long["case_idx"].map(vals).to_numpy()
    return long


def read_timeseries(path, date_col="date"):
    d = pd.read_csv(path)
    d[date_col] = _utc_naive(d[date_col])
    return d.set_index(date_col).sort_index()

def attach_unit_specific(long, df_units, name, level="hourly",
                         offset_days=0, unit_map=None):
    cats = long["unit"].cat.categories
    cols = [unit_map[c] for c in cats] if unit_map else list(cats)
    missing = [c for c in cols if c not in df_units.columns]
    if missing:
        raise KeyError(f"No column in df_units for: {missing}")

    arr = df_units[cols].to_numpy(dtype=np.float32)   # columns now in cats order

    if level == "hourly":
        key = long["timestamp"]
    elif level == "daily":
        key = long["timestamp"].dt.floor("D")
    elif level == "case":
        start = long["timestamp"] - pd.to_timedelta(long["hour"].astype("int64"), unit="h")
        key = start.dt.floor("D") + pd.Timedelta(days=offset_days)
    else:
        raise ValueError(level)

    t_pos = df_units.index.get_indexer(key)
    u_pos = long["unit"].cat.codes.to_numpy()
    out = np.full(len(long), np.nan, dtype=np.float32)
    ok = t_pos >= 0
    out[ok] = arr[t_pos[ok], u_pos[ok]]
    long[name] = out
    return long

# Calculate correlations

def filter_scores(df, feature_cols, target_col):
    METRICS = ["correlation", "auc", "strength", "sign_consistency", "frac_missing"]
    cols = ["feature"] + METRICS
    if df[target_col].nunique() < 2:                     # constant target
        return pd.DataFrame(columns=cols)
    rows = []
    for f in feature_cols:
        d = df[[f, target_col, "year"]].dropna()
        # constant feature (e.g. Hyljelihyl / Vatjern storage) -> no information, skip
        if d[target_col].nunique() < 2 or d[f].nunique() < 2:
            continue
        x, y = d[f].to_numpy(), d[target_col].to_numpy()
        auc = roc_auc_score(y, x)
        per_year = [roc_auc_score(g[target_col], g[f])
                    for _, g in d.groupby("year")
                    if g[target_col].nunique() == 2 and g[f].nunique() > 1]
        rows.append(dict(
            feature=f,
            correlation=spearmanr(x, y).correlation,
            auc=auc,
            strength=abs(auc - 0.5),
            sign_consistency=(np.mean(np.sign(np.array(per_year) - 0.5) == np.sign(auc - 0.5))
                              if per_year else np.nan),
            frac_missing=1 - len(d) / len(df),
        ))
    # fixed columns, so an empty result is still a valid (empty) DataFrame
    return pd.DataFrame(rows, columns=cols).sort_values("strength", ascending=False)