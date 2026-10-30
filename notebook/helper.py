"""
commitment_corr.py - helper framework for testing many correlations between
unit commitment decisions and reservoir-/plant-/market-related data.
 
Typical use
-----------
    from commitment_corr import CommitmentCorrelator
 
    cc = CommitmentCorrelator(
        data.commitment_decision_data,
        reservoir_to_plants={"Kjelavatn": ["Hogga"], "Totak": ["Tokke", "Vinje"]},
    )
    cc.add_feature("volume", data.volume_data)                    # columns = reservoirs
    cc.add_feature("water_value", data.water_value_data)
    cc.add_feature("inflow", data.inflow_data)
    cc.add_feature("price", data.price_data, kind="global")       # one value column
 
    res = cc.correlate(entity="reservoir", granularity="day", lags=[0, 1, 2])
    res = cc.correlate(entity="plant", granularity="hour", group_by="month")
 
Everything is returned as a tidy DataFrame, so you can sort / pivot / plot it.
"""
from __future__ import annotations
 
import re
import warnings
from typing import Callable, Iterable, Sequence
 
import numpy as np
import pandas as pd
 
_COMMIT_RE = re.compile(r"^result_committed_(?P<plant>.+)_(?P<unit>[^_]+)_t(?P<hour>\d+)$")
_FREQ = {"hour": "h", "day": "D"}
 
 
# --------------------------------------------------------------------------- #
# 1. Wide commitment table -> long table
# --------------------------------------------------------------------------- #
def commitments_to_long(
    outcome: pd.DataFrame,
    run_col: str = "Run No",
    start_col: str = "starttime",
) -> pd.DataFrame:
    """
    One row per (run, plant, unit, hour-in-horizon).
 
    Columns: run, plant, unit, hour_idx, time (= starttime + hour_idx hours, UTC),
             committed (int8)
    """
    cols = [c for c in outcome.columns if _COMMIT_RE.match(c)]
    if not cols:
        raise ValueError("No columns matching result_committed_<plant>_<unit>_t<hour>")
 
    meta = pd.DataFrame(
        [_COMMIT_RE.match(c).groupdict() for c in cols], index=cols
    ).astype({"hour": int})
    meta = meta.rename(columns={"hour": "hour_idx"})
 
    base = outcome[[run_col, start_col] + cols].copy()
    base[start_col] = pd.to_datetime(base[start_col], utc=True)
 
    long = base.melt(id_vars=[run_col, start_col], value_vars=cols,
                     var_name="col", value_name="committed")
    long = long.join(meta, on="col").drop(columns="col")
    long["time"] = long[start_col] + pd.to_timedelta(long["hour_idx"], unit="h")
    long = long.rename(columns={run_col: "run"}).drop(columns=start_col)
    long["committed"] = long["committed"].astype("int8")
    for c in ("plant", "unit"):
        long[c] = long[c].astype("category")
    return long[["run", "plant", "unit", "hour_idx", "time", "committed"]]
 
 
# --------------------------------------------------------------------------- #
# 2. The framework
# --------------------------------------------------------------------------- #
class CommitmentCorrelator:
    """
    entity kinds (what a 'row group' is):
        "unit"      -> "<plant>_<unit>"   e.g. Hogga_G1
        "plant"     -> plant name
        "reservoir" -> reservoir name (needs reservoir_to_plants)
        "all"       -> single entity "ALL" (mean over every unit)
 
    Target value = share of units "on" (mean of 0/1) in each period. If several
    runs cover the same period (overlapping cases) they are averaged.
    """
 
    def __init__(
        self,
        outcome: pd.DataFrame,
        reservoir_to_plants: dict[str, Sequence[str]] | None = None,
        run_col: str = "Run No",
        start_col: str = "starttime",
    ):
        self.long = commitments_to_long(outcome, run_col, start_col)
        self.reservoir_to_plants = dict(reservoir_to_plants or {})
        self.features: dict[str, tuple[str, pd.DataFrame]] = {}
        self._target_cache: dict[tuple[str, str], pd.DataFrame] = {}
 
        known = set(self.long["plant"].cat.categories)
        for r, plants in self.reservoir_to_plants.items():
            missing = [p for p in plants if p not in known]
            if missing:
                warnings.warn(f"Reservoir {r}: plants not in commitment data: {missing}")
 
    # -- helpers ----------------------------------------------------------- #
    @property
    def plants(self) -> list[str]:
        return list(self.long["plant"].cat.categories)
 
    def add_reservoir_mapping(self, reservoir: str, plants: Sequence[str]) -> None:
        self.reservoir_to_plants[reservoir] = list(plants)
        self._target_cache.clear()
 
    # -- features ---------------------------------------------------------- #
    def add_feature(
        self,
        name: str,
        df: pd.DataFrame,
        kind: str = "reservoir",
        date_col: str = "date",
    ) -> None:
        """
        Register a wide DataFrame: one datetime column + one column per entity
        (reservoir / plant / unit names). kind="global" means the columns are NOT
        entities (e.g. price) and are broadcast to every entity; each column
        becomes its own feature called "<name>:<col>" (or just <name> if one column).
        """
        if kind not in ("reservoir", "plant", "unit", "global"):
            raise ValueError("kind must be reservoir | plant | unit | global")
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], utc=True)
        value_cols = [c for c in df.columns if c != date_col]
 
        if kind == "global":
            for c in value_cols:
                fname = name if len(value_cols) == 1 else f"{name}:{c}"
                f = df[[date_col, c]].rename(columns={date_col: "time", c: "value"})
                self.features[fname] = (kind, f.dropna().sort_values("time"))
        else:
            f = df.melt(id_vars=date_col, var_name="entity", value_name="value")
            f = f.rename(columns={date_col: "time"}).dropna()
            self.features[name] = (kind, f.sort_values("time"))
 
    # -- targets ----------------------------------------------------------- #
    def target(self, entity: str = "reservoir", granularity: str = "day") -> pd.DataFrame:
        """Columns: entity, time, y (share of units on)."""
        key = (entity, granularity)
        if key in self._target_cache:
            return self._target_cache[key]
 
        u = self.long.assign(time=self.long["time"].dt.floor(_FREQ[granularity]))
        if entity == "unit":
            u["entity"] = u["plant"].astype(str) + "_" + u["unit"].astype(str)
        elif entity == "plant":
            u["entity"] = u["plant"].astype(str)
        elif entity == "all":
            u["entity"] = "ALL"
        elif entity == "reservoir":
            pairs = pd.DataFrame(
                [(r, p) for r, ps in self.reservoir_to_plants.items() for p in ps],
                columns=["entity", "plant"],
            )
            if pairs.empty:
                raise ValueError("Provide reservoir_to_plants for entity='reservoir'")
            u = u.assign(plant=u["plant"].astype(str)).merge(pairs, on="plant")
        else:
            raise ValueError("entity must be unit | plant | reservoir | all")
 
        out = (
            u.groupby(["entity", "time"], observed=True)["committed"]
            .mean().rename("y").reset_index()
        )
        self._target_cache[key] = out
        return out
 
    # -- align one feature to the target grid ------------------------------ #
    @staticmethod
    def _step(times: pd.Series) -> pd.Timedelta:
        t = np.sort(times.unique())
        return pd.Timedelta(np.median(np.diff(t))) if len(t) > 1 else pd.Timedelta("1D")
 
    def _aligned(self, tgt, fname, granularity, lag) -> pd.DataFrame:
        kind, f = self.features[fname]
        grid = pd.Timedelta(1, "h") if granularity == "hour" else pd.Timedelta(1, "D")
        fstep = self._step(f["time"])
 
        f = f.copy()
        if fstep < grid:  # finer than grid -> average down
            f["time"] = f["time"].dt.floor(_FREQ[granularity])
            by = ["time"] if kind == "global" else ["entity", "time"]
            f = f.groupby(by, observed=True)["value"].mean().reset_index()
        f["time"] = f["time"] + lag * grid          # value at t-lag lines up with target at t
        f = f.sort_values("time")
 
        tol = max(fstep, grid)                      # coarser feature -> forward-fill, but only
        t = tgt.sort_values("time")                 # within one feature step (no stale data)
        if kind == "global":
            m = pd.merge_asof(t, f, on="time", direction="backward", tolerance=tol)
        else:
            m = pd.merge_asof(t, f, on="time", by="entity", direction="backward", tolerance=tol)
        return m.dropna(subset=["value", "y"])
 
    # -- main entry -------------------------------------------------------- #
    def correlate(
        self,
        features: Iterable[str] | None = None,
        entity: str = "reservoir",
        granularity: str = "day",
        lags: Sequence[int] = (0,),
        methods: Sequence[str] = ("pearson", "spearman"),
        group_by: str | Callable[[pd.DataFrame], pd.Series] | None = None,
        min_obs: int = 30,
    ) -> pd.DataFrame:
        """
        Correlate every registered feature (or the given subset) with the target.
 
        lags      : in units of `granularity`; lag=1 means feature at t-1 vs decision at t
        group_by  : None | "month" | "hour" | "year" | callable(df)->Series.
                    Correlation is computed within each group (useful to strip
                    seasonality / hour-of-day effects).
        Returns tidy DataFrame: feature, entity, lag, group, method, corr, n
        Features whose kind doesn't fit `entity` are skipped (global fits any).
        """
        tgt = self.target(entity, granularity)
        names = list(features) if features is not None else list(self.features)
        rows = []
 
        for fname in names:
            kind, f = self.features[fname]
            if kind not in ("global", entity):
                continue
            if kind != "global":
                unmatched = set(f["entity"].unique()) - set(tgt["entity"].unique())
                if unmatched:
                    warnings.warn(f"[{fname}] columns with no matching {entity}: "
                                  f"{sorted(unmatched)}")
            for lag in lags:
                m = self._aligned(tgt, fname, granularity, lag)
                if m.empty:
                    continue
                m["_group"] = self._group(m, group_by)
                for (ent, grp), g in m.groupby(["entity", "_group"], observed=True):
                    if len(g) < min_obs or g["value"].nunique() < 2 or g["y"].nunique() < 2:
                        continue  # constant series -> correlation undefined
                    for meth in methods:
                        rows.append(dict(
                            feature=fname, entity=ent, lag=lag, group=grp, method=meth,
                            corr=g["value"].corr(g["y"], method=meth), n=len(g),
                        ))
 
        cols = ["feature", "entity", "lag", "group", "method", "corr", "n"]
        res = pd.DataFrame(rows, columns=cols)
        if res.empty:
            return res
        return res.reindex(res["corr"].abs().sort_values(ascending=False).index).reset_index(drop=True)
 
    @staticmethod
    def _group(m: pd.DataFrame, group_by) -> pd.Series:
        if group_by is None:
            return pd.Series("all", index=m.index)
        if callable(group_by):
            return group_by(m)
        return {"month": m["time"].dt.month,
                "hour": m["time"].dt.hour,
                "year": m["time"].dt.year}[group_by]
 
    # -- convenience ------------------------------------------------------- #
    @staticmethod
    def pivot(res: pd.DataFrame, method: str = "spearman", lag: int = 0) -> pd.DataFrame:
        """feature x entity matrix of correlations (no grouping)."""
        r = res[(res["method"] == method) & (res["lag"] == lag)]
        return r.pivot_table(index="feature", columns="entity", values="corr")