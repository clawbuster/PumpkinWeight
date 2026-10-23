"""Statistics for weight models; repeated photos are grouped by physical pile."""
from __future__ import annotations
import numpy as np
from .features import SampleTable
from .grouping import group_samples


def evaluate_model(model, samples: SampleTable) -> dict:
    """Photo-level errors and empirical interval coverage (not training accuracy)."""
    rows = []
    for s in samples.with_weight():
        e = model.predict(s.features, s.crop)
        rows.append({"file": s.file, "crop": s.crop, "true_g": s.weight_g,
                     "pred_g": e.weight_g, "error_g": e.weight_g - s.weight_g,
                     "low_g": e.low_g, "high_g": e.high_g,
                     "covered": e.low_g <= s.weight_g <= e.high_g
                     if e.low_g is not None and e.high_g is not None else None,
                     "pile_id": s.meta.get("pile_id"), "note": e.note})
    result = _metrics(rows)
    result["per_crop"] = {c: _metrics([r for r in rows if (r["crop"] or "?") == c])
                          for c in sorted({r["crop"] or "?" for r in rows})}
    result["rows"] = rows
    covered = [r["covered"] for r in rows if r["covered"] is not None]
    result["interval_coverage"] = float(np.mean(covered)) if covered else None
    result["n_with_bounds"] = len(covered)
    return result


def _metrics(rows):
    n = len(rows)
    if not n:
        return {"n": 0, "mae_g": None, "mape": None, "bias_g": None, "r2": None}
    err = np.asarray([r["error_g"] for r in rows], dtype=float)
    true = np.asarray([r["true_g"] for r in rows], dtype=float)
    nz = true != 0
    ss_tot = float(np.sum((true - true.mean()) ** 2))
    return {"n": n, "mae_g": float(np.abs(err).mean()), "bias_g": float(err.mean()),
            "mape": float(np.mean(np.abs(err[nz]) / np.abs(true[nz]))) if nz.any() else None,
            "r2": 1 - float(np.sum(err ** 2)) / ss_tot if n > 1 and ss_tot > 0 else None}


def pile_xy(samples, x_key, y_key="weight_g", crop=None):
    """One mean feature/target per pile; correlation of pile means, not individual photos."""
    rows = []
    for group in group_samples(samples.with_weight().for_crop(crop)).values():
        pairs = [(s.get(x_key), s.get(y_key)) for s in group]
        pairs = [(float(x), float(y)) for x, y in pairs
                 if x is not None and y is not None and np.isfinite(x) and np.isfinite(y)]
        if pairs:
            rows.append(np.mean(pairs, axis=0))
    return np.asarray(rows, dtype=float).reshape(-1, 2)


def _statistic(xy, stat):
    x, y = xy[:, 0], xy[:, 1]
    dx, dy = x - x.mean(), y - y.mean()
    xx, yy = float(dx @ dx), float(dy @ dy)
    if xx == 0 or (stat == "r" and yy == 0):
        return float("nan")
    if stat == "slope":
        return float(dx @ dy / xx)
    return float(np.clip(dx @ dy / np.sqrt(xx * yy), -1, 1))


def _summary(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not len(v):
        return None
    lo, hi = np.quantile(v, [0.025, 0.975])
    return {"mean": float(v.mean()), "p025": float(lo), "p975": float(hi),
            "valid_repeats": len(v)}


def learning_curve(samples: SampleTable, x_key: str, sizes: list[int], y_key="weight_g",
                   crop=None, repeats=200, *, seed=0):
    """Stability vs independent PILES. Average repeat photos before sampling.

    r/slope measure the relationship; slope is diagnostic OLS WITH an intercept.
    heldout_mae fits y=x*mean(y/x) to chosen piles and tests on remaining pile means.
    It is not the final single-photo error. Unavailable sizes are explicitly flagged.
    """
    xy = pile_xy(samples, x_key, y_key, crop)
    if len(xy) < 3:
        return None
    if repeats < 1:
        raise ValueError("repeats must be positive")
    rng = np.random.default_rng(seed)
    result = []
    for n in sizes:
        row = {"n": n, "available_piles": len(xy)}
        if n < 3 or n > len(xy):
            row["status"] = "need 3 <= n <= available_piles"
            result.append(row)
            continue
        rs, slopes, errors = [], [], []
        for _ in range(repeats):
            chosen = rng.choice(len(xy), n, replace=False)
            train = xy[chosen]
            rs.append(_statistic(train, "r"))
            slopes.append(_statistic(train, "slope"))
            mask = np.ones(len(xy), dtype=bool)
            mask[chosen] = False
            test = xy[mask]
            positive = train[:, 0] > 0
            if len(test) and positive.any():
                c = np.mean(train[positive, 1] / train[positive, 0])
                errors.append(float(np.abs(c * test[:, 0] - test[:, 1]).mean()))
        row.update(status="ok", r=_summary(rs), slope=_summary(slopes),
                   heldout_mae=_summary(errors))
        result.append(row)
    return result


def bootstrap_ci(samples: SampleTable, x_key: str, y_key="weight_g", crop=None,
                 stat="r", n_boot=1000, *, seed=0):
    """Approximate 95% percentile CI for r or OLS slope of PILE MEANS.

    Sample independent piles with replacement; repeat photos are averaged first.
    This is a CI for a statistic, NOT a prediction interval for a new sample's weight.
    """
    if stat not in ("r", "slope"):
        raise ValueError("stat must be 'r' or 'slope'")
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    xy = pile_xy(samples, x_key, y_key, crop)
    if len(xy) < 3:
        return None
    rng = np.random.default_rng(seed)
    values = [_statistic(xy[rng.integers(0, len(xy), len(xy))], stat) for _ in range(n_boot)]
    valid = np.array([v for v in values if np.isfinite(v)])
    if len(valid) < max(1, n_boot // 2):
        return None
    return tuple(float(v) for v in np.quantile(valid, [0.025, 0.975]))


def propagate(rel_errors: dict[str, float], weight_g: float) -> dict:
    """RSS for independent, zero-mean STANDARD relative uncertainties (0.02=2%).

    Inputs must be contributions to OUTPUT weight, not raw stage tolerances.
    For area=pixels*s**2, use 2*u_s/s for the scale contribution. Correlations
    require covariance terms. Correct biases separately; do not double-count noise.
    """
    values = {k: float(v) for k, v in rel_errors.items()}
    if not np.isfinite(weight_g) or any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError("weight must be finite; uncertainties finite and nonnegative")
    variance = sum(v * v for v in values.values())
    total = float(np.sqrt(variance))
    return {"rel_total": total, "weight_u_g": abs(float(weight_g)) * total,
            "dominant": max(values, key=values.get) if variance else None,
            "budget": {k: v * v / variance if variance else 0.0 for k, v in values.items()}}
