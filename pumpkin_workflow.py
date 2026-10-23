"""Companion analysis, interval calibration, and frozen-model evaluation.

Run from the original project root after copying this kit's files into it.
No detector training or invented measurements. See START_HERE.md.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np
from detkit.project import Project
from detkit.cmd_weight import configured_model
from detkit.weight import (SampleTable, get_model, load_config, save_config,
                           bootstrap_ci, learning_curve, evaluate_model)
from detkit.weight.grouping import group_samples
from detkit.weight.stats import pile_xy, _statistic


def read_samples(workdir, predictions=None):
    project = Project.load(workdir)
    table = SampleTable.from_project(project, reviewed_only=True,
                                    use="predictions" if predictions else "reviewed",
                                    predictions_path=predictions).with_weight().for_crop("pumpkin")
    if not len(table):
        raise ValueError("No SAVED/reviewed pumpkin photos with weights. Check filenames and crop.")
    if any(not s.meta.get("pile_id") for s in table):
        raise ValueError("Enter a globally unique pile_id for every photo, e.g. A001, J001.")
    for group in group_samples(table).values():
        weights = [s.weight_g for s in group]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Weights must be finite and nonnegative.")
        if max(weights) - min(weights) > 1e-6:
            raise ValueError(f"{group[0].meta['pile_id']}: repeat photos have different weight labels. "
                             "Use the same measured mean weight for the unchanged sample; log repeat readings separately.")
    return table


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_metrics(model, table):
    """Give each independent pile equal weight, even with unequal repeat counts."""
    per_pile = []
    for group in group_samples(table).values():
        result = evaluate_model(model, SampleTable(group))
        per_pile.append(result)
    return {"n_piles": len(per_pile),
            "pile_macro_mae_g": float(np.mean([r["mae_g"] for r in per_pile])),
            "pile_macro_bias_g": float(np.mean([r["bias_g"] for r in per_pile]))}


def analyse(args):
    table = read_samples(args.workdir, args.predictions)
    groups = group_samples(table)
    n = len(groups)
    if n < 3:
        raise ValueError("Need at least 3 distinct piles for this analysis; start with a small pilot.")
    features = ["count", "total_area_px", "mean_area_px", "median_area_px", "mean_diag_px"]
    if all("total_area_mm2" in s.features for s in table):
        features.append("total_area_mm2")
    comparisons = []
    curves = {}
    for feature in features:
        xy = pile_xy(table, feature)
        r = _statistic(xy, "r")
        ci = bootstrap_ci(table, feature)
        comparisons.append({"feature": feature, "r_pile_means": r if np.isfinite(r) else None,
                            "r_ci_low": ci[0] if ci else None, "r_ci_high": ci[1] if ci else None})
        sizes = sorted(set([k for k in (3, 5, 10, 15, 20, 30, 40) if k < n] + [n]))
        curves[feature] = learning_curve(table, feature, sizes)
    models = []
    for name in ("count_x_constant", "area_x_constant"):
        errors, biases = [], []
        for key, group in groups.items():
            train = SampleTable([s for k, ss in groups.items() if k != key for s in ss])
            model = get_model(name).fit(train)
            result = evaluate_model(model, SampleTable(group))
            errors.append(result["mae_g"])
            biases.append(result["bias_g"])
        models.append({"model": name, "lopo_pile_macro_mae_g": float(np.mean(errors)),
                       "lopo_pile_macro_bias_g": float(np.mean(biases))})
    repeatability = []
    for key, group in groups.items():
        if len(group) < 2:
            continue
        repeatability.append({"pile_id": group[0].meta["pile_id"], "photos": len(group),
                              "count_sd": float(np.std([s.features["count"] for s in group], ddof=1)),
                              "area_px_sd": float(np.std([s.features["total_area_px"] for s in group], ddof=1))})
    detector_errors = [{"file": s.file, "pile_id": s.meta["pile_id"],
                        "detected": s.features["count"], "manual_count": s.meta["manual_count"],
                        "count_error": s.features["count"] - s.meta["manual_count"]}
                       for s in table if s.meta.get("manual_count") is not None]
    result = {"photos": len(table), "independent_piles": n,
              "feature_source": "automatic predictions" if args.predictions else "reviewed boxes",
              "features": comparisons, "model_comparison": models, "learning_curves": curves,
              "repeatability": repeatability, "count_error": detector_errors,
              "interpretation": "Exploratory model selection, not an independent test. Bootstrap CIs use pile means. "
                                "Learning-curve heldout errors use pile means; LOPO model errors use individual photos "
                                "and then equal weighting per pile. Final deployment needs raw automatic predictions."}
    out = Path(args.out or Path(args.workdir) / "reports/pumpkin_analysis")
    write_json(out / "analysis.json", result)
    write_csv(out / "feature_correlations.csv", comparisons)
    write_csv(out / "model_comparison.csv", models)
    write_csv(out / "repeatability.csv", repeatability)
    write_csv(out / "count_errors.csv", detector_errors)
    print(json.dumps({"photos": len(table), "piles": n, "models": models, "report": str(out)}, indent=2))


def bounds(args):
    table = read_samples(args.samples_workdir, args.predictions)
    cfg = load_config(args.model_workdir)
    model = configured_model(cfg, args.model or cfg["model"])
    model.calibrate_bounds(table, alpha=args.alpha)
    cfg.setdefault("fitted_models", {})[model.name] = model.to_dict()
    save_config(args.model_workdir, cfg)
    model_dir = Path(args.model_workdir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / f"{model.name}.json")
    print(json.dumps(model.bounds, indent=2))


def evaluate(args):
    table = read_samples(args.samples_workdir, args.predictions)
    cfg = load_config(args.model_workdir)
    model = configured_model(cfg, args.model or cfg["model"])
    if not model.fit_groups:
        raise ValueError("No saved fitted model; calibrate using the companion code first.")
    used = set(model.fit_groups) | set(model.calibration_groups)
    if used & set(group_samples(table)):
        raise ValueError("Test piles overlap fitting or interval-calibration piles. Use new physical samples.")
    result = evaluate_model(model, table)
    result.update(macro_metrics(model, table))
    first_photos = SampleTable([sorted(g, key=lambda s: s.file)[0] for g in group_samples(table).values()])
    first = evaluate_model(model, first_photos)
    result["first_photo_per_pile"] = {k: v for k, v in first.items() if k not in ("rows", "per_crop")}
    result["evaluation"] = "Frozen model; automatic predictions; test data not used to fit or calibrate intervals."
    out = Path(args.out or Path(args.samples_workdir) / "reports/frozen_evaluation")
    write_json(out / "evaluation.json", result)
    write_csv(out / "per_photo.csv", result["rows"])
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "per_crop")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("analyse", help="correlations, grouped learning curves, LOPO model comparison")
    p.add_argument("--workdir", required=True)
    p.add_argument("--predictions", help="raw predictions.json; omit only for reviewed-box diagnosis")
    p.add_argument("--out")
    p.set_defaults(fn=analyse)
    for name, fn in (("bounds", bounds), ("evaluate", evaluate)):
        p = sub.add_parser(name)
        p.add_argument("--model-workdir", required=True)
        p.add_argument("--samples-workdir", required=True)
        p.add_argument("--predictions", required=True, help="raw predictions from the frozen detector settings")
        p.add_argument("--model", choices=["count_x_constant", "area_x_constant"])
        if name == "bounds":
            p.add_argument("--alpha", type=float, default=0.1)
        else:
            p.add_argument("--out")
        p.set_defaults(fn=fn)
    args = parser.parse_args()
    try:
        args.fn(args)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
