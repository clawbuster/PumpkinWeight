"""Regression checks for grouping, leakage prevention and deployed persistence."""
import copy
import math
import numpy as np
import pytest
from detkit.weight import (Sample, SampleTable, CountTimesConstant, WeightModel,
                           bootstrap_ci, learning_curve, propagate, load_config, save_config)
from detkit.cmd_weight import configured_model
from detkit.app.inference import load_weight_model, estimate_row
from detkit.cli import main
from detkit.project import Project


def sample(i, prefix="A", count=None):
    n = i + 10 if count is None else count
    return Sample(f"{prefix}{i:03}_01.jpg", "pumpkin", {"count": n, "total_area_px": n * 400},
                  (i + 10) * (0.12 + (i % 3) * 0.002), {"pile_id": f"{prefix}{i:03}"})


def test_repeats_do_not_increase_fit_weight_or_bootstrap_sample_size():
    t = SampleTable([sample(i) for i in range(12)])
    repeated = SampleTable(t.rows + [copy.deepcopy(t.rows[0]) for _ in range(30)])
    assert CountTimesConstant().fit(t).constants == CountTimesConstant().fit(repeated).constants
    assert bootstrap_ci(t, "count", n_boot=100) == bootstrap_ci(repeated, "count", n_boot=100)
    curve = learning_curve(repeated, "count", [5, 12, 30], repeats=10)
    assert curve[-1]["status"] != "ok" and curve[-1]["available_piles"] == 12
    assert curve[1]["heldout_mae"] is None


def test_bounds_round_trip_and_edit_invalidates_old_uncertainty(tmp_path):
    model = CountTimesConstant().fit(SampleTable([sample(i) for i in range(12)]))
    cfg = load_config(tmp_path)
    cfg["constants"] = model.constants
    cfg["fitted_models"] = {model.name: model.to_dict()}
    save_config(tmp_path, cfg)
    restored = load_weight_model(tmp_path, model.name, load_config(tmp_path))
    assert restored.predict({"count": 15}, "pumpkin") == model.predict({"count": 15}, "pumpkin")
    assert restored.predict({"count": 15}, "pumpkin").low_g is not None
    cfg["constants"]["pumpkin"] += 0.01
    assert configured_model(cfg, model.name).predict({"count": 15}, "pumpkin").low_g is None


def test_separate_calibration_exact_rank_and_leakage_guard():
    fitted = SampleTable([sample(i) for i in range(12)])
    model = CountTimesConstant().fit(fitted)
    with pytest.raises(ValueError, match="overlap"):
        model.calibrate_bounds(fitted)
    with pytest.raises(ValueError, match="at least"):
        model.calibrate_bounds(SampleTable([sample(i, "B") for i in range(8)]))
    fresh = SampleTable([sample(i, "B") for i in range(9)])
    errors = [abs(s.weight_g - model.predict(s.features, s.crop).weight_g) for s in fresh]
    model.calibrate_bounds(fresh, alpha=0.1)
    assert model.bounds["pumpkin"]["q_g"] == pytest.approx(max(errors))
    restored = WeightModel.from_dict(model.to_dict())
    assert restored.calibration_groups == model.calibration_groups
    assert "90%" in restored.predict({"count": 20}, "pumpkin").note


def test_total_detector_failure_is_scored_in_interval_calibration():
    model = CountTimesConstant().fit(SampleTable([sample(i) for i in range(12)]))
    fresh = [sample(i, "B") for i in range(9)]
    fresh[0].features["count"] = 0
    model.calibrate_bounds(SampleTable(fresh), alpha=0.1)
    e = model.predict({"count": 0}, "pumpkin")
    assert e.high_g >= fresh[0].weight_g and e.low_g == 0


def test_uncertainty_budget_math_and_invalid_inputs():
    result = propagate({"count": 0.03, "constant": 0.04}, 10)
    assert result["rel_total"] == pytest.approx(0.05)
    assert result["budget"]["constant"] == pytest.approx(0.64)
    assert propagate({}, 10)["weight_u_g"] == 0
    with pytest.raises(ValueError):
        propagate({"count": -0.1}, 10)


def test_calibrate_cli_survives_new_app_model_load(tmp_path):
    project = Project(tmp_path, classes=["pumpkin"])
    project.save()
    for i in range(5):
        s = sample(i)
        boxes = [{"bbox": [j * 20, 0, j * 20 + 10, 20], "class": "pumpkin"}
                 for j in range(s.features["count"])]
        project.store().upsert_image(s.file, 500, 500, boxes,
                                    {"crop": "pumpkin", "pile_id": s.meta["pile_id"],
                                     "total_weight_g": s.weight_g}, reviewed=True)
    assert main(["weight", "calibrate", "--workdir", str(tmp_path), "--reviewed-only"]) == 0
    cfg = load_config(tmp_path)
    model = load_weight_model(tmp_path, "count_x_constant", cfg)
    row = estimate_row("new.jpg", [{"bbox": [0, 0, 10, 20], "label": "pumpkin"}],
                       {"crop": "pumpkin"}, model, cfg)
    assert row["low_g"] is not None and row["high_g"] is not None
