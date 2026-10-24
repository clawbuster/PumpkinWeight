"""Weight models: features -> Estimate. weight = feature x constant per crop, for kernel count
(``CountTimesConstant``, the baseline) or box area (``AreaTimesConstant``). Bounds: student task 3."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG
from .features import SampleTable
from .grouping import group_samples


@dataclass
class Estimate:
    weight_g: float
    low_g: float | None = None       # lower/upper bound of an interval, if the model can give one
    high_g: float | None = None
    note: str = ""


def _validate_pile_weights(groups: dict) -> None:
    """An unchanged physical sample must have one measured mass across its photos."""
    conflicts = []
    for key, group in groups.items():
        weights = [float(s.weight_g) for s in group]
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Measured weights must be finite and nonnegative")
        if max(weights) - min(weights) > 1e-6:
            details = ", ".join(f"{s.file}={float(s.weight_g):g} g" for s in group)
            conflicts.append(f"  {key}: {details}")
    if conflicts:
        raise ValueError(
            "Conflicting measured weights within the same pile_id:\n"
            + "\n".join(conflicts)
            + "\nUse one weight for repeated photos of an unchanged sample. "
              "If seeds were added or removed, assign that sample a different pile_id. "
              "Check the original measurements before recalibrating."
        )


class WeightModel:
    """Interface. Subclass, set ``name``/``description``, implement ``fit`` and ``predict``.

    ``params()``/``set_params()`` hold everything needed to save and restore a fitted model as JSON.
    """
    name = "base"
    description = ""
    implemented = False

    def fit(self, samples: SampleTable) -> "WeightModel":
        raise NotImplementedError

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        raise NotImplementedError

    def params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

    def to_dict(self) -> dict:
        return {"name": self.name, "params": self.params()}

    @staticmethod
    def from_dict(d: dict) -> "WeightModel":
        m = get_model(d["name"])
        m.set_params(d.get("params", {}))
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "WeightModel":
        return WeightModel.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FeatureTimesConstant(WeightModel):
    """weight = feature x constant, one constant per crop (``feature`` is a key of ``image_features``).

    ``constants`` = {"default": c, "<crop>": c}. Subclasses pick the feature; ``config_key`` is where the
    constants live in weight_config.json.
    """
    name = "feature_x_constant"
    feature = "count"
    unit = "g per unit"
    config_key = "constants"
    implemented = True

    def __init__(self, constants: dict[str, float] | None = None):
        self.constants = dict(DEFAULT_CONFIG[self.config_key] if constants is None else constants)
        self.bounds = {}
        self.fit_groups = []
        self.calibration_groups = []

    def fit(self, samples: SampleTable) -> "FeatureTimesConstant":
        """Equal weight per pile: mean of each pile's mean(weight / feature).

        Bounds after fit are EXPLORATORY leave-one-pile-out absolute-error bands.
        They have no claimed coverage probability. Use calibrate_bounds on separate
        piles for split-conformal prediction intervals, then an independent test.
        """
        groups = group_samples(samples.with_weight())
        _validate_pile_weights(groups)
        ratios, by_crop = {}, {}
        for key, group in groups.items():
            for s in group:
                x = float(s.features.get(self.feature, 0))
                if not math.isfinite(x) or x < 0 or not math.isfinite(s.weight_g) or s.weight_g < 0:
                    raise ValueError("features and weights must be finite and nonnegative")
            r = [s.weight_g / s.features[self.feature] for s in group if s.features.get(self.feature, 0) > 0]
            if r:
                ratios[key] = statistics.fmean(r)
            by_crop.setdefault(group[0].crop or "default", []).append(key)
        if not ratios:
            raise ValueError("No positive features with measured weights; check detections and labels")
        self.bounds, self.fit_groups, self.calibration_groups = {}, sorted(groups), []
        for crop, keys in by_crop.items():
            r = [ratios[k] for k in keys if k in ratios]
            if not r:
                continue
            c = statistics.fmean(r)
            self.constants[crop] = c
            scores = []
            for key in keys:
                other = [ratios[k] for k in keys if k != key and k in ratios]
                if len(other) >= 2:
                    heldout_c = statistics.fmean(other)
                    scores.append(max(abs(s.weight_g - heldout_c * s.features.get(self.feature, 0))
                                      for s in groups[key]))
            if len(scores) == len(keys) and len(scores) >= 3:
                xs = [s.features.get(self.feature, 0) for k in keys for s in groups[k]]
                q = sorted(scores)[math.ceil(0.9 * len(scores)) - 1]
                self.bounds[crop] = {"method": "exploratory_lopo", "q_g": q,
                                     "n_piles": len(keys), "constant": c,
                                     "feature_min": min(xs), "feature_max": max(xs)}
        self.constants["default"] = statistics.fmean(ratios.values())
        return self

    def calibrate_bounds(self, samples: SampleTable, alpha=0.1):
        """Split-conformal ABSOLUTE-error bounds from independent, unused piles.

        Use the lexicographically first photo of each pile, preselected without
        looking at errors. Rank=ceil((n+1)*(1-alpha)); insufficient n raises an error.
        Coverage is marginal for comparable, exchangeable new one-photo samples,
        not a promise for every pile size, a new seed variety, or a different setup.
        Never use these calibration piles to tune the model/threshold/procedure.
        """
        if not 0 < alpha < 1 or not self.fit_groups:
            raise ValueError("Fit a model first, and use 0 < alpha < 1")
        samples = samples.with_weight()
        if any(not s.meta.get("pile_id") for s in samples):
            raise ValueError("Every interval-calibration sample needs a globally unique pile_id")
        groups = group_samples(samples)
        _validate_pile_weights(groups)
        overlap = set(groups) & set(self.fit_groups)
        if overlap:
            raise ValueError("Interval-calibration piles overlap fitted piles; use new physical samples")
        scores = {}
        for group in groups.values():
            s = sorted(group, key=lambda s: s.file)[0]
            crop = s.crop or "default"
            if crop not in self.constants:
                raise ValueError(f"No fitted constant for {crop}")
            if not math.isfinite(s.weight_g) or s.weight_g < 0:
                raise ValueError("Calibration weights must be finite and nonnegative")
            e = self.predict(s.features, crop)
            scores.setdefault(crop, []).append(abs(s.weight_g - e.weight_g))
        if not scores:
            raise ValueError("No weighed interval-calibration piles")
        updates = {}
        for crop, values in scores.items():
            rank = math.ceil((len(values) + 1) * (1 - alpha) - 1e-12)
            if rank > len(values):
                raise ValueError(f"Only {len(values)} independent calibration piles for {crop}; "
                                 f"need at least {math.ceil(1 / alpha - 1 - 1e-12)} for finite bounds")
            previous = self.bounds.get(crop, {})
            updates[crop] = {**previous, "method": "split_conformal", "q_g": sorted(values)[rank - 1],
                             "alpha": alpha, "n_piles": len(values), "constant": self.constants[crop]}
        self.bounds.update(updates)
        self.calibration_groups = sorted(set(self.calibration_groups) | set(groups))
        return self

    def predict(self, features: dict, crop: str | None = None) -> Estimate:
        known = crop in self.constants
        if not known and "default" not in self.constants:
            raise ValueError(f"no {self.unit} constant for crop {crop!r}: "
                             f"run `detkit weight calibrate --model {self.name}`")
        c = self.constants[crop] if known else self.constants["default"]
        note = "" if known or crop is None else f"no constant for crop {crop!r}, used default"
        x = float(features.get(self.feature, 0))
        if not math.isfinite(x) or x < 0 or not math.isfinite(c) or c < 0:
            raise ValueError("feature and constant must be finite and nonnegative")
        weight = x * c
        b = self.bounds.get(crop if known else "default")
        if b is None or not math.isclose(c, b["constant"], rel_tol=1e-12, abs_tol=0):
            return Estimate(weight, note="; ".join(filter(None, [note, "bounds unavailable; calibrate first"])))
        if b["method"] == "split_conformal":
            label = f"{100 * (1 - b['alpha']):g}% split-conformal interval; same-procedure samples"
        else:
            label = "exploratory leave-one-pile-out band; coverage not validated"
        if "feature_min" in b and not b["feature_min"] <= x <= b["feature_max"]:
            label += "; outside fitted feature range"
        if x == 0:
            label += "; no detected feature: inspect image for missed seeds"
        return Estimate(weight, max(0.0, weight - b["q_g"]), weight + b["q_g"],
                        "; ".join(filter(None, [note, label])))

    def params(self) -> dict:
        return {"constants": self.constants, "bounds": self.bounds,
                "fit_groups": self.fit_groups, "calibration_groups": self.calibration_groups}

    def set_params(self, params: dict) -> None:
        self.constants = dict(params["constants"])
        self.bounds = dict(params.get("bounds", {}))
        self.fit_groups = list(params.get("fit_groups", []))
        self.calibration_groups = list(params.get("calibration_groups", []))


class CountTimesConstant(FeatureTimesConstant):
    """Baseline: weight = number of detected kernels x grams per kernel of the crop."""
    name = "count_x_constant"
    description = "weight = kernel count x g/kernel per crop (baseline)"
    feature = "count"
    unit = "g/kernel"
    config_key = "constants"


class AreaTimesConstant(FeatureTimesConstant):
    """weight = total box area (px^2) x grams per px^2 of the crop. The constant depends on the camera
    distance (pixels per mm), so it is only valid for photos taken the same way as the calibration photos."""
    name = "area_x_constant"
    description = "weight = total box area (px^2) x g/px^2 per crop"
    feature = "total_area_px"
    unit = "g/px^2"
    config_key = "area_constants"


MODELS: dict[str, type[WeightModel]] = {
    m.name: m for m in (CountTimesConstant, AreaTimesConstant)
}


def get_model(name: str, **kw) -> WeightModel:
    if name not in MODELS:
        raise KeyError(f"unknown weight model {name!r}; available: {sorted(MODELS)}")
    return MODELS[name](**kw)
