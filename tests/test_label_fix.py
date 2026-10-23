"""Regression checks for bulk correction of a single-crop photo."""
import copy
import json

import pytest
from PIL import Image

from detkit.app import logic as L
from detkit.project import Project
from detkit.weight import image_features


def boxes(n=220):
    return [{"bbox": [(i % 20) * 20 + .25, (i // 20) * 20, (i % 20) * 20 + 15.75,
                      (i // 20) * 20 + 17],
             "class": "pumpkin" if i == 0 else "wheat", "origin": "rfdetr", "score": .946,
             "weight_g": .15 if i == 1 else None, "box_notes": f"seed {i}"} for i in range(n)]


def test_bulk_relabel_preserves_220_boxes_and_weight_features():
    before = boxes()
    snapshot = copy.deepcopy(before)
    after, changed = L.relabel_boxes(before, "pumpkin", ["wheat", "pumpkin"])
    assert changed == 219 and len(after) == 220
    assert before == snapshot
    assert {b["class"] for b in after} == {"pumpkin"}
    assert [b["bbox"] for b in after] == [b["bbox"] for b in before]
    for old, new in zip(before, after):
        assert new["weight_g"] == old["weight_g"] and new["box_notes"] == old["box_notes"]
    assert after[0] == before[0]
    assert all(b["origin"] == "manual" and b["score"] is None for b in after[1:])
    original_features, new_features = image_features(before), image_features(after)
    for key in ("count", "total_area_px", "mean_area_px", "median_area_px", "mean_diag_px"):
        assert original_features[key] == new_features[key]


def test_invalid_target_leaves_input_unchanged_and_repeat_is_idempotent():
    original = boxes(3)
    for target in (None, "", "corn"):
        with pytest.raises(ValueError, match="crop"):
            L.relabel_boxes(original, target, ["wheat", "pumpkin"])
    after, n = L.relabel_boxes(original, "pumpkin", ["wheat", "pumpkin"])
    again, n_again = L.relabel_boxes(after, "pumpkin", ["wheat", "pumpkin"])
    assert n == 2 and n_again == 0 and after == again
    again[0]["bbox"][0] = 999
    assert after[0]["bbox"][0] != 999


def make_project(tmp_path):
    project = Project(tmp_path / "project", classes=["wheat", "pumpkin"])
    project.save()
    project.photos_dir.mkdir()
    for name in ("current.jpg", "other.jpg"):
        Image.new("RGB", (500, 500), "white").save(project.photos_dir / name)
        project.store().upsert_image(name, 500, 500, boxes(4),
                                    {"crop": "pumpkin", "pile_id": name, "total_weight_g": 33.0,
                                     "manual_count": 4}, reviewed=True)
    return project


def test_saving_corrected_photo_preserves_metadata_and_other_photo(tmp_path):
    project = make_project(tmp_path)
    store = project.store()
    before_other = store.get_image("other.jpg")
    before, meta = store.get_image("current.jpg")
    corrected, _ = L.relabel_boxes(before, "pumpkin", project.classes)
    L.save_review(project, "current.jpg", corrected, meta, proposer="rfdetr")
    saved, saved_meta = store.get_image("current.jpg")
    assert len(saved) == len(before) and all(b["class"] == "pumpkin" for b in saved)
    assert [b["bbox"] for b in saved] == [b["bbox"] for b in before]
    for key in ("crop", "pile_id", "total_weight_g", "manual_count"):
        assert saved_meta[key] == meta[key]
    assert store.get_image("other.jpg") == before_other


def test_actual_button_callback_and_save_commit_current_photo_only(tmp_path):
    gr = pytest.importorskip("gradio")
    import pandas as pd
    from detkit.app.ui import build_app, HAVE_EDITOR
    from detkit.schema import IMAGE_FIELDS

    project = make_project(tmp_path)
    demo = build_app(project.workdir)
    relabel = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "relabel_to_crop")
    save = next(fn for fn in demo.fns.values() if getattr(fn.fn, "__name__", "") == "save_only")
    state_component = next(c for c in relabel.inputs if isinstance(c, gr.State))
    table_component = next(c for c in relabel.inputs if getattr(c, "label", None) == "Boxes")
    state = copy.deepcopy(state_component.value)
    current, meta = project.store().get_image("current.jpg")
    preview, size, scale = L.make_preview(project.photos_dir / "current.jpg", project.workdir / "preview")
    form = L.meta_to_form(meta, project.image_fields())
    state.update(file="current.jpg", boxes=current, size=size, scale=scale, preview=str(preview),
                 sig0=json.dumps(current), form0=form, proposer="rfdetr")
    rows = L.boxes_to_rows(current)
    headers = L.table_headers()
    if not HAVE_EDITOR:
        headers += ["x1", "y1", "x2", "y2"]
        rows = [row + box["bbox"] for row, box in zip(rows, current)]
    values = {c: c.value for c in relabel.inputs}
    values[state_component] = state
    values[table_component] = pd.DataFrame(rows, columns=headers)
    by_label = {L.field_label(f): meta.get(f.key) for f in IMAGE_FIELDS}
    by_label["crop / class"] = "pumpkin"
    for c in relabel.inputs:
        if getattr(c, "label", None) in by_label:
            values[c] = by_label[c.label]
    before = project.store().get_image("current.jpg")
    other = project.store().get_image("other.jpg")
    output = dict(zip(relabel.outputs, relabel.fn(values)))
    corrected_state = output[state_component]
    assert all(b["class"] == "pumpkin" for b in corrected_state["boxes"])
    assert len(corrected_state["boxes"]) == 4
    assert project.store().get_image("current.jpg") == before  # explicit Save is still required
    saved_inputs = {c: output.get(c, values.get(c, c.value)) for c in save.inputs}
    save.fn(saved_inputs)
    assert all(b["class"] == "pumpkin" for b in project.store().get_image("current.jpg")[0])
    assert project.store().get_image("other.jpg") == other
