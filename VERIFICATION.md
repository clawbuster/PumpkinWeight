# Verification

Verified 23 October 2026 against the course source supplied in this conversation.

Command: `python3 -m pytest tests/test_label_fix.py tests/test_app.py -q -rs --tb=short`

Result: **28 passed**. Existing warnings concern Gradio row-count deprecations and intentional extra-property validation. No failed or skipped tests remained in this run.

The new regression checks exercise:
- 219 wheat-to-pumpkin corrections in a 220-box photo, preserving all geometry, total count, total box area, per-seed weights and notes.
- Rejection of missing/unknown target crops and safe repeat clicks.
- Preservation of photo metadata and of another photo's annotations when saving the corrected photo.
- The actual Gradio button callback, including requiring a later Save, followed by the existing Save callback committing only the current photo.

The existing app checks also ran, including local server/client save, inference with a mocked detector, and draft/refresh behavior. Actual RF-DETR inference and the Windows laptop's browser were not exercised. This patch improves annotation editing; it does not establish improved detector accuracy or weight-prediction accuracy.

Environment: Python 3.12; Gradio 6.30.0.
