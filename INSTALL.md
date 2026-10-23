# Bulk correction of seed labels

The screenshot shows 220 boxes: 1 pumpkin and 219 wheat. The existing app locks the class column in its box table. This patch adds a button to relabel every box in the CURRENT photo to its selected crop, keeping all box positions and the total number of boxes.

## Install on your laptop

1. In the existing app, click **Save** once to preserve your current boxes and metadata, even if some labels are currently wrong.
2. Stop the app with **Ctrl+C** in its terminal.
3. Extract this ZIP. Copy the CONTENTS of `pumpkin_label_fix` into `C:\KU\CNH\manama_data_collection_course-main`. Merge `detkit` and `tests`, replacing the included source files. The two replacement app files are `detkit/app/ui.py` and `detkit/app/logic.py`.
4. Start the app from the main project folder:

```powershell
.\.venv\Scripts\python.exe -m detkit app --workdir runs\pumpkin --ckpt weights\grain_rfdetr_small_v1\final --device cuda
```

5. Open http://127.0.0.1:7860 and hard-refresh with **Ctrl+Shift+R**.

## Correct the current photo

1. Open it in **Review**.
2. Set **crop / class** to **pumpkin** in Photo metadata.
3. Find **Relabel all boxes to photo crop** immediately below the Delete box controls, beneath the box table. Click it once.
4. For the pictured example, the result should be 220 pumpkin boxes, with a status saying that 219 labels changed and all 220 box positions were kept.
5. Inspect the image and click **Save** or **Save & next** to commit the correction. The new button itself only changes the current review/draft, consistent with the existing editing workflow.

Use this action when all retained boxes are intended to mark the selected crop. It does not validate that boxes cover real seeds. Remove background detections or duplicates separately, and add any genuinely missed seeds. Compare the total with your independent manual count.

Existing manually drawn pumpkin boxes are kept. Corrected wheat boxes are marked `manual`, and their old wheat confidence scores are cleared. Coordinates, box numbers, per-seed weights and notes are preserved. Other photos are only changed when you open and correct them yourself.

## Effect on weight prediction

The count model uses the total number of boxes, so changing class labels alone leaves that feature unchanged. The correction provides reviewed reference annotations; it does not retrain RF-DETR or guarantee that future automatic labels will be correct.

In **Inference**, explicitly enter **pumpkin** in the upload table's crop column. If crop is left blank, the app can infer wheat from the dominant detected label and use that crop's weight constant. Keeping crop explicit ensures that your pumpkin calibration is selected.

Judge final weight accuracy on fresh AUTOMATIC detections. Reference boxes you corrected by hand are useful for measuring detection errors, but they do not represent what the automatic system can do unaided.

## Verify the patch

Run from the main project folder:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_label_fix.py -q
```

The regression checks cover the 219-of-220 relabelling case, unchanged geometry/count/area, preserved weights/notes, invalid crop handling, repeated clicks, current-photo-only persistence, and the actual app callback followed by Save. See VERIFICATION.md for the checks completed while preparing this patch.
