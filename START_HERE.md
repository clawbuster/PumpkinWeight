# Pumpkin-seed weight project: start here

This companion kit is for the uploaded CNH `manama_data_collection_course-main` starter. It completes the statistical tasks and connects saved prediction bounds to the existing app. It is not a trained pumpkin model: no seed photographs or measured weights were supplied. Real accuracy must be measured on your samples.

## 1. What you are building

The input X is a photograph; the target y is the mass in grams of ALL seeds visible in that photograph. The supplied pipeline is:

**Photo -> pretrained RF-DETR detector -> seed count / summed bounding-box area -> estimated grams.**

Two simple models already exist:

- `count_x_constant`: detected count multiplied by a calibrated grams-per-seed constant.
- `area_x_constant`: summed bounding-box area in pixels multiplied by a calibrated grams-per-pixel-squared constant.

For an illustrative example only, if a constant were 0.12 g/seed, 50 detected seeds would predict 6 g. This is NOT a measured constant for your seeds. The starter's default constants are NOT calibrated for pumpkin; calibrate before using any displayed weight.

Detector retraining is optional. The assignment focuses on collection design, measured errors, justified uncertainty, reproducibility and a teammate's independent repeat. The README explicitly says the grade is not based on detector accuracy. The presentation is due before the 26 October 2026 session; the PDF does not specify an exact submission time. Check Toledo for the submission details.

## 2. Decide the team's roles before combining data

The course's task 4 requires the second collector to apply the first collector's model to new samples WITHOUT refitting first. If your partner's 15 photographs are usable calibration data, they can form an initial fitting dataset. Your first new samples can be the independent repeat. Do not immediately mix them into the fitted model.

Fifteen photos might represent 15 different piles, or five piles photographed three times. Those have different evidential value. Check `pile_id`, photographs, measured weights, camera setup, and whether the 15 photos were actually used to retrain a detector or merely calibrate a constant. A saved detector checkpoint is separate from a saved weight calibration.

Use globally unique IDs, such as `A001` for the first collector and `J001` for Joseph. The SAME physical sample keeps the SAME ID when rearranged or photographed again. Changing the number of seeds creates a different sample. Nested samples and reuse of the same small seed stock introduce dependence; do not present them as fully independent new biological samples. If feasible, set aside a fresh seed portion for the other collector.

## 3. Collect a five-sample pilot first

Use a table or other flat surface with a clean, matte, plain sheet or board. Choose a background that contrasts with the actual seeds. A patterned floor makes detection harder and is awkward to reproduce at the demo. Use the same board at the demo.

1. Choose a representative handful, including ordinary size variation. Assign its pile ID.
2. Put an empty container on a level, stable balance and tare it. Weigh ONLY the seeds. Record grams, the scale's displayed increment and any unstable readings. For several pilot piles, remove and replace the sample three times; record all readings separately and use their mean as the repeated photos' label.
3. Transfer exactly those seeds to the board. Count them by hand for the pilot. Do not lose any, add any or leave some in the container.
4. Spread them in one layer with visible gaps. No stacks, touching clusters, hidden seeds or cropped-off seeds. Keep every seed within the photograph.
5. Photograph from directly above, with the phone parallel to the board, sharp focus and diffuse light. Use a normal still-photo mode and the main camera; avoid portrait blur, digital zoom, flash glare and your own shadow. Pick a reproducible height based on the pilot (around 30-40 cm is only a starting experiment, not a required value).
6. Keep board framing, height, camera mode and output resolution consistent. Include a ruler or known-length marker in the same plane, away from the seeds, to document scale. Do not include a readable scale display or weight label in the model image.
7. Save the original photo, e.g. `A001_01.jpg`. Take a second photo after moving the phone away and setting it up again; `A001_02.jpg` still has the same pile ID and weight. For some piles, also rearrange the seeds without changing the sample.
8. Check detection overlays now, before collecting dozens of photos. Compare automatic count to the manual count; identify missed seeds, duplicate boxes and background detections. If you change the setup, record a protocol version and recollect incompatible data.

Prefer JPEG/PNG originals supported by the starter, transferred as files. Do not use screenshots or messaging-app compressed copies. If converting HEIC, normalize orientation and convert consistently once, before import.

### Scale resolution matters

A 1 g display increment is coarse for a 5 g sample: one increment is 20% of its mass. That is not a full uncertainty calculation, but it shows why small targets can be unreliable. A 0.01 g or 0.1 g balance is more useful, subject to actual accuracy and repeatability. With a 0.1 g increment, a 10 g sample has a 1% increment-to-mass ratio. Choose sample sizes accordingly. Repeated identical readings do not prove zero uncertainty: rounding and calibration bias remain. If rounding to increment d is approximately uniform, its standard uncertainty contribution is d/sqrt(12), not zero.

Weighing before or after the photograph both work if exactly the same seeds are included; the course describes photographing then weighing. Follow one written order consistently.

## 4. Expand only after the pilot works

A practical planning target, NOT a CNH requirement or an accuracy guarantee:

| Set | Suggested independent piles | Use |
|---|---:|---|
| Fitting / model selection | 25-40 | Compare count and area; fit the constant; includes suitable existing partner samples |
| Optional interval calibration | 10-20 | Fresh, unused piles for stronger prediction intervals |
| Teammate repeat / final test | 10-15 | Frozen model, new collector, no fitting or threshold tuning |

Cover small, medium and large sample sizes within the range you can photograph clearly and weigh reliably. Include ordinary seed-size and orientation variation. Randomize collection order. Rephotograph 5-10 selected fitting piles three times to measure photographic repeatability. Extra photos help measure repeatability; extra physical samples help establish generalization.

Use the learning curve to judge whether more data still improves stability. If only 15 independent samples are feasible, report the limited evidence honestly; a simple constant can still be fitted. Do not manufacture certainty by treating augmentations or repeat photographs as additional independent samples.

Freeze the model choice, detector threshold and photography procedure BEFORE using interval-calibration or final-test data. If you change the method after seeing the test results, call that development data and collect a fresh test.

## 5. Install the companion code on Windows

1. Extract the original uploaded course ZIP.
2. Extract this companion ZIP somewhere else.
3. Copy this companion folder's CONTENTS into the original project's folder, the one containing `pyproject.toml`, `README.md` and `detkit`. Merge folders and replace the included source files. Keep the original downloaded ZIP as your reference.
4. Open PowerShell in that project folder. If your partner already has a working environment, reuse it. Otherwise install `uv` using the original README's Windows instructions, then run:

```powershell
uv sync --python 3.12 --extra cpu --extra app --extra dev
.venv\Scripts\python.exe -m detkit doctor
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\hf.exe download niru-5/manama_course_rfdetr_grain_detection --local-dir weights\grain_rfdetr_small_v1\final
```

The detector checkpoint is not in either ZIP; it needs this download. No Hugging Face token is needed for the public baseline. CPU inference may take time. Use the starter's GPU option only if the laptop has compatible NVIDIA hardware. Preserve the supplied package pins; if installation fails, retain the complete error for diagnosis instead of changing random versions. This environment's offline checks did not exercise the full Windows installation or real RF-DETR inference.

Commands below use the virtual environment's Python explicitly, so no activation or PowerShell execution-policy change is needed.

## 6. Import and label your pilot

Make a folder `my_photos` containing the photographs. Create a project:

```powershell
.venv\Scripts\python.exe -m detkit init --workdir runs\fit --classes corn,peanuts,popcorn_corn,pumpkin,sunflower,wheat --flow cpu --tile 1500 --proposer-ckpt weights\grain_rfdetr_small_v1\final
```

Use `weights_template.csv` as a header-only starting file. Enter one row per photograph. For example (illustrative numbers ONLY):

```csv
file,crop,pile_id,total_weight_g,manual_count
A001_01.jpg,pumpkin,A001,6.20,50
A001_02.jpg,pumpkin,A001,6.20,50
A002_01.jpg,pumpkin,A002,10.15,82
```

Use the exact filename and a decimal point; record grams, not kilograms. Leave unknown manual counts blank. Keep your actual measurements in the file, never these illustrative values.

```powershell
.venv\Scripts\python.exe -m detkit import-photos --workdir runs\fit --images my_photos --meta weights_template.csv
.venv\Scripts\python.exe -m detkit app --workdir runs\fit --ckpt weights\grain_rfdetr_small_v1\final
```

Open `http://127.0.0.1:7860`. In Review, choose each photo, confirm `crop=pumpkin`, use the RF-DETR proposals, check metadata and detections, and press **Save**. You can correct boxes for reference measurements. Record the automatic result separately before correction. Only saved/reviewed photos are used by this companion workflow.

The starter's reference validation threshold is 0.38. Its app can default to 0.5 when no evaluation report exists, so set it deliberately. Start with 0.38, then use your development data and the provided `detkit eval` workflow to select a threshold; freeze it for all later predictions. Threshold, tile size, overlap, downscale and crop/ROI must match your deployment settings. The commands below use 0.38, tile 1500, overlap 0.2, no downscale and no ROI. Replace the threshold everywhere if you select another.

Add useful extra properties such as collector, phone model, camera height and protocol version in Review. Define those properties BEFORE importing extra CSV columns: the starter ignores unknown columns. Keep repeated scale readings in `collection_log_template.csv` as a separate experiment log.

**Starter gotcha:** re-importing a file already registered does not update its metadata, even with a changed CSV. Update the values in Review and Save, or import a correctly prepared dataset into a new workdir. Do not assume re-import changed the weights.

## 7. Compare models using automatic detections

After reviewing/saving labels, generate a fresh, UNCORRECTED prediction file. These detections match what the model must use at the demo; corrected boxes remain useful for diagnosis.

```powershell
.venv\Scripts\python.exe -m detkit predict --model weights\grain_rfdetr_small_v1\final --images runs\fit\photos --tile 1500 --overlap 0.2 --score 0.38 --out runs\fit\automatic
.venv\Scripts\python.exe pumpkin_workflow.py analyse --workdir runs\fit --predictions runs\fit\automatic\predictions.json
```

Read `runs/fit/reports/pumpkin_analysis/analysis.json` and its small CSV tables. The analysis shows feature correlations with bootstrap CIs, grouped learning curves, repeat-photo variability, manual-versus-automatic count errors and leave-one-pile-out model errors. Count and area are compared by holding each physical pile out of fitting in turn. Each pile has equal weight in the comparison.

Start with count as the baseline. Choose area only if evidence supports it under the actual demo setup. Correlation alone is not enough: compare error in grams and systematic bias. Area here means axis-aligned BOUNDING-BOX area, not the true seed outline. Rotation changes box area; distance, camera sensor, resolution and cropping change pixel scale. Fixed phone height alone does not match scale across different phones. A known-length marker permits mm-per-pixel measurements, but the shipped area model and app do not automatically use that marker or `total_area_mm2`. Physical-area deployment would require another implementation. Count is less sensitive to pixel scale, although detection quality still changes with scale.

To fit and SAVE the selected count model:

```powershell
.venv\Scripts\python.exe -m detkit weight calibrate --workdir runs\fit --model count_x_constant --predictions runs\fit\automatic\predictions.json --reviewed-only
```

For the area alternative, use `--model area_x_constant`. This saves constants AND uncertainty state in `runs/fit/weight_config.json`, plus a model JSON in `runs/fit/models`. Calibration sets that model as the app's default. Do not run this on the teammate's final-test samples before reporting the independent evaluation.

## 8. Prediction bounds and error propagation

The fitted model initially displays an **exploratory** band based on the 90th percentile of leave-one-pile-out errors, using the worst repeat-photo error per pile. That is a descriptive error band, NOT a validated 90% prediction interval. The app's note says this. With fewer than three suitable fitting piles, bounds are unavailable.

For stronger intervals, create a separate `runs/bounds` project following the same import/review/predict steps, with fresh unused piles and globally unique IDs. Keep the detector and point model frozen. Then:

```powershell
.venv\Scripts\python.exe pumpkin_workflow.py bounds --model-workdir runs\fit --samples-workdir runs\bounds --predictions runs\bounds\automatic\predictions.json --alpha 0.1
```

This calibrates an absolute-error split-conformal interval. It uses the lexicographically first photo (e.g. `_01`) of each pile, selected without looking at its error. For n piles, it uses the error at sorted rank ceil((n+1)*(1-alpha)). At least 9 independent calibration piles are needed for a finite 90% interval; at least 19 for 95%. Those are mathematical minima, not recommendations that the result will be tight or stable. Aim for more when feasible. Coverage is marginal for comparable, exchangeable future samples, not guaranteed for every seed size or after a camera/seed-distribution shift. Never tune the method using this interval-calibration set. Bounds calibrated this way target the measured balance label; scale bias is still a separate concern for true physical mass.

The companion refuses fitting/calibration overlap by pile ID. Correct IDs are essential; renamed repeats are still leakage. Re-fitting the model clears its old bounds, and manually changing a constant makes its previous bounds unavailable. Recalibrate bounds after any such change.

For task 3's bottom-up uncertainty budget, `propagate` implements:

`relative_standard_uncertainty = sqrt(u1**2 + u2**2 + ...)`

Feed it measured, independent standard uncertainties as fractions, not percentages or arbitrary guesses. For weight=count*constant, count uncertainty contributes u_count/count and constant uncertainty contributes u_constant/constant. For area=pixel_area*scale**2, the relative scale contribution is doubled. Correct systematic biases separately; correlated sources require covariance terms. Do not count detector error both inside an end-to-end residual estimate and again as a separate contribution. The task asks you to compare these bottom-up estimates with the model's observed spread; disagreement is a finding to investigate.

## 9. Independent teammate test and CNH demo

Create `runs/teammate` from NEW samples following only the written procedure, using the same save/review step for metadata and a separate AUTOMATIC predictions file. Do not correct the predictions used to evaluate the deployed pipeline. Then:

```powershell
.venv\Scripts\python.exe pumpkin_workflow.py evaluate --model-workdir runs\fit --samples-workdir runs\teammate --predictions runs\teammate\automatic\predictions.json
```

It loads the frozen saved model, refuses pile overlap, and writes per-photo errors, MAE, bias, MAPE and interval coverage. MAPE in JSON is a fraction: 0.05 means 5%. It also reports equal-pile-weighted errors and first-photo-per-pile coverage. Small test sets have substantial uncertainty; report the number of independent piles and the tested mass range alongside metrics. Repeated photos of the same pile do not provide independent coverage trials.

Task 4 also asks whether the constant and preferred feature agree in the other collector's sample. AFTER recording the frozen-model evaluation, analyse and fit the teammate dataset separately for comparison; retain the first model and its test results. A later pooled refit is a different model requiring fresh validation. There is no need to 'top up' a neural network to fit these simple constants.

Before the demo, reopen the app and check that constants and bounds persisted. Test an upload from a second phone on the same network, using fresh development samples. CNH uses its own phone following your procedure, so document board framing, scale, lighting and sample spreading sufficiently for that phone. A count baseline generally makes cross-phone scale differences easier to manage.

```powershell
.venv\Scripts\python.exe -m detkit app --workdir runs\fit --ckpt weights\grain_rfdetr_small_v1\final --host 0.0.0.0
```

Use `ipconfig` to find the laptop's IPv4 address; open `http://<IPv4>:7860` on the phone. Use a shared hotspot if campus Wi-Fi isolates devices. Allow Python on the private network if Windows asks. In Inference choose pumpkin, the selected model and exactly the frozen detector settings. At the real demo leave measured weight blank until the prediction is recorded; then compare with the balance. Stop the app with Ctrl+C after use.

Back up the full workdir (photos, annotations, metadata, project.json, weight_config.json and models), the downloaded detector checkpoint, your code, protocol and environment lockfile. `runs/` is git-ignored in the starter, so a GitHub code commit alone does not back up those files. Never commit tokens. Do not claim detection has been improved by retraining unless you actually retrained and evaluated it.

## 10. What this kit changes and what you must explain

- `detkit/weight/stats.py`: grouped bootstrap CIs, learning curves, uncertainty propagation, evaluation with bounds.
- `detkit/weight/grouping.py`: shared pile grouping.
- `detkit/weight/models.py`: equal-pile calibration, exploratory bounds, optional held-out conformal bounds, complete serialization.
- `detkit/cmd_weight.py`: save/load the full model, including bounds, while preserving the existing CLI.
- `detkit/app/inference.py`: restore full saved parameters in the existing Inference tab.
- `pumpkin_workflow.py`: real-data analysis, interval calibration and frozen evaluation.
- `tests/test_pumpkin_companion.py`: regression checks for grouping, persistence, overlap prevention and uncertainty math.

The existing app still contains some generic 'student TODO' explanatory text; the implemented functions and prediction notes are authoritative. The original tests are preserved. Read `VERIFICATION.md` for the actual checks run here. Software checks do not establish predictive accuracy.

You should be able to explain why count/area is useful, how the constant is fitted, why repeated photos are grouped, why an error band is not the same as a confidence interval for the mean, and what evidence supports your stated accuracy. The assignment explicitly permits AI assistance while making you responsible for the process and outcome.

References: uploaded assignment PDF (project and deliverables); original `README.md`, `docs/STUDENT_TASKS.md`, `docs/DATA_COLLECTION.md`, `docs/WEIGHT.md`, `docs/APP.md`; scikit-learn grouped cross-validation documentation at https://scikit-learn.org/stable/modules/cross_validation.html ; Angelopoulos and Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*, https://arxiv.org/abs/2107.07511 .
