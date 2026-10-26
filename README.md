# Course project: estimate the weight of grain from a photo

> Photo of grain  ->  count / pixels  ->  weight (g)

You get a working grain detector (RF-DETR) and a toolkit (`detkit`) to collect photos and turn detections into
a weight. Each group works on one crop. The course is about **designing the data-collection process** so that the provided model can be
used to estimate weight: plan it, identify the sources of error, and measure them where you can. You are graded on
how you **collect data, measure uncertainty and justify your conclusions**, not on detector accuracy.

Retraining the detector is possible (`detkit train`) but optional and not part of the course.

## Links
- Code: https://github.com/niru-5/manama_data_collection_course
- Dataset: https://huggingface.co/datasets/niru-5/manama_course_dataset_grain_detection
- Baseline model: https://huggingface.co/niru-5/manama_course_rfdetr_grain_detection

## Linux/macOS vs Windows
Commands are given twice: **Linux/macOS** (bash) and **Windows** (PowerShell). The differences:
* Path separators: Linux/macOS use `/`, Windows uses `\` (`runs/mine` vs `runs\mine`).
* Activating the virtual environment: `source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on
  Windows (works in PowerShell and cmd; see below).
* A long command continues on the next line with `\` (bash) or `` ` `` (PowerShell).

**Activate the virtual environment in every new terminal.** All commands below (`python`, `hf`, `pytest`) assume it
is active. Run the activate command at least once in each terminal you open, from the repository folder; the
prompt then starts with `(.venv)`. A new terminal (or a closed one) needs it again. If PowerShell refuses to run
the activate script ("running scripts is disabled"), run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once,
then try again.

The docs in `docs/` show the Linux/macOS form only; translate it the same way on Windows. `$W` there stands for your
workdir (e.g. `runs/mine`). Alternative on Windows: use WSL2 (Ubuntu) and follow the Linux commands as written.

## Quick start

**Linux/macOS**
```bash
# 1. install (ONE of cpu / gpu), then activate the environment
curl -LsSf https://astral.sh/uv/install.sh | sh                 # uv, once
uv sync --extra cpu --extra app --extra dev                      # or: --extra gpu
source .venv/bin/activate                                        # once in every new terminal
python -m detkit doctor
python -m pytest -q                                              # environment check (offline)

# 2. download the baseline model (public, no token needed)
hf download niru-5/manama_course_rfdetr_grain_detection --local-dir weights/grain_rfdetr_small_v1/final

# 3. project folder (one workdir = one batch of photos)
python -m detkit init --workdir runs/mine --classes corn,peanuts,popcorn_corn,pumpkin,sunflower,wheat \
    --flow cpu --tile 1500 --proposer-ckpt weights/grain_rfdetr_small_v1/final    # --flow cpu | gpu | hf

# 4. add photos, then open the app (http://127.0.0.1:7860)
python -m detkit import-photos --workdir runs/mine --images my_photos/ --crop wheat
python -m detkit app --workdir runs/mine --ckpt weights/grain_rfdetr_small_v1/final
```

**Windows (PowerShell)**
```powershell
# 1. install (ONE of cpu / gpu; gpu needs an NVIDIA driver), then activate the environment
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # uv, once
uv sync --extra cpu --extra app --extra dev                      # or: --extra gpu
.venv\Scripts\activate                                           # once in every new terminal
python -m detkit doctor
python -m pytest -q                                              # environment check (offline)

# 2. download the baseline model (public, no token needed)
hf download niru-5/manama_course_rfdetr_grain_detection --local-dir weights\grain_rfdetr_small_v1\final

# 3. project folder (one workdir = one batch of photos)
python -m detkit init --workdir runs\mine --classes corn,peanuts,popcorn_corn,pumpkin,sunflower,wheat `
    --flow cpu --tile 1500 --proposer-ckpt weights\grain_rfdetr_small_v1\final    # --flow cpu | gpu | hf

# 4. add photos, then open the app (http://127.0.0.1:7860)
python -m detkit import-photos --workdir runs\mine --images my_photos\ --crop wheat
python -m detkit app --workdir runs\mine --ckpt weights\grain_rfdetr_small_v1\final
```
A Hugging Face token (`HF_TOKEN=...` in `.env`) is only needed for pushing a model to the Hub (optional). Never commit it.

## Data and baseline model (Hugging Face Hub, public: no token needed)
Dataset: 6 classes, COCO format; read its README. Then try the baseline (downloaded in Quick start, step 2) on a
test photo (score 0.38 = best-F1 threshold chosen on validation).

**Linux/macOS**
```bash
hf download niru-5/manama_course_dataset_grain_detection --repo-type dataset --local-dir ../data/grain_detection_public
python -m detkit predict --model weights/grain_rfdetr_small_v1/final \
    --images ../data/grain_detection_public/photos/test/corn_0007.jpg \
    --tile 1500 --overlap 0.2 --score 0.38 --out runs/first_predictions
```

**Windows (PowerShell)**
```powershell
hf download niru-5/manama_course_dataset_grain_detection --repo-type dataset --local-dir ..\data\grain_detection_public
python -m detkit predict --model weights\grain_rfdetr_small_v1\final `
    --images ..\data\grain_detection_public\photos\test\corn_0007.jpg `
    --tile 1500 --overlap 0.2 --score 0.38 --out runs\first_predictions
```

Optional, not part of the course: to train on the dataset, create a workdir and link the dataset folders into it
(same for `val`); `detkit train` reads `<workdir>/coco/{train,val}/labels.json`.
* Linux/macOS: `ln -s $(realpath ../data/grain_detection_public/coco/train) runs/mine/coco/train`
* Windows: `New-Item -ItemType Junction -Path runs\mine\coco\train -Target (Resolve-Path ..\data\grain_detection_public\coco\train)`
  (a junction needs no admin rights), or copy the folder.

How the baseline was trained, with commands, settings and the scores to expect: `weights/grain_rfdetr_small_v1/TRAINING.md`.
Baseline (photo level): F1 0.986 on val, 0.972 on test, measured against the dataset's automatic (unreviewed) labels.
Licenses: the code in this repository is MIT (`LICENSE`); the dataset and the model are CC BY-NC 4.0 (education, non-commercial).

## Workflow

| Step | Command / place | Doc |
|---|---|---|
| Plan + collect photos and weights | phone + balance | [docs/DATA_COLLECTION.md](docs/DATA_COLLECTION.md) |
| Check detections, enter metadata | `detkit app`, tab **Review** | [docs/APP.md](docs/APP.md) |
| (optional) Split + train | `detkit split --from-reviewed`, `detkit train` | [docs/TRAINING.md](docs/TRAINING.md) |
| Evaluate the detector | `detkit eval` | [docs/EVAL.md](docs/EVAL.md) |
| Count / area -> weight | app tab **Inference**, `detkit weight ...` | [docs/WEIGHT.md](docs/WEIGHT.md) |
| Your tasks (1-6) | experiments, `TODO(student)` markers, report | [docs/STUDENT_TASKS.md](docs/STUDENT_TASKS.md) |

The tasks: [docs/STUDENT_TASKS.md](docs/STUDENT_TASKS.md). The code parts: `git grep -n "TODO(student)" -- detkit`.
Check your work (environment activated): `python -m pytest tests/test_student_tasks.py -q -rs`
(Windows: `python -m pytest tests\test_student_tasks.py -q -rs`).

## Folder map
```
detkit/app/       Gradio app (Review tab: ui.py, Inference tab: inference_ui.py)
detkit/weight/    features, models*, stats*, config   (* contains TODOs)
detkit/           train.py predict.py evaluate.py split.py tiling.py store.py schema.py
detkit/vendor/    copied trainer (leave alone)
docs/             the documents above
weights/grain_rfdetr_small_v1/ baseline model (README.md = model card, TRAINING.md = how it was trained)
tests/            offline tests
runs/             YOUR data and checkpoints (git-ignored)
```
`detkit/schema.py` defines the metadata fields (app form + tables).

## Rules
1. **Split by photo, never by tile** (`detkit split` does it).
2. **One pile = one sample.** Photos of the same pile share a `pile_id`.
3. **Only saved (reviewed) photos count**, for the tables and for (optional) training.
4. **Use the best-F1 threshold** from `detkit eval`, not 0.5.
5. **State what your metrics are measured against** (reviewed boxes or pseudo-labels).
6. Keep every checkpoint you report from. Never paste your `HF_TOKEN`.



# Notes to have it running with our updated calibrations of RF detector and countx
Running the submitted pumpkin-seed model

The application detects pumpkin seeds with the trained RF-DETR checkpoint, counts the detections, and estimates mass using the saved grams-per-seed calibration. The submitted detector and calibration are loaded directly; retraining is not required to run inference.

Requirements

Windows PowerShell for the commands below.

Git and Git LFS.

uv for dependency installation and Python environment management.

Internet access for the initial model and dependency downloads.

An NVIDIA GPU is optional. The default instructions use the CPU.

1. Clone the repository and download the model

git lfs install
git clone "[link_to_repo](https://github.com/clawbuster/PumpkinWeight.git)" pumpkin-seed-project
cd pumpkin-seed-project
git lfs pull
git lfs ls-files

Git LFS downloads the actual checkpoint binaries. Run all remaining commands from the repository root, where pyproject.toml and uv.lock are located.

2. Install dependencies

If uv is not installed, install it once:

powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

If uv is not recognised afterward, reopen PowerShell and return to the cloned project folder.

Install the CPU environment:

uv sync --locked --python 3.12 --extra cpu --extra app
.\.venv\Scripts\python.exe -m detkit doctor

uv creates .venv and installs the locked dependencies, downloading Python 3.12 if needed. The commands below use this environment's Python directly, so environment activation is unnecessary.

3. Start the application

.\.venv\Scripts\python.exe -m detkit app --workdir runs/pumpkin_aug_trial --ckpt runs/pumpkin_aug_trial/checkpoints/pumpkin_aug_v1/final --device cpu

Open http://127.0.0.1:7860 in a browser on the same computer. Keep the terminal running while using the application; press Ctrl+C in that terminal to stop it.


Optional: run on an NVIDIA GPU

With a supported NVIDIA GPU and compatible driver, install the GPU environment instead of the CPU environment:

uv sync --locked --python 3.12 --extra gpu --extra app
.\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available())"

If the check prints True, launch with CUDA:

.\.venv\Scripts\python.exe -m detkit app --workdir runs/pumpkin_aug_trial --ckpt runs/pumpkin_aug_trial/checkpoints/pumpkin_aug_v1/final --device cuda