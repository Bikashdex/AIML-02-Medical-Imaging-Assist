# AIML-02 — AI-Assisted Medical Imaging and Diagnosis System

> ## ⚠ For education only. Not for clinical use.
>
> This system is an internship capstone project. It is **not** a medical device, has **no regulatory clearance**, and
> must **never** be used to diagnose, screen, or treat a patient. Any output shown here is illustrative only.

**Intern:** Bikash Kumar Singh · **ID:** NEX-INT-2026-0134 · **Track:** AI / Machine Learning
**Company:** Nexvia Technologies · **Project:** AIML-02 (of 2) · **Deadline:** 10 November 2026

## What this is

A radiologist uploads a chest X-ray to a DICOM server. A Python service notices the new study, converts it, runs a
deep learning classifier, produces a **Grad-CAM heatmap** showing which region of the image drove the prediction, and
stores the result. A React viewer shows the scan with the overlay, a confidence score, and **Approve / Reject**
controls. Every human decision is written to an audit log.

A second, independent track segments the spleen from abdominal CT volumes using a **3D U-Net in MONAI**.

## Two model tracks

| Track | Task | Dataset | Model | Headline metric |
|---|---|---|---|---|
| A | Lung opacity vs not (binary classification; opacity is the sign RSNA used as the pneumonia surrogate) | [RSNA Pneumonia Detection, stage_2](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) — Option 4 in the brief; 26,684 DICOM files | DenseNet121, ImageNet transfer learning | AUROC |
| B | Spleen segmentation from CT | [Medical Segmentation Decathlon Task09](http://medical-segmentation-decathlon.org/) — Option 5 | 3D U-Net (`monai.networks.nets.DynUNet`) | Dice score |

## Tech stack

Required by the brief, and where each one is used:

| Component | Used for |
|---|---|
| **PyTorch** | Both model tracks |
| **MONAI** | 3D U-Net, NIfTI loading, CT intensity windowing |
| **Grad-CAM** | Classification explainability |
| **Orthanc** | DICOM server that receives and serves studies |
| **DVC** | Data and model versioning, remote on Google Drive |
| **Docker** | Containerising the inference API and Orthanc |
| **React** | Radiologist review viewer |

## Repository layout

```
├── README.md                  you are here
├── docs/
│   ├── architecture.md        system diagram + data flow + API contract
│   ├── dataset_card.md        provenance, licence, label policy, splits
│   └── model_card.md          limits, bias, intended use  ← required by the brief
├── notebooks/                 Colab-executable analysis notebooks (01 → 04)
├── src/
│   ├── data/                  DICOM I/O, patient-level splits, datasets
│   ├── models/                classifier, 3D U-Net, Grad-CAM
│   ├── api/                   FastAPI inference service
│   ├── ingest/                Orthanc poller → inference → audit log
│   └── monitor/               input drift detection
├── frontend/                  React viewer
├── tests/                     pytest suite (runs in CI)
├── configs/                   YAML config, hyperparameters
├── artifacts/                 model weights (git-ignored, tracked by DVC)
├── data/                      raw + processed data (git-ignored)
├── results/                   metrics.json, plots, tables
├── Dockerfile
└── docker-compose.yml         orthanc + api
```

## Quick start

> **Use Python 3.11, not 3.13.** The pinned `torch==2.4.1` has no cp313 wheels, so `pip install -r requirements.txt`
> fails on a 3.13 interpreter. 3.11 also matches the `python:3.11-slim` image used by `Dockerfile`. If 3.11 is not
> installed, get it from python.org or `winget install Python.Python.3.11` and create the venv with
> `py -3.11 -m venv .venv`.

> **Read this before you reproduce the numbers.** The executed notebooks in `notebooks/` and
> `results/baseline_metrics.json` were produced on this machine's existing Python 3.13.1 with torch 2.11.0+cpu and
> numpy 2.4.3 — *not* the pinned environment above, which cannot be installed on 3.13. The exact versions are
> captured in `results/environment.json`. Building the 3.11 venv below gives you a working deployment environment,
> but it may reproduce the baselines to a few decimals rather than exactly. This is disclosed, not hidden.

```bash
# 1. Environment
py -3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Launch the Docker daemon (Docker Desktop must be *running* on Windows)
docker compose up --build

# 3. Inference API docs
open http://localhost:8000/docs

# 4. Orthanc UI
open http://localhost:8042

# 5. Tests
pytest -q
```

## Getting the data

Track A uses the RSNA stage_2 archives. `data/` is git-ignored, so the raw files never enter the repo; they are
tracked by DVC instead. Measured on disk: the competition download is 3.5 GB of zips and extracts to about 3.4 GB,
so allow ~7 GB while unzipping and get back to ~3.4 GB once the archives are deleted.

```bash
pip install kaggle
mkdir -p data/raw/rsna
kaggle competitions download -c rsna-pneumonia-detection-challenge -p data/raw/rsna
cd data/raw/rsna
unzip stage_2_train_labels.csv.zip && unzip stage_2_detailed_class_info.csv.zip
unzip stage_2_train_images.zip -d stage_2_train_images
# ~26,684 .dcm files. Delete the zips afterwards - keeping both wastes 3.5 GB.

# Cache every DICOM header once, so the notebooks never re-read 26,684 files:
python tools/scan_dicom_headers.py --images data/raw/rsna/stage_2_train_images \
    --out data/raw/rsna/dicom_headers.csv

# Then run, in order:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_logreg.ipynb
```

`notebooks/01_eda.ipynb` writes `results/eda_summary.json`, `results/split_{train,val,test}.csv` and
`results/train_pixel_stats.json`. The split CSVs store image paths relative to the data folder, so the same split
works locally and in Colab — point `RSNA_DATA_DIR` at the new location and nothing else has to change.

If the download is blocked to you, `python tools/make_synthetic_rsna.py` generates 108 valid DICOM files with the
same schema so the pipeline can still be smoke-tested. Say plainly in the dataset card that any figure produced that
way is synthetic — it is not a result.

### Versioning the data with DVC

`data/` never enters git; the `.dvc` pointer files next to it do. Two things to know before you type these:

* `requirements.txt` pins `pathspec==0.12.1` on purpose. DVC 3.55 imports a symbol pathspec 1.x deleted, so on a
  fresh environment without that pin every `dvc` command dies with `ImportError: cannot import name '_DIR_MARK'`.
  CI would not catch it — CI installs a deliberately light dependency set and skips torch.
* The remote is Google Drive because the free disk here is ~25 GB and Track B adds another ~2.3 GB.

```bash
dvc init
dvc add data/raw/rsna/stage_2_train_images data/raw/rsna/stage_2_train_labels.csv \
        data/raw/rsna/stage_2_detailed_class_info.csv data/raw/rsna/dicom_headers.csv
git add data/raw/rsna/*.dvc && git commit -m "chore: track RSNA data with DVC"

# One-time: create a folder in your Drive, open it, and copy its id from the URL.
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify gdrive gdrive_acknowledge_abuse true   # medical data trips Drive's abuse scan
dvc push                                                   # opens a browser consent screen
```

`dvc add` hardlinks into `.dvc/cache` on the same volume, so it costs no extra disk — verify with `df -h` before and
after. `dvc push` is the step that needs you: the OAuth consent belongs to your Google account and cannot be done
for you, and it must not be done from a machine whose credentials could end up in the repo.

To pull the data on Colab later, mount Drive and `dvc pull` with the same remote — or, faster, read the split CSVs
straight from the repo and point `RSNA_DATA_DIR` at the mounted folder.

Training runs in **Google Colab with a T4 GPU** — see `notebooks/`. This project's development machine has 8 GB RAM
and no GPU, so nothing here should be trained locally.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo scaffold | ✅ done |
| 1 | EDA + leakage-safe splits | ✅ done — `notebooks/01_eda.ipynb`, executed on the real 26,684 images |
| 1 | DVC versioning | 🟡 `dvc init` done and data tracked; Google Drive remote needs the browser OAuth step |
| 2 | Baselines → DenseNet121 | 🟡 baselines measured in `notebooks/02_baseline_logreg.ipynb` - **constant 0.5000, pixel 0.7400, frozen-CNN 0.7250 test AUROC**. DenseNet must beat 0.740 |
| 3 | Grad-CAM + fairness | ⬜ |
| 3b | MONAI 3D U-Net | ⬜ |
| 4 | FastAPI + Orthanc + audit log + Docker | 🟡 code written, not yet run — needs the Docker daemon started |
| 5 | React viewer | ⬜ |
| 6 | Docs, tests, demo, presentation | 🟡 21 tests pass locally; docs carry verified dataset and baseline numbers, DenseNet results still TODO |

## Licence and data terms

RSNA Pneumonia Detection Challenge data is licensed for **non-commercial use only**. CheXpert and MIMIC alternatives
carry their own data-use agreements. No data in this repository may be redistributed commercially.
