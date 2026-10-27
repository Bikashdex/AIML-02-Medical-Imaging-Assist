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
│   ├── model_card.md          limits, bias, intended use  ← required by the brief
│   ├── demo_script.md         10-minute recording script: shot list, verified commands, narration
│   ├── reflection.md          individual reflection (draft to personalise)
│   └── presentation/          the 13-slide deck (aiml02-presentation/AIML-02-Medical-Imaging-Assist.pptx)
├── notebooks/                 01-02 run locally, 03-05 on Kaggle (also Colab; CPU-tested locally)
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
├── weights/                   pretrained ImageNet backbones, .pth only (git-ignored, shipped inside repo.zip
│                              so a Kaggle kernel with no internet can still load DenseNet121)
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
>
> If `python`, `pip`, `dvc` and `jupyter` all stop working at once on Windows with "The system cannot find the file
> specified" while `pythonw.exe` and `Lib\site-packages` are still present, the interpreter binary itself has gone
> missing — this project's development machine hit exactly that. Look for the cause in *Windows Security → Protection
> History* (this machine's Recycle Bin was empty, so a delete is not what did it), add an exclusion for the Python
> install folder, then run the original installer and choose **Repair**. Repair restores the binary and keeps every
> installed package; uninstalling would throw away the whole environment.

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
# ~26,684 .dcm files. Keep stage_2_train_images.zip - it is both the single DVC transport object and
# the one file Colab can actually receive (26,684 loose uploads is not a plan).

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

### Running the GPU notebooks (Kaggle, Colab, or verified on CPU)

Notebooks 03 and 04 need a GPU, not this laptop. Both are written to run unchanged on **Kaggle, Colab or this
laptop**: cell 4 sets `RUN_ON = "kaggle" if os.path.isdir("/kaggle") else ("colab" if os.path.isdir("/content") else
"local")` and picks a path table from it, so there is one notebook and no forked copy to keep in sync. `local` is not
a shortcut — it is how every code cell in both notebooks got executed on CPU before handing them over, with tiny
subsample sizes and `device = cpu` (the notebooks only allow that when `n_predict <= 600`).

They also import the real package (`from src.data.dicom_io import to_uint8`) rather than pasting a second copy of
the windowing code, so the hosted run needs the code as well as the images. It gets that from
**`repo.zip` (~32 MB, 58 files)** — `src/`, `notebooks/`, `configs/`, `tools/`, `results/`, `tests/`, the requirement
files, and `weights/densenet121-a639ec97.pth`. Never include `data/`, `.dvc/config.local`, `.git`, `kaggle.json` or
`__pycache__`; the build script re-opens the archive and asserts those are absent.

**Kaggle (the route in use now — Colab's free tier stopped handing out T4 sessions after ~4 disconnects):**

1. Kaggle → **Create → Notebook**. In the right sidebar set **Accelerator → GPU T4 or P100** and leave **Internet off**
   — the pretrained backbone comes from `repo.zip`, which is why it is 32 MB rather than 2 MB.
2. **Add Input → Competitions → rsna-pneumonia-detection-challenge**. That mounts the 3.3 GB
   `stage_2_train_images.zip` under `/kaggle/input/`, so nothing has to be uploaded.
3. Attach `repo.zip` too — either **Add Input → Files** (upload it), or turn on **Settings → Google Drive** and keep
   it in `My Drive/aiml02/`. Cell 6 globs `/kaggle/input/**/repo.zip` and `/kaggle/working/repo.zip` rather than
   assuming one location.
4. Download `03_train_densenet.ipynb` from `My Drive/aiml02/` and import it (**File → Import Notebook** in the
   kernel's `...` menu) — do not paste cells by hand, because pasting silently loses the comment that explains
   which host is which. **Save version before leaving**; outputs land in `/kaggle/working`
   (`densenet_metrics.json`, `densenet121_lungopacity.pt`, `densenet_last.pt`, `densenet_training.png`) and Kaggle
   keeps them as the kernel's output. Scratch (extracted DICOMs, JPEG cache) goes to `/tmp` because `/kaggle/working`
   is what gets saved.
5. Kaggle free tier gives ~30 h/week of GPU, and a session is capped at 12 h — enough for the ~2 h run below.
6. Notebook 04 is the same competition input plus notebook 03's output: after saving 03's version, reopen
   **Add Input → Your Code → Accelerated Computed Tomography Scanning… (the 03 kernel) > the saved version**.
   Its files appear under `/kaggle/input/<kernel-slug>/` (the slug is the kernel's title in lower-case hyphens — the
   default is `quick-start-notebook` until you rename it), which is exactly where cell 8 looks for
   `densenet121_lungopacity.pt` and `densenet_metrics.json`; a missing checkpoint stops the run with that menu
   spelled out instead of failing later. Outputs: `gradcam_gallery.png`, `gradcam_focus.csv`,
   `test_predictions_with_metadata.csv`, `missing_data_audit.csv`, `gradcam_fairness_summary.json`.

**Colab, if the quota comes back:** open the Drive copy of the notebook, **Runtime → Change runtime type → T4 GPU**,
upload `repo.zip` to `My Drive/aiml02/` (or drag it into Files), make sure `stage_2_train_images.zip` is in My Drive,
and run every cell. Updating the notebook through **right-click → Manage versions → Upload new version** keeps the
same link; a plain re-upload creates a second file and it is easy to run the stale one.

Time budget on a T4, measured: unzip ~10 min, DICOM → 448 px JPEG cache ~25 min, then roughly 10-15 min per epoch with
patience 3 stopping it early. `densenet_last.pt` is rewritten after every epoch because a session disconnected at
epoch 8 would otherwise lose the whole run.

If the download is blocked to you, `python tools/make_synthetic_rsna.py` generates 108 valid DICOM files with the
same schema so the pipeline can still be smoke-tested. Say plainly in the dataset card that any figure produced that
way is synthetic — it is not a result.

### Versioning the data with DVC

`data/` never enters git; the `.dvc` pointer files next to it do. Google Drive is the remote target because this
machine has ~24 GB free and Track B adds another ~2.3 GB. What is done, and what is not:

| step | state |
|---|---|
| `dvc init` | done |
| `dvc add` of the 26,684 images + 3 CSVs | done — dir hash `3fd56c2857f39d9f9df084ddadd7c8de.dir` |
| `dvc add` of `stage_2_train_images.zip` | done — md5 `a8307fbc6d9138a7760dafde21383714`, 3,534,067,864 B. This is the **transport** object; see "one object, not 26,688" below |
| `.dvc` pointers committed | done — git alone tells a reviewer which exact 26,684 bytes (md5 + total size) every number was computed on. Getting the pixels themselves still needs the remote below |
| `dvc remote add gdrive …` | done — dedicated Drive folder `aiml02-dvc-remote`, id `1_MZ3txKpgRucQ5Ep8YxZAbKovIzDJFiL`, held in `.dvc/config` (so it is public once the repo is pushed; keep that folder private) |
| `dvc push` | **done and verified** — 4 objects, 3,546,066,217 B in the remote: the archive `a8307fbc6d9138a7760dafde21383714` (3,534,067,864 B), `stage_2_train_labels.csv` (1,490,034 B), `stage_2_detailed_class_info.csv` (1,647,396 B) and `dicom_headers.csv` (8,860,923 B). `dvc status -c` reports *Cache and remote 'gdrive' are in sync* and each Drive object's size was read back over the API. The image **folder** is deliberately not pushed — 26,688 objects for bytes already inside the archive. How it was unblocked: "What it actually took" below |

**Disk cost, measured rather than assumed.** DVC's docs promise the cache is free because it hardlinks. Here it is
free for *single-file* targets and not free for *directories*. After `dvc config cache.type hardlink,copy` and a full
cache rebuild: 26,689 cache objects, 6.62 GiB, and exactly **4 of them have `st_nlink > 1`** — the zip (3,534,067,864 B)
and the three CSVs, which is precisely the set of individually-tracked files. The 26,684 images inside the tracked
folder are still byte-for-byte copies, so `dvc add` of that folder really did cost ~3.1 GB. `os.link()` succeeds on
those `.dcm` files when called directly, so this is DVC's per-file fallback inside directory imports, not a filesystem
limitation. Two consequences worth knowing: the archive is the only thing worth pushing, and freeing the folder's
copies means untracking the folder — which this project deliberately does not do, because the folder pointer is the
per-image md5 provenance record.

**One object, not 26,688.** `find .dvc/cache -type f | wc -l` → **26,688**. `dvc push` uploads one Drive object per
cache file, so pushing the image *folder* means ~26,700 HTTP requests and Drive's rate limiter will make that hours.
The archive is therefore what gets pushed; the folder pointer stays in git purely as the per-image md5 record.

The commands that produced that state, for anyone reproducing from a clean clone:

```bash
dvc init
dvc config cache.type hardlink,copy
dvc add data/raw/rsna/stage_2_train_images data/raw/rsna/stage_2_train_labels.csv \
        data/raw/rsna/stage_2_detailed_class_info.csv data/raw/rsna/dicom_headers.csv
dvc add data/raw/rsna/stage_2_train_images.zip     # the single object that actually gets pushed
git add data/raw/rsna/*.dvc && git commit -m "chore: track RSNA data with DVC"

# One-time: create a folder in your Drive, open it, and copy its id from the URL.
dvc remote add -d gdrive gdrive://<FOLDER_ID>
dvc remote modify gdrive gdrive_acknowledge_abuse true   # chest X-rays trip Drive's abuse scan
dvc push data/raw/rsna/stage_2_train_images.zip.dvc      # push the archive; the folder is provenance only
```

**Why the default `dvc push` is blocked, from the installed source rather than from a forum.** `dvc-gdrive` ships one shared OAuth
client that every user of the plugin inherits (`dvc_gdrive/__init__.py:23-24`, client id
`710796635688-…apps.googleusercontent.com`), and `pydrive2` requests the **full `drive` scope** with it
(`pydrive2/fs/spec.py:26-29`). Google classifies that scope as sensitive and refuses to grant it to an unverified
third-party app, so the consent screen ends at "This app is blocked". Retrying does not help: the block is on the
shared client, not on your account.

Two ways forward.

**Option A — register your own OAuth client. Done, and this is the exact sequence that worked.**
In Google Cloud Console, as the same Google account that owns the Drive: create/open a project → **enable the
*Google Drive API*** → OAuth consent screen (type *External*) → **add your own Google address under Test users**
(this is the step that gets past the block) → Credentials → *OAuth client ID* → application type **Desktop app**.
Then, in the repo:

```bash
dvc remote modify --local gdrive gdrive_client_id     <CLIENT_ID>
dvc remote modify --local gdrive gdrive_client_secret <CLIENT_SECRET>
dvc push data/raw/rsna/stage_2_train_images.zip.dvc
```

Three things cost time and are worth writing down:

* **Enabling the API is not the same as creating the client.** With the client configured but `drive.googleapis.com`
  still off, consent succeeds and *then* the first API call answers `accessNotConfigured`: "Google Drive API has not
  been used in project … before or it is disabled."
* **The folder in `gdrive://<FOLDER_ID>` must already exist, and the id must be its real one.** A nonexistent id does
  not produce a friendly error: `pydrive2` asks the folder for its `driveId`, gets HTTP 404, and turns that into
  `PermissionError` ([`pydrive2/fs/spec.py:284-300`](file:///C:/Users/kumar/AppData/Local/Programs/Python/Python313/Lib/site-packages/pydrive2/fs/spec.py#L284)).
  Create the folder in Drive, open it, and copy the long id from the URL.
* **Consent can report a false rejection.** `dvc push` starts a server on `localhost:8080` and waits for the redirect;
  if a stale tab hits that port without a `code`, pydrive2 raises `AuthenticationRejected` ("User rejected
  authentication") even though nothing was refused. Close stray `localhost` tabs and retry — the second attempt on this
  machine returned `Authentication successful.`

**Desktop app, not Web — this is the part that costs an hour if you guess.** pydrive2 authenticates by starting a
local web server on `http://localhost:8080/` (falling back to 8090) and having Google redirect the code to it
([`pydrive2/auth.py:249-269`](file:///C:/Users/kumar/AppData/Local/Programs/Python/Python313/Lib/site-packages/pydrive2/auth.py#L249)).
Loopback redirect URIs are allowed for *installed* client types and refused for *Web application* clients, so a Web
client fails the flow before consent with `redirect_uri_mismatch`. That error is reproducible without a browser:

```bash
curl -s -D - -o /dev/null "https://accounts.google.com/o/oauth2/auth?client_id=<ID>&redirect_uri=http%3A%2F%2Flocalhost%3A8080%2F&response_type=code&scope=https%3A%2F%2Fwww.googleapis.com%2Fauth%2Fdrive&access_type=offline" | grep -i '^location'
```

A `Location:` header containing `oauth/error?authError=` means the client is not usable yet; a redirect straight to
the consent page means it is.

`--local` is not a style preference: it writes `.dvc/config.local`, which is git-ignored, whereas `.dvc/config` is
committed. Putting the client secret there would publish it in the same repo whose CI workflow exists to keep
credentials out. With an unverified app the consent page still warns "Google hasn't verified this app" — choose
*Advanced → Go to … (unsafe)* — and refresh tokens stop working after 7 days, so expect to re-authenticate weekly.

**Option B — no third-party OAuth at all (the route `notebooks/03_train_densenet.ipynb` uses).** DVC keeps doing the
versioning job locally, and transport to the GPU machine goes through Colab's own Drive mount, a first-party
integration that needs no app registration:

```bash
# laptop: upload the original 3.3 GB stage_2_train_images.zip (or a zip of .dvc/cache) to MyDrive/aiml02/
# colab:  drive.mount("/content/drive"), then either unzip, or use the mount as a *local* DVC remote:
dvc remote add --default drivecache /content/drive/MyDrive/aiml02/dvc-cache
dvc pull
```

`notebooks/03_train_densenet.ipynb` implements Option B's unzip path. This substitution is recorded in
`docs/dataset_card.md` as well, because "the DVC remote is not reachable from the submission environment" is a
limitation of the work, not a detail.

Two other things to know before you type `dvc` commands:

* `requirements.txt` pins `pathspec==0.12.1` on purpose. DVC 3.55 imports a symbol pathspec 1.x deleted, so on a
  fresh environment without that pin every `dvc` command dies with `ImportError: cannot import name '_DIR_MARK'`.
  CI would not catch it — CI installs a deliberately light dependency set and skips torch.
* `dvc push`/`dvc pull` also need `pydrive2[fsspec]==1.21.1` with `pyOpenSSL==26.4.0`; newer pydrive2 caps pyOpenSSL
  at 24.2.1, which cannot coexist with the `cryptography>=48` that `scmrepo`/`asyncssh` require, and `dvc push` dies
  with `module 'lib' has no attribute 'GEN_EMAIL'`.

Training runs on a **free hosted GPU - Kaggle now, Colab if its T4 quota comes back** — see `notebooks/`. This
project's development machine has 8 GB RAM and no GPU, so nothing here should be trained locally. Notebooks 01 and
02 do run locally (CPU, tabular work); 03 and 04 detect the host and treat a laptop as a smoke test only.

## Status

| Phase | Deliverable | State |
|---|---|---|
| 0 | Repo scaffold | ✅ done |
| 1 | EDA + leakage-safe splits | ✅ done — `notebooks/01_eda.ipynb`, executed on the real 26,684 images |
| 1 | DVC versioning | ✅ `dvc init` + `dvc add` + committed pointers + `dvc push` of the archive and 3 CSVs, verified by reading each object's size back from the Drive API and by `dvc status -c` reporting sync. Needed a personal **Desktop app** OAuth client and a real Drive folder id; both are recorded above so the next person does not rediscover them |
| 2 | Baselines → DenseNet121 | ✅ trained on a **Kaggle T4** (2026-10-26, 8 epochs, early stop at 4): **test AUROC 0.8819**, val 0.8882, PR-AUC 0.7085, at threshold 0.5108 sens 0.7661 / spec 0.8055. Beats the 0.740 pixel baseline by +0.142 (`results/densenet_metrics.json`). The frozen-ResNet row scoring 0.7250 is the measured argument for fine-tuning |
| 3 | Grad-CAM + fairness | ✅ run on the **Kaggle T4** over all 4,003 test images: AUROC 0.8819 (CI 0.8697–0.8916), 300 Grad-CAM maps scored for attention inside the body (43 % under half), subgroup tables for sex, age band and label source, missing-data audit all zero (`results/gradcam_fairness_summary.json`). Calibration measured post hoc: over-confident, ECE 0.1256 |
| 3b | MONAI 3D U-Net | ✅ rerun on the **Kaggle T4** (2026-10-27, 90 epochs, 119.3 GPU-min, kernel version 8): **mean test Dice 0.8586** — median 0.9124, range 0.5972–0.9480 over the 6 held-out volumes, no zero-Dice cases, `n_below_0_5` = 0, best quick-val 0.9357 at epoch 79 (`results/spleen_metrics.json`, per-case table in `results/spleen_per_case.csv`, overlays in `results/plots/spleen_overlays.png`). The first 40-epoch run scored 0.329 because the sampler drew positives from a voxel inside the organ and negatives from the volume centre, teaching the net a blob near the middle of the window; jittered positive centres plus uniformly drawn negatives is the whole difference between 0.329 and 0.8586 |
| 4 | FastAPI + Orthanc + audit log + Docker | 🟡 image built offline from `wheelhouse/` (2026-10-27, `aiml-02-medical-imaging-api:latest`, 2.09 GB) and the container serves `/health`, `/model-info` and `/predict` on port 8010: `positive.dcm` → **0.8685 opacity**, `negative.dcm` → **0.1826 no_opacity**, tags serialised, 2 tests in `tests/test_api.py` pass. Build fixes: `uvicorn[standard]` needs `uvloop`, which was not in the wheelhouse, so the `--no-index` install died — plain `uvicorn` is pinned instead, and both pip lines are offline. Latency measured twice: **7.1–14 s per study inside the container** (4 vCPU, 2 torch threads) versus **0.42–0.48 s** for the same code and weights in the host venv — 30× slower in Docker Desktop's VM on this 4-core laptop, which is a platform cost, not a model cost. Still open: `orthancteam/orthanc` never pulled (nine attempts: build-proxy DNS and CDN timeouts; the last one listed 6 layers as "Pulling fs layer" and never completed a single one in 20 minutes), so the stack runs as `docker compose up -d --no-deps api` and the audit DB stays unwired — there is no audit endpoint yet. What *is* proven is the poller's half: `tests/test_ingest.py` drives `poll_once` against a stub that speaks Orthanc's REST shape, asserting that it forwards new instances, skips `Removed` ones, advances the `/changes` marker so a restart does not replay old studies, and survives one fetch failing mid-batch. Writing that test found a real bug, now fixed: `fetch_dcm` never called `raise_for_status`, so a failed fetch would have posted Orthanc's 500 response body to `/predict` as if it were a study |
| 5 | React viewer | ✅ exercised 2026-10-27 **against the running container on 8010** with real browser screenshots this time: `positive.png` → **86.9 % opacity**, `negative.png` → **18.3 % no_opacity**, `positive.dcm` → **86.9 %** with its real tags (patient id, CR, 1024×1024) and Approve moved it to **approved**; the 60 % overlay slider and the education banner render, the heatmap toggle stays disabled with "(none from API yet)" because `/predict` returns no `overlay_url`, and the panel states in the UI that decisions are session-only until the audit endpoint is wired to `audit.db`. Reject shares the same handler but was not clicked. Note: a `.dcm` upload cannot be rendered as an image in the browser ("Browser preview not available"), so the picture itself comes from the PNG path |
| 6 | Docs, tests, demo, presentation | 🟡 **83 tests pass locally**; dataset card, both halves of the model card and both runbooks carry measured numbers. **13-slide deck** in `docs/presentation/` follows the brief's narrative (two tracks → numbers → explainability → live service → the sampler bug → limits); **`docs/demo_script.md`** is a 10-minute shot list whose commands were re-run on 2026-10-27 against the **running container** (0.8685 / 0.1826 at threshold 0.5108, viewer proxying 5173 → 8010), with the Orthanc gap stated on camera; **`docs/reflection.md`** is a first-person draft marked for personalisation. Outstanding: the recording itself |

## Licence and data terms

RSNA Pneumonia Detection Challenge data is licensed for **non-commercial use only**. CheXpert and MIMIC alternatives
carry their own data-use agreements. No data in this repository may be redistributed commercially.
