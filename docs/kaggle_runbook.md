# Kaggle runbook — notebooks 03 and 04

**Primary runbook while the Colab quota is exhausted.** Both notebooks detect the
host from the filesystem (`RUN_ON = "kaggle"` when `/kaggle` exists), so nothing
in them needs editing — this file is about the Kaggle-specific clicks.

---

## 0. One-time prep

- [ ] **Kaggle account** — sign up at kaggle.com.
- [ ] **Accept the competition rules.** Open the *RSNA Pneumonia Detection
      Challenge* page → Rules → "I Understand and Accept". Without this the
      dataset does not appear in any Add Input list.
- [ ] **Phone verification** — Settings → verify your phone number. Kaggle
      enables GPU T4 only after this (and it is needed for Internet on).
- [ ] Quota: ~30 GPU hours/week on the free tier — notebook 03 needs only a few.

## 1. repo.zip — the code archive (once)

Use `repo.zip` from the workspace folder (regenerated 2026-10-25, ~38 MB). It
carries `src/`, `results/split_*.csv`, the notebooks, the vendored
monai/nibabel wheels, and `weights/densenet121-a639ec97.pth` — the ImageNet
weights are inside on purpose: `src/models/classifier.py` copies them into the
torch hub cache so **no internet is needed during the run**.

Upload it once as a private dataset:

1. kaggle.com → Datasets → **New Dataset** → drag `repo.zip` → Create.
   (Private by default — keep it that way; it is your coursework.)
2. From then on, attaching it to any notebook is one click (Add Input → Your
   Datasets). Alternatively drag the zip into Add Input → Upload per session.

## 2. Session A — `03_train_densenet.ipynb`

1. kaggle.com → **Code → New Notebook**.
2. Right sidebar → **Accelerator → GPU T4**. Leave **Internet off** — the
   Kaggle image ships torch/pydicom/sklearn, and repo.zip carries the weights.
3. **Add Input** (right panel):
   - *Competitions* → RSNA Pneumonia Detection Challenge → `stage_2_train_images.zip`;
   - *Your Datasets* → the `repo.zip` dataset.
4. Import the notebook: in the editor **File → Import Notebook** → upload
   `notebooks/03_train_densenet.ipynb` from the repo. (If import is not
   offered, copy-paste all cells into a fresh notebook instead.)
5. **Run All.** Watch for, in order:
   - `RUN_ON: kaggle` and the CONFIG printout;
   - zip found (`/kaggle/input/...`), repo extracted, split CSVs loaded with
     label rates;
   - the decode phase (26,684 DICOM → 448 px JPEG) — the long CPU part;
   - epoch rows printing; final cell with **test metrics** and the saved-file
     list (`densenet_metrics.json`, `densenet121_lungopacity.pt`, figure).
6. **Save Version** (top right) → **Save & Run All**. This is what stores
   `/kaggle/working` as the kernel's Output — Session B depends on it.
7. **Screenshots for the report:** the test-metrics cell, the saved-files cell,
   and the Accelerator setting showing T4.

## 3. Session B — `04_gradcam_fairness.ipynb`

1. New notebook, **GPU T4**, Internet off.
2. **Add Input:**
   - RSNA competition data (test images are decoded again);
   - your `repo.zip` dataset;
   - **Your Code → the notebook-03 kernel you saved** — its Output (checkpoint
     `densenet121_lungopacity.pt` + `densenet_metrics.json`) lands under
     `/kaggle/input/<kernel-slug>/`, which is exactly where this notebook hunts;
   - `dicom_headers.csv` **if** you have it — grab it from your Drive `aiml02`
     folder and attach it the same way. Without it the notebook still runs,
     falling back to demographics inside the split CSVs (the cached scan is
     better; plan a scan session later with `tools/scan_dicom_headers.py`).
3. Import `notebooks/04_gradcam_fairness.ipynb`, **Run All**.
4. **Save Version → Save & Run All** so the deliverables persist.
5. **Screenshots for the report:** the Grad-CAM overlay figure, the subgroup
   fairness table, the missing-data audit, the final summary cell.

## 4. Bring the results home

1. Open the kernel's **Output** tab → Download:
   - from notebook 03: `densenet_metrics.json`, the figure PNG, and
     `densenet121_lungopacity.pt`;
   - from notebook 04: the Grad-CAM PNGs, `focus.csv`,
     `test_predictions_with_metadata.csv`, `missing_data_audit.csv`, and the
     summary JSON.
2. Drop the metrics/CSVs/PNGs into the repo's `results/` folder, and the
   checkpoint into `artifacts/`.
3. Tell the chat — it will `dvc add` + `dvc push` the checkpoint (git stores
   only the pointer; `*.pt` is gitignored on purpose) and commit the rest.

## 5. Gotchas

- **No GPU?** Phone verification missing. Settings → verify phone → re-open the
  notebook.
- **RSNA not in Add Input?** Competition rules not accepted yet.
- **Run died mid-training?** Re-run via Save Version. Outputs already in
  `/kaggle/working` only survive in a saved version.
- **Weekly quota** resets weekly; a full 03+04 cycle is a few GPU-hours, well
  inside it.
- **Never quote a `host: local` run** as a result — the notebooks record the
  host inside their metrics JSON for exactly this reason.

## 6. Not covered here yet

- **Notebook 05 (MONAI spleen, DynUNet)** does not exist yet — the previous
  chat discussed it and staged the vendored wheels for it, but no notebook file
  was ever saved. Building it is a coding task in the repo; it then runs here as
  Session C (same Add Input pattern; `tools/get_msd_spleen.py` fetches
  Task09_Spleen — that download does need Internet on, once).
