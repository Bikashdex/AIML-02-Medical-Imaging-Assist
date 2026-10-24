# Model card

> Portal step 7: *"Containerize with Docker and document the model's limits and bias. Include a clear 'for education
> only, not clinical use' notice."* This file **is** that deliverable. Fill every `TODO` from your own runs.

## ⚠ Intended use

**Education only. Not for clinical use.**

This is not a medical device. It has no FDA, CE, or CDSCO clearance. It must not be used to triage, diagnose,
screen, or treat any patient, and no clinical decision should rely on its output.

---

## Track A — Chest X-ray pneumonia classifier

### Identity

| Field | Value |
|---|---|
| Architecture | DenseNet121, ImageNet-pretrained, final layer replaced |
| Task | Binary classification: lung opacity vs not. Opacity is the **sign** the label encodes, used by RSNA as the pneumonia surrogate — it is not a pneumonia diagnosis |
| Input | 224×224 RGB from 8-bit DICOM, ImageNet normalisation |
| Output | Sigmoid probability + Grad-CAM heatmap |
| Training data | RSNA 2018 stage_2, 26,684 images, 22.53 % positive (see `dataset_card.md`) |
| Split policy | **Image-level split on one-row-per-picture data**, seed 7, 70/15/15, folds asserted disjoint |
| Version / date | `TODO` e.g. `densenet121-2026-11-02` |

### Performance

| Metric | Val | Test |
|---|---|---|
| AUROC | `TODO` | `TODO` |
| Sensitivity (recall) | `TODO` | `TODO` |
| Specificity | `TODO` | `TODO` |
| Precision | `TODO` | `TODO` |
| F1 | `TODO` | `TODO` |
| Accuracy | `TODO` | `TODO` |
| Threshold used | `TODO` | `TODO` |

Accuracy is reported but **is not the headline metric** — the class balance makes it misleading. AUROC, sensitivity
and specificity are the numbers that matter for a screening aid.

### Baseline comparison

Measured in `notebooks/02_baseline_logreg.ipynb`, full detail in `results/baseline_metrics.json`. Subsets are not
identical between rows - level 1 uses 4,000 train images, level 2 uses 1,200 - so compare the test columns, not the
train ones.

| Model | AUROC val | AUROC test | Note |
|---|---|---|---|
| Constant predictor | 0.5000 | - | 77.47 % accuracy for a switch. The reason accuracy never leads a table |
| Logistic regression on 1,024 pixels (32x32 downsample) | 0.7520 | **0.7400** | Course Day 10 baseline. Fits its own train subset at 0.9037 |
| Logistic regression on 512 frozen ResNet-18 ImageNet features | 0.7234 | 0.7250 | Course Day 12 without the fine-tuning. Train AUROC 0.9999 on 1,200 samples |
| DenseNet121, fine-tuned (this model) | `TODO` | `TODO` | **must beat 0.740 on test AUROC**, or transfer learning added nothing |

The frozen-feature row scoring *below* the pixel row is a result, not a mistake. Pretrained ImageNet features are
tuned for natural photographs; the value of transfer learning here appears only once the weights are fine-tuned.
If the DenseNet lands around 0.85, this table is the evidence for where the gain came from. If it lands near 0.74,
say so.

Scoring convention: test numbers are taken at threshold **0.163**, Youden's J read on validation only. At that
threshold the pixel baseline reaches sensitivity 0.698 / specificity 0.663 on the test subset.

A random forest was not used as a baseline. The three levels above - constant, hand-of-pixels, frozen-CNN - isolate
the two things worth separating (representation and fine-tuning), which a second tree model on the same 1,024
features would not.

### Calibration

`TODO` — a probability of 0.87 should mean roughly 87% of the time. Report a reliability diagram and say whether the
model is over- or under-confident. Thresholds chosen on an uncalibrated score are fragile.

### Explainability — and its honest limits

Grad-CAM produces a ReLU'd activation map from the final dense block, upsampled onto the input.

**What it is not:** it is not SHAP, it is not a causal explanation, and it is not a guarantee of where the model
looks. A Grad-CAM map over the heart does not prove the model used the heart. It shows a region whose activations
contributed to the class score. Say this in the viva; do not oversell the heatmap.

`TODO` include the failure gallery: 2 correct, 2 wrong, 2 uncertain. The wrong ones are the interesting ones.

### Bias and subgroup performance

| Slice | n | AUROC | Gap vs overall |
|---|---|---|---|
| Sex = M | 15,166 | `TODO` | `TODO` |
| Sex = F | 11,518 | `TODO` | `TODO` |
| Age < 40 | 9,044 | `TODO` | `TODO` |
| Age 40–59 | 11,667 | `TODO` | `TODO` |
| Age 60–79 | 5,731 | `TODO` | `TODO` |
| Age 80–110 | 242 | `TODO` | `TODO` |
| Label source = `Normal` | 8,851 | `TODO` | `TODO` |
| Label source = `No Lung Opacity / Not Normal` | 11,821 | `TODO` | `TODO` |

Slice counts and the sex/age values come from the DICOM tags, recovered in `notebooks/01_eda.ipynb` §3 and written
into `results/split_*.csv`. Observed label prevalence before any model is involved: 21.7 % female vs 23.1 % male,
and 25.5 % under 40, 21.0 % in 40–59, 20.9 % in 60–79, 24.8 % in 80–110. Prevalence barely moves across these slices,
which means **a large AUROC gap between them would point at the model, not at the data** — that is the useful
comparison to make.

The last two rows matter more than the demographic ones. The negative class is not "normal chest": 11,821 of the
20,672 negatives are abnormal for some other reason. Measuring accuracy separately on the 8,851 truly-normal films
versus the 11,821 other-abnormal films tells you whether the model learned "opacity" or merely learned "not
obviously normal". Report both.

**The PA vs AP view confound cannot be examined here, and that is stated rather than skipped.** `PatientPosition`
is empty on all 26,684 files, so view is not recoverable from this release. The mechanism is real and worth naming in
the viva: AP films are usually portable X-rays on patients who cannot stand, so view position correlates with illness
severity, and a model can learn "portable machine → opacity" without learning the pathology. This dataset simply
cannot show it.

What can be measured, and was:

- **Gross exposure as a shortcut.** Mean image brightness correlates with the label at **-0.061** on 4,000 training
  images (`results/plots/baseline_pixels.png`). A plausible confounder, measured, and ruled out as a major driver.
  Ruled out for brightness specifically - not for every low-frequency cue the pixel model uses.
- **Prevalence by slice**, so an AUROC gap can be attributed to the model rather than to unequal base rates
  (table above).
- **Performance by slice** on a 2,000-image test subset for the pixel baseline: female 0.7274, male 0.7500, and
  0.7625 under 40 / 0.7557 in 40-59 / 0.6576 in 60-79. The 80-110 band is too small to score. With 397 images and
  75 positives in the weakest band, that 0.10 spread is suggestive and not proven; it is exactly the kind of
  quantity a confidence interval settles, which notebook 03 computes on the full test set.

Still open: duplicate renders of the same film within the release, and whether `PixelSpacing` clusters by
acquisition. Neither is claimed as done.

### Known limitations

1. Single-dataset training; no external validation on a different hospital's cameras or population.
2. Binary task only — cannot report the other 13 findings a real chest X-ray study needs assessed.
3. **Not paediatric-capable.** 1,276 of 26,684 images (4.8 %) are from patients under 18, which is enough to keep
   them in training but nowhere near enough to evaluate a paediatric subgroup. Paediatric chests also look
   structurally different. Do not present this as valid on children.
4. No 3D information — overlapping structures can hide findings a radiologist would resolve from two views.
5. Label noise: bounding boxes came from 18 radiologists across 16 institutions.
6. Demographic distribution mirrors one US dataset; transferability to the local population is untested.
7. Confidence scores are not calibrated unless `TODO` says otherwise.

### Failure modes observed

`TODO` — describe at least three concrete cases: what the image showed, what the model said, what Grad-CAM
highlighted, and what you think the model actually keyed on.

### Intended deployment and guardrails as built

- Human-in-the-loop is mandatory: the viewer's Approve / Reject decision is recorded and the model never finalises a
  result alone.
- Full audit trail of every prediction and every human decision.
- Confidence thresholding plus an explicit "uncertain" band rather than forcing a binary call.
- Input drift monitor (PSI on pixel statistics) to warn when incoming studies stop looking like training data.
- Education-only banner rendered inside the React viewer, not just in this document.

### What would change the assessment

External validation on a second institution, a multi-label target, calibration, and a prospective review of whether
the model changes radiologist behaviour at all.

---

## Track B — Spleen segmentation, 3D U-Net (MONAI `DynUNet`)

| Field | Value |
|---|---|
| Architecture | `DynUNet`, 3D, deep supervision |
| Input | CT volume resampled to 1.5 mm iso, windowed to [-150, 200] HU |
| Output | Binary spleen mask |
| Training data | MSD Task09_Spleen, 41 volumes |
| Metric | Mean Dice, plus per-case Dice |
| Inference | Sliding window, ROI 192³ |

| Metric | Value |
|---|---|
| Mean Dice (val) | `TODO` |
| Mean Dice (test) | `TODO` |
| Worst-case Dice | `TODO` |
| Median inference time / volume | `TODO` |

**Limits:** trained on 41 volumes from a single collection. Small spleens, post-surgical anatomy, and unusual
contrast timing are underrepresented. A low-Dice tail matters more than the mean — report the worst cases, not just
the average. `TODO` include the per-case table.

---

## Reproducibility

| Item | Value |
|---|---|
| Commit hash | `TODO` (repo has no commit yet) |
| DVC data revision | images `3fd56c2857f39d9f9df084ddadd7c8de.dir` (26,684 files, 3,555,311,924 B); `stage_2_train_labels.csv` `010a01a3f9a7cf76cb14704a95b22263`; `stage_2_detailed_class_info.csv` `c3bb15f6d279cfe39e09da8903902478`; derived header cache `dicom_headers.csv` `28b6ca9e1af3dde6e89e0cc3a755a3c9` |
| DVC remote | Google Drive (`gdrive://<FOLDER_ID>`) — `dvc push` pending the one-time browser OAuth |
| Seed | 7 |
| Runtime | Google Colab, Tesla T4 |
| Package versions | `requirements.txt` |
| Environment that produced the baseline numbers | `results/environment.json` — Python 3.13.1, torch 2.11.0+cpu, numpy 2.4.3, scikit-learn 1.9.1, pydicom 3.0.2 |
| Reproduction status of the baselines | Ran twice end to end, every AUROC and PR-AUC identical to 4 decimal places, threshold reproduced at 0.163. Determinism holds *within* this environment; it is not claimed across environments, because the pinned `torch==2.4.1` / `numpy==1.26.4` cannot even install on 3.13 (no cp313 wheels) |
| Training command / notebook | `notebooks/03_train_densenet.ipynb` |
