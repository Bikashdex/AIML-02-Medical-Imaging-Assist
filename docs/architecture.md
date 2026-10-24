# Architecture

> Portal deliverable: *"Architecture diagram and data or API documentation."*
> Keep the three sections below — system diagram, data flow, API contract.

## 1. System diagram

```
   Radiologist uploads DICOM
             |
             v
  +---------------------+   DICOMweb / C-FIND    +----------------------+
  |  Orthanc (Docker)   | <--------------------> |  Ingestion service   |
  |  AE title: AIML02   |                        |  src/ingest/         |
  |  :4242 DICOM        |                        |  orthanc_poller.py   |
  |  :8042 HTTP         |                        |  DICOM -> array+meta |
  +---------------------+                        +----------+-----------+
                                                            |
                                          POST /predict     |
                                                            v
  +---------------------+   load    +----------------------------------------+
  |  artifacts/  (DVC)  | ------->  |  FastAPI inference service (Docker)    |
  |  classifier.pt      |           |  src/api/main.py                       |
  |  unet3d.pt          |           |  DenseNet121  +  Grad-CAM heatmap      |
  |  model.onnx         |           +----------+-----------------------------+
  +---------------------+                      | writes
                                               v
                                    +------------------------+     +------------------+
                                    |  audit.db (SQLite)     | <-- |  React viewer    |
                                    |  study_uid, label,     |     |  image + overlay |
                                    |  confidence, decision, |     |  Approve/Reject  |
                                    |  radiologist_id, ts    |     +------------------+
                                    +------------------------+
                                               ^
                                    +------------------------+
                                    |  Drift monitor         |
                                    |  PSI on pixel stats    |
                                    +------------------------+
```

A second, offline track trains a 3D U-Net (MONAI `DynUNet`) on MSD Task09 spleen CT volumes. It shares the model
artefact store and the Docker build but not the live DICOM path.

## 2. Data flow, step by step

| # | Step | Where | Notes |
|---|---|---|---|
| 1 | Study arrives | Orthanc | Pushed via DICOM C-STORE to AE title `AIML02` on port 4242 |
| 2 | New instance detected | `orthanc_poller.py` | Polls `/discover/changes`; `StableAge: true` avoids half-arrived studies |
| 3 | DICOM read | `src/data/dicom_io.py` | `pydicom` reads tags; **windowing** applied before the model sees pixels |
| 4 | Preprocess | `src/data/` | Resize 224×224, ImageNet normalisation |
| 5 | Predict | `src/api/main.py` | DenseNet121 → sigmoid probability |
| 6 | Explain | `src/models/gradcam.py` | Hook on the last dense block, ReLU'd activation map overlaid |
| 7 | Persist | `audit.db` | One row per study, decision starts as `pending` |
| 8 | Review | React viewer | Radiologist approves or rejects, with a free-text reason |
| 9 | Audit | `audit.db` | Decision, user and timestamp appended — never overwritten |
| 10 | Monitor | `src/monitor/drift.py` | PSI of incoming pixel statistics vs training reference |

## 3. API contract

### `POST /predict`

Request: `multipart/form-data`
- `file`: one DICOM instance **or** a PNG/JPG
- `study_uid`: optional string, defaults to the DICOM `SOPInstanceUID`

Response `200`:
```json
{
  "study_uid": "1.2.840.113619.2.55.3...",
  "prediction": "opacity",
  "probability": 0.87,
  "threshold_used": 0.55,
  "overlay_path": "/overlays/1.2.840....png",
  "model_version": "densenet121-2026-11-02",
  "inference_ms": 41.2,
  "disclaimer": "For education only. Not for clinical use."
}
```

Errors: `400` unreadable/unsupported image, `413` over the size limit, `503` model not loaded.

### `GET /health`
`{"status": "ok", "model_loaded": true, "gpu": false}` — used by the Docker `HEALTHCHECK`.

### `GET /model-info`
Model name, training dataset and version, split policy, metrics, training date, licence, intended-use statement.

### `GET /studies?status=pending&limit=50`
Paged list from the audit table for the viewer.

### `POST /studies/{study_uid}/decision`
`{"decision": "approve" | "reject", "radiologist_id": "...", "reason": "..."}` → appends to the audit log.

## 4. Design decisions and their trade-offs

| Decision | Why | What it cost |
|---|---|---|
| Split on the image, never the CSV row | `stage_2_train_labels.csv` has one row per **bounding box**: 30,227 rows describe 26,684 pictures. Shuffling rows leaked 16.8 % of test images into training (measured in notebook 01). `src/data/splits.py` groups on `patient_id`, which also covers the multi-film case real archives have | The group is one image here, so it buys nothing measurable in this release — the code is written for the general case |
| Binary opacity-vs-not, not 14 findings | One clean question, measurable; multi-label on a small subset is unreliable | Ignores the richer label space, and 57 % of negatives are "abnormal for another reason" |
| Grad-CAM instead of SHAP | SHAP over 224×224 pixels is computationally impractical; an activation map is what a radiologist can read | Not a Shapley-value guarantee — do not call it one |
| SQLite for the audit log | Single-node demo, zero ops | No row-level security, no concurrent writers |
| Google Drive as DVC remote | Local disk is ~27 GB; the dataset is ~3.4 GB extracted plus Track B's ~2.3 GB | Slow first pull; a real deployment wants S3/GCS |
| Orthanc polls rather than pushes | Simpler to reason about and to record a demo against | Adds up to `POLL_INTERVAL_SECONDS` latency |

## 5. Security and privacy notes

- Patient identifiers are hashed before they reach the audit log; raw `PatientID` stays inside Orthanc.
- Orthanc runs with `AuthenticationEnabled: false` and `RestApiAllowedIps` open **for the demo only**. In a real
  deployment: enable authentication, bind ports to localhost or an internal network, and put TLS in front.
- `kaggle.json` and any Orthanc credentials are git-ignored. Never commit them.
- No PHI is committed to this repository. RSNA data is de-identified at source.
