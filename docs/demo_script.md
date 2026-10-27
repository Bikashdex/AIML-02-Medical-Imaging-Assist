# Demo recording script — 10 minutes

**What this is:** a shot list plus word-for-word narration for the AIML-02 demo video. Every command below was
executed on this machine on **2026-10-27** and the outputs shown are the real ones, not illustrative. Record it in one
take if you can; the numbers do not drift between runs because the model runs on CPU in eval mode.

**What the video must prove** (that is what is being graded):

1. The classifier exists, is trained, and answers with real numbers on real DICOM files.
2. The service is a service — HTTP, not a notebook cell.
3. A radiologist-facing viewer shows the scan, the score and an audit decision.
4. The segmentation track works too (0.8586 Dice, not the 0.329 first attempt).
5. You can state the limits without being asked.

**What it must not do:** claim clinical use, show a Docker/Orthanc flow that is not actually running, or quote a
number that is not in `results/`. The disclosure lines in shots 8 and 9 exist for that reason — say them.

---

## Before you hit record

| Check | Command / action |
|---|---|
| Terminal font | 16 pt or larger; window 1200 px wide; dark theme |
| Clean desktop | Close Slack/Explorer; hide desktop icons (right-click desktop → View → uncheck) |
| Repo updated | `git pull`; confirm `artifacts/densenet121_lungopacity.pt` exists (28,401,697 B) |
| Sample files exist | `dir data\raw\rsna\stage_2_train_images\019d950b-*.dcm` and `000924cf-*.dcm` — two real RSNA studies used below |
| Venv | `C:\Users\kumar\venvs\aiml02-py312\Scripts\activate` (Python 3.12, torch 2.14.1+cpu) |
| API boot time | ~10 s (container already running; check first) |
| Viewer boot | 10–35 s on this laptop; start terminal B early, then switch to it |
| Container | `docker ps` shows `aiml02-api` **healthy** on `0.0.0.0:8010->8000/tcp` |

Three windows, in this order:

1. **Terminal A** — the API container, port **8010** (`API_PORT=8010`).
2. **Terminal B** — viewer, port **5173**, proxying to 8010.
3. **Browser** — `http://localhost:5173` (use `localhost`, not `127.0.0.1`; Vite binds to the IPv6 loopback).
4. **Slides** — `aiml02-presentation/AIML-02-Medical-Imaging-Assist.pptx`, 13 slides, presenter view on.

> **Docker is part of the recording, with one gap.** The engine is back and the API container (`aiml02-api`, 2.09 GB,
> built offline from `wheelhouse/`) is healthy on **8010**. What is still missing is `orthancteam/orthanc`: nine pulls
> have stalled at "Pulling fs layer" and never completed a single layer, so the stack runs as
> `docker compose up -d --no-deps api` and there is no C-STORE → ingest → audit path to show. Say that in shot 8; do
> not imply the Orthanc box exists.
>
> **Latency you will see on camera:** the container answers a study in **7–14 s** on this laptop (4 vCPU, 2 torch
> threads inside Docker Desktop's VM). The same code and weights in the host venv answer in **0.42 s**. Record with the
> container — it is the deployable artefact — and use the wait to talk. The venv fallback at the end of this section
> keeps the recording honest if the container misbehaves.

Start the stack:

**Terminal A**
```bat
cd C:\Users\kumar\Documents\Qoder\2026-10-22\410be6bc\AIML-02-medical-imaging
set API_PORT=8010
docker compose up -d --no-deps api
docker ps --format "{{.Names}}  {{.Status}}  {{.Ports}}"
curl -s http://localhost:8010/health
```

Expected: `aiml02-api  Up About an hour (healthy)  0.0.0.0:8010->8000/tcp` and
`{"status":"ok","model_loaded":true,"device":"cpu", ...}`.

**Terminal B**
```bat
cd C:\Users\kumar\Documents\Qoder\2026-10-22\410be6bc\AIML-02-medical-imaging\frontend
set VITE_API_TARGET=http://localhost:8010
npm run dev
```

**Fallback if the container will not start** — same predictions, 0.4 s per study, no Docker:

```bat
cd C:\Users\kumar\Documents\Qoder\2026-10-22\410be6bc\AIML-02-medical-imaging
C:\Users\kumar\venvs\aiml02-py312\Scripts\activate
set MODEL_PATH=artifacts\densenet121_lungopacity.pt
set MODEL_VERSION=demo-2026-10-27
python -m uvicorn src.api.main:app --port 8011
```

---

## Shot list

### Shot 1 — 0:00–0:40 · Opening (slide 1)

> "Twenty-six thousand six hundred and eighty-four chest X-rays. A model that ranks them at 0.88 AUROC. And a
> question I could not answer for most of this internship: does the model look at the lungs, or at the corners of the
> image? This is AIML-02 — an AI-assisted medical imaging system, built as an internship capstone. It is for
> education only. It is not a medical device."

Hold on the title slide. Do not read the ID number out loud.

### Shot 2 — 0:40–2:10 · Problem + the two tracks (slides 2–3)

> "The brief had two tracks and I did both, on purpose. Track A is classification: is there a lung opacity in this
> chest X-ray. Track B is segmentation: find the spleen in a 3D abdominal CT. They look unrelated — one is a picture,
> one is a volume — but they fail in the same way. A model that scores well and explains nothing is not usable by a
> radiologist, so both tracks are judged on more than their headline number."

Point at slide 3's fact table: RSNA 26,684 DICOM files versus MSD 41 CT volumes, one image per patient in RSNA
(so the split can be grouped by patient), binary mask versus bounding boxes.

### Shot 3 — 2:10–3:40 · Track A numbers (slides 4–5)

> "DenseNet121, ImageNet transfer learning, trained on a free Kaggle T4 for 8 epochs with early stopping at 4 — about
> two GPU-hours. Test AUROC is 0.8819 on 4,003 held-out images, confidence interval 0.8697 to 0.8916. The bar I had to
> beat was a pixel-intensity baseline at 0.740; that is plus 0.142.
>
> The useful comparison is the middle bar: frozen backbone features with a logistic-regression head score 0.7250 —
> it barely beats the pixel baseline. That number is on a 1,000-image subset, so treat it as a direction, not a
> ranking. The direction is that fine-tuning is what produced the signal, not the backbone on its own.
>
> At the operating threshold, 0.5108, sensitivity is 0.7661 and specificity is 0.8055. That means roughly one in four
> true opacities is missed, which is why this cannot be a screening tool. Accuracy would have sounded better. It would
> also have been the wrong headline, because the classes are unbalanced."

### Shot 4 — 3:40–5:00 · Explainability, calibration, fairness (slides 6–7)

> "Three checks that a leaderboard score does not do.
>
> Grad-CAM: I scored 300 heatmaps for how much of their mass lands inside the body. Mean is 0.54, and 43 percent put
> less than half their attention inside the patient. So the model is right more often than it is looking in the right
> place — that is a real finding, not a rounding error.
>
> Calibration: the model is over-confident. Expected calibration error 0.1256, and the average prediction is 0.35
> where the true prevalence is 0.23.
>
> Fairness: by sex, 0.892 against 0.874 — a gap that the confidence intervals do not separate. By age band, 0.895,
> 0.891, 0.843 — the oldest band is weaker. And the one that surprised me: the same model scores about 0.99 AUROC on
> 'lung opacity' labelled images and about 0.67 on 'no opacity' labelled ones. That is not a demographic gap, it is a
> label-semantics gap. The label 1 in RSNA means 'any lung opacity', not 'pneumonia'. A model can nail that label and
> still be useless for a pneumonia question."

### Shot 5 — 5:00–6:30 · Live API (Terminal A)

**Type (or paste) in Terminal A — a second window, API already running:**

```bat
curl -s http://localhost:8010/health
```

Real output:

```
{"status":"ok","model_loaded":true,"device":"cpu","disclaimer":"For education only. Not for clinical use."}
```

> "Two endpoints matter. Health tells you whether the model actually loaded — not just whether the process is up.
> And model-info states the label semantics in the API itself, so a client cannot misuse it by accident."

```bat
curl -s http://localhost:8010/model-info
```

Real output:

```
{"model":"densenet121","version":"unversioned","task":"binary lung-opacity detection on chest X-ray",
 "label_semantics":"1 = lung opacity present; 0 = normal OR another abnormality",
 "split_policy":"grouped split on patient_id (one image per group in RSNA stage_2)",
 "disclaimer":"For education only. Not for clinical use."}
```

**Now the prediction — a study the model has never seen, straight from the held-out split:**

```bat
curl -s -F "file=@data/raw/rsna/stage_2_train_images/019d950b-dd38-4cf3-a686-527a75728be6.dcm" http://localhost:8010/predict
```

Real output from the container (trimmed to the fields that matter) — it arrives after 7–14 s:

```
"prediction":"opacity","probability":0.8685,"threshold_used":0.5107557773590088,
"model_version":"unversioned","inference_ms":7135.5,
"dicom_tags":{"patient_id":"019d950b-dd38-4cf3-a686-527a75728be6","modality":"CR","rows":1024,"columns":1024, ...}
```

> "0.8685 against a threshold of 0.5108 — opacity. The response carries the DICOM tags back, the model version, and
> the threshold that was actually used, so a decision can be reconstructed later. Seven seconds is what CPU inference
> costs inside Docker Desktop's VM on a four-core laptop; the same weights from the host's Python are half a second,
> so the gap is the VM, not the model."

**The negative:**

```bat
curl -s -F "file=@data/raw/rsna/stage_2_train_images/000924cf-0f8d-42bd-9158-1af53881a557.dcm" http://localhost:8010/predict
```

Real output:

```
"prediction":"no_opacity","probability":0.1826,"threshold_used":0.5107557773590088,"inference_ms":9380.5, ...
```

> "Same endpoint, same threshold, 0.1826 — no opacity. The threshold is not something I tuned per demo: it comes out
> of the checkpoint as the operating point."

### Shot 6 — 6:30–7:30 · The viewer (browser)

Switch to `http://localhost:5173`. **Upload in this order — the PNGs first**: a `.dcm` upload gets a card and its tags,
but the browser cannot draw the DICOM pixels, so that panel reads "Browser preview not available". The PNG path is what
shows the actual radiograph.

```
%TEMP%\aiml02_test\positive.png     -> image + "opacity - 86.9%"
%TEMP%\aiml02_test\negative.png     -> image + "no_opacity - 18.3%"
%TEMP%\aiml02_test\positive.dcm     -> tags + Approve
```

> "This is the radiologist-facing part. Upload a study, and the viewer calls the same endpoint through a dev proxy, so
> what you see here is the service, not a mocked response."

Upload `positive.png`:

> "86.9 percent, opacity. The education banner is on screen permanently — 'For education only, not for clinical use' —
> and the 60 percent overlay slider is where a Grad-CAM heatmap would fade in."

Upload `negative.png`:

> "18.3 percent, no opacity. Two studies in the list now, each with its own decision state."

Upload `positive.dcm`, then click **Approve**:

> "Same study as a DICOM: 86.9 percent with the real tags read out of the file — patient ID, CR, 1024 by 1024. This
> one shows tags instead of pixels, because the viewer does not decode DICOM, and I would rather show that than hide it.
>
> The heatmap toggle stays disabled and says why: '(none from API yet)'. The API returns a probability, not a Grad-CAM
> overlay. A disabled control with an honest label beats a toggle that does nothing.
>
> Approve moves the study to 'approved'. Note the line underneath: decisions live in this session only until the audit
> endpoint is wired to `audit.db` — that is the row still red in my README. Reject shares the same handler; I am not
> clicking it here to keep the recording short."

### Shot 7 — 7:30–8:30 · Track B and the bug (slides 8–9)

> "Track B is the spleen. The first run scored 0.329 Dice after 40 epochs — a result I could have shipped quietly,
> because 0.329 beats the zero-Dice floor that a broken model produces. Instead I looked at the predictions, and they
> were all a blob near the middle of the window.
>
> The cause was the sampler, not the network. It took positives from a voxel inside the spleen and negatives from the
> exact centre of the volume. The cheapest way to satisfy that is to learn 'something bright near the middle'. It did.
>
> I jittered the positive centres and drew negatives uniformly. Same architecture, same data, same seed, 90 epochs,
> and the test Dice went to 0.8586 — median 0.9124, range 0.597 to 0.948, no empty masks, zero cases below 0.5. Two
> GPU-hours.
>
> The mean hides one failure: five of six volumes score 0.846 or better and land within 3 to 13 percent of the
> reference volume. The sixth predicts 159 millilitres against a reference of 357 — it under-segments badly. That is
> the case I would show a clinician, not the 0.86."

### Shot 8 — 8:30–9:20 · What is not done (slides 10–12)

> "Now the part that is usually skipped. The Orthanc wiring is written but not wired: the ingest poller and the SQLite
> audit trail exist in the repo, and they have never pulled a DICOM, because the Docker engine on this machine would
> not start and the Orthanc image could not be pulled. The API image itself was built offline from a local wheelhouse
> and does serve predictions — 2.09 gigabytes, running — but it runs with `--no-deps` and the compose stack is
> incomplete. That is a red row in my README, deliberately.
>
> The viewer verification is DOM-level: the in-app browser was hidden during that session, so I have the values and no
> screenshot. I am telling you that rather than showing a screenshot from somewhere else.
>
> Six held-out CT volumes is a small test set. A 0.13 calibration error is over-confident. And there is no regulatory
> clearance of any kind — this is a capstone, not a device."

### Shot 9 — 9:20–10:00 · Close (slide 13)

> "Four things I would do next, in this order. Train the spleen model longer — validation was still improving at epoch
> 79, so 100 to 200 epochs is the cheapest path to 0.90. Calibrate the classifier with temperature scaling; it is a
> one-evening fix for a 0.13 ECE. Finish the Orthanc wiring so a DICOM arriving at a server triggers the whole chain
> without a human uploading a file. And validate outside RSNA, because one hospital's images are one distribution.
>
> The number I trust most is not 0.8819. It is that the same model scores 0.99 on 'opacity' labels and 0.67 on 'no
> opacity' labels — because that is the number that told me what the model actually learned. Thank you."

---

## Timing table

| Shot | Content | Start | Length |
|---|---|---|---|
| 1 | Title, the question | 0:00 | 0:40 |
| 2 | Two tracks, why both | 0:40 | 1:30 |
| 3 | AUROC 0.8819 and the +0.142 | 2:10 | 1:30 |
| 4 | Grad-CAM, calibration, fairness | 3:40 | 1:20 |
| 5 | Live `/health`, `/model-info`, two `/predict` calls | 5:00 | 1:30 |
| 6 | Viewer: 18.3 %, 86.9 %, Approve | 6:30 | 1:00 |
| 7 | The sampler bug, 0.329 → 0.8586 | 7:30 | 1:00 |
| 8 | Honest gaps | 8:30 | 0:50 |
| 9 | Next steps, close | 9:20 | 0:40 |

If you run long, cut shot 4 to the label-semantics paragraph only. Never cut shot 8.

---

## Recovery lines (if something breaks on camera)

| Problem | What to do |
|---|---|
| `Address already in use` on 8010 | `netstat -ano \| findstr :8010`; usually the old container — `docker compose down` then up again |
| `Address already in use` on 5173 | A leftover Vite is running; open `http://localhost:5173` anyway — if it loads, keep going |
| Browser cannot reach the API | Use `localhost`, not `127.0.0.1`; Vite binds IPv6. Check `set VITE_API_TARGET=http://localhost:8010` in Terminal B |
| `/predict` returns 503 `no model at …` | (venv fallback only) `MODEL_PATH` is relative — the shell must be in the repo root, not `frontend/` |
| First request takes longer than 14 s | The machine is loaded (a docker pull, a Vite boot). Wait it out; if it passes 60 s, restart the container |
| Docker question from a reviewer | "The API runs in its own image; Orthanc never pulled — nine attempts stalled at the first layer — so the stack is `--no-deps api` and ingest is unwired. That is row 4 in the README, still yellow." |

## Facts you may not improvise

Quote only from `results/`: `densenet_metrics.json`, `gradcam_fairness_summary.json`, `calibration.json`,
`spleen_metrics.json`, `spleen_per_case.csv`. Do not say "about 90 %", "nearly perfect", "clinically validated",
"FDA", or any accuracy figure that is not the ones above. Do not present the Grad-CAM tour as evidence the model
looks at the right thing — 43 % of maps put less than half their attention inside the body.
