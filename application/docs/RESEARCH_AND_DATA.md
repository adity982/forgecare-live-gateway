# Real-image research: automotive line, cast parts, and phone screens

## Primary automotive study: Renault AutoVI pipe clips

The [AutoVI dataset](https://zenodo.org/records/10459003) was collected by Renault Group and Université de technologie de Compiègne on real automotive production lines. The `pipe_clip` category contains 195 acceptable training photos, 195 acceptable test photos, and 142 staged defect photos. Renault reports that the staged defects were corrected after photography. The category is licensed CC BY-NC-SA 4.0; keep this copy and any derivative research artifacts local for noncommercial educational use, with attribution.

The downloaded ZIP, extracted source images, source hash and license note are at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\autovi_pipe_clip`. The trained Anomalib PaDiM checkpoint, source image SHA-256 manifest, metrics, and example anomaly overlays are at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\outputs\ForgeCareAI-Release\research\autovi_pipe_clip_20261105`. The checkpoint is under `results\Padim\AutoVI\pipe_clip\v0\weights\lightning\model.ckpt`.

This model fitted normal-only 256-pixel images through a ResNet18 feature extractor and PaDiM with 50 sampled feature dimensions. The seeded evaluation split used 169 held-out photos. The business result is direct:

| Outcome | Held-out photos | What it means |
| --- | ---: | --- |
| Defective pipe clips flagged for review | 67 of 71 | Most staged defects reached a worker |
| Defective pipe clips missed | 4 of 71 | These defects would pass this screening step |
| Acceptable pipe clips falsely flagged | 67 of 98 | A large unnecessary review workload |
| Acceptable pipe clips passed | 31 of 98 | No model alert |

The image ROC AUC was 0.633; pixel ROC AUC was 0.884. These technical scores do not replace the four work counts. The false-alert burden makes this candidate unsuitable for production decisions. **Models & experiments** shows a genuine held-out source image and its saved model overlay. The model has not run on the user's phone camera, factory camera, or a separate site. Reproduce the dataset acquisition and fit with `scripts\get_autovi_dataset.py` and `scripts\train_autovi_pipe_clip.py` using the prepared research environment.

### Actual Flower training on the automotive photos

A separate supervised study trained a 96×96 grayscale CNN on the same AutoVI `pipe_clip` category with **two actual in-process Flower clients and three FedAvg rounds**. This deliberately changes AutoVI's normal-only benchmark: staged defect labels from the original test folder enter the adapted training split. Its result is **not directly comparable** to the PaDiM result above. The script excluded 13 exact duplicate defect files before assigning 519 unique images to train, validation, and test; it kept neighbouring blocks of ten filenames in one split to reduce related-frame leakage. There were 358 training photos, 179 per client, 80 validation photos, and 81 held-out test photos. Both clients used three local epochs per round; the validation set chose the 0.45 alert threshold.

| Worker-facing result on 81 held-out photos | Count |
| --- | ---: |
| Staged defective clips sent to review | 19 of 21 |
| Staged defective clips missed | 2 of 21 |
| Acceptable clips sent to unnecessary review | 30 of 60 |
| Acceptable clips passed without alert | 30 of 60 |

The 0.786 image ROC AUC is a technical comparison value; the four counts show the operating burden. A 50% false-alert rate on acceptable clips makes this a research demonstration, not a production candidate. The source images remain at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\autovi_pipe_clip`. The saved global model (`flower_autovi_cnn.pt`, `flower_autovi_cnn_scripted.pt`), every unique source hash and split/client assignment (`split_manifest.csv`), timestamps, all round records, and metrics are at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\outputs\ForgeCareAI-Release\research\autovi_federated_20261105`. **Federation** now leads with its round-by-round start and finish timeline, and **Models & experiments** runs the saved weights on selectable held-out automotive photos.

From the release folder, use the prepared environment to reproduce or audit:

```powershell
..\..\work\forgecare-venv\Scripts\python.exe -m scripts.train_autovi_federated --data ..\..\work\datasets\autovi_pipe_clip --out research\autovi_federated_20261105
..\..\work\forgecare-venv\Scripts\python.exe scripts\verify_autovi_federated.py --run research\autovi_federated_20261105
```

Both clients and raw images were on the same laptop. The dataset has no independent factory-site partition or component identity; grouped nearby filenames reduce but do not eliminate related-part leakage. This run makes no cross-factory privacy, deployment, or production accuracy claim.

## Manufacturing and Flower study: cast pump impellers

The [Kaggle casting product dataset](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) supplies 1,300 unaugmented 512-pixel photos of **cast submersible-pump impellers**: 519 acceptable and 781 defective. These are real manufactured components and a useful surface-inspection example, but they are **not specifically automobile parts**. Its license is CC BY-NC-ND 4.0, so retain the dataset locally for educational research and do not redistribute it as part of the app.

The original photos and archive are at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\casting_impeller`. The source SHA-256 and licensing note are in `source.json`. The trained artifact, split manifest with every image SHA-256, and complete per-round timestamps and metrics are at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\outputs\ForgeCareAI-Release\research\casting_impeller_20261105`. The model files are `flower_casting_cnn.pt` and `flower_casting_cnn_scripted.pt`.

The fixed stratified split (`seed=42`) used 910 training, 195 validation, and 195 untouched test photos. Two in-process Flower clients trained on **455 different images each** for three FedAvg rounds, three local epochs per round, with a 96×96 grayscale CNN, Adam 0.001, batch size 32, and weighted binary cross-entropy. The validation set chose the 0.37 alert threshold; no test photo selected it. This is actual optimization and Flower aggregation, but both clients ran on this laptop and did not provide cross-factory privacy.

| What a plant manager can understand | Held-out result | Operational meaning if this pattern persisted |
| --- | ---: | --- |
| Defective parts sent for review | 96 of 117 | A worker receives an alert on these parts |
| Defective parts missed | 21 of 117 | These defects would pass this screening step |
| Acceptable parts sent for unnecessary review | 23 of 78 | Extra checking time and possible line delay |
| Acceptable parts passed without alert | 55 of 78 | No model alert on these parts |

In this test set, **119 of 195 parts would enter the worker's review queue**: 96 defective and 23 acceptable. The model gave no alert on 76: 55 acceptable and 21 defective. If checking each alert takes five minutes, the 119 alerts would require about 9.9 checking hours, including about 1.9 hours on acceptable parts. Five minutes is an editable demonstration assumption, not measured plant time. **Models & experiments** now lets a plant team enter its own recheck minutes and expected per-event costs to see the arithmetic for this test set. It does not call that estimate a saving or extrapolate to a shift. The 195-image ROC AUC is 0.827 and F1 is 0.814, kept under technical details. Actual cycle time, defect prevalence, scrap costs, and comparison with existing inspection were not measured. The misses and false alerts require a target-line pilot before the model could influence release decisions. The page can run the saved model on a genuine held-out part image; **Federation** shows all three completed rounds, start/finish times, and client checks.

Reproduce with `scripts\get_casting_dataset.py`, `scripts\train_casting_research.py`, and `scripts\verify_casting_research.py`. From the release folder, using the prepared research environment:

```powershell
..\..\work\forgecare-venv\Scripts\python.exe scripts\verify_casting_research.py --run research\casting_impeller_20261105
```

## Phone side study: screen cracks

The public [Kaggle cracked/intact smartphone sample](https://www.kaggle.com/datasets/axondata/cracked-and-intact-smartphone-images-dataset) downloaded here contains **71 usable images**: 34 intact and 37 cracked. The page advertises a larger collection, but that full collection is not present in the public download. The local images are at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\cracked_intact_phones`. Its license is CC BY-NC 4.0; keep the sample local for educational research.

The first phone model fitted a logistic classifier on frozen ImageNet ResNet18 features. Its stratified split had 42 train, 14 validation, and 15 untouched test images. On that small test it flagged **6 of 8 cracked screens**, missed 2, correctly passed 7 of 7 intact screens, and produced 0 false alerts. The first model and manifest remain at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\outputs\ForgeCareAI-Release\research\phone_crack_20261105`. A separate positive-only external check used the public [DataCluster cracked-screen sample](https://www.kaggle.com/datasets/dataclusterlabs/cracked-screen-dataset): 300 images downloaded, 299 readable, only **186 of 299** flagged. The sample has bounding boxes around the damaged screen, not each crack or dent. Its Kaggle metadata says CC0 while its description asserts ownership and invites separate licensing for a larger collection; keep it local for this educational evaluation. The local copy is at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\datacluster_cracked_screens`.

A second candidate was actually fitted using both public samples: **370 readable images** in all. It used balanced logistic regression on frozen ResNet18 features and source-stratified train/validation/test splits. Its 75-photo test contained only seven intact photos. It flagged **66 of 68 cracked screens**, missed 2, correctly passed 5 of 7 intact screens, and falsely flagged 2. The saved model used by the optional research preview in Capture review is `C:\Users\crazy\Documents\Codex\2026-11-03\ba\outputs\ForgeCareAI-Release\research\phone_crack_combined_20261105\phone_crack_screen_scripted.pt`; that folder also contains its `metrics.json` and SHA-256 split manifest. The improvement on cracked photos is visible, but the false-alert evidence is still very weak because of the seven intact images.

Reproduce with `scripts\get_phone_dataset.py` for each Kaggle reference, `scripts\train_phone_research.py --data <Axon-folder> --extra-cracked <DataCluster-folder> --out <candidate-folder>`, and `python -m scripts.evaluate_external_phone --data <DataCluster-folder> --run <first-model-folder>`. Verify the saved combined candidate from the release folder with `..\..\work\forgecare-venv\Scripts\python.exe -m scripts.verify_phone_research --run research\phone_crack_combined_20261105`; it hashes all 370 source files and recomputes every test prediction. Public samples lack physical-phone IDs, so related views may cross splits. There is no iPhone 14 Plus or realme-specific evaluation and no annotation for individual scratches or dents in those two sources. **Capture review** can run the saved classifier on a clear front-screen photo as a labeled research preview; it does not automatically set the review label. An optional Groq visual draft may describe tentative visible marks, but it is a separate vision service and still requires correction.

## Phone side study: localized damage and candidate counts

The [Roboflow Universe Mobile Phone Defect Detection version 30](https://universe.roboflow.com/pharmacy-shelf-medicine-detection/mobile-phone-defect-detection-faraf/dataset/30) is a public **CC BY 4.0** object-detection dataset with `Broken`, `Dent`, `Major Scratch`, and `Small Scratch` boxes. Its export is stored at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\roboflow_phone_damage_v30`; the original ZIP SHA-256 is `55d6fc58b5c303866ed70128f8e8a8694760309bb61dfd2a1b3d57ff4f0fd3b4`. Attribution to the dataset creator, Mobile Phone Defect, should accompany any public display of its images.

The supplied 1,686 files were audited before training. Exact duplicates crossed the supplied splits; 341 unique image hashes had conflicting box annotations; and frames of the same named video clip appeared in different splits. The prepared dataset at `C:\Users\crazy\Documents\Codex\2026-11-03\ba\work\datasets\phone_damage_grouped` excludes all conflicting copies and exact duplicate bytes. It groups the same original photo and named video clip into a single split and uses one copy per original photo in validation and test. The resulting split is **625 train, 107 validation, and 99 held-out test photos**. `preparation.json` and `split_manifest.csv` record every source file and decision. This improves the independence of the held-out check, but the source has no physical-phone IDs, so the study cannot prove performance across distinct phone owners or models.

The detector is a locally fine-tuned YOLO11n object detector. Its checkpoint, training log, threshold selection, box-level evaluation, and per-photo predictions are under `C:\Users\crazy\Documents\Codex\2026-11-03\ba\outputs\ForgeCareAI-Release\research\phone_damage_20261105`. On the 99-photo held-out check, it correctly located **20 of 45 Broken, 0 of 14 Dent, 9 of 24 Major Scratch, and 10 of 20 Small Scratch** labeled regions; the corresponding extra box counts were 35, 0, 4, and 20. Therefore, **dent detection is disabled in the local camera preview**. The remaining boxes are research suggestions only, and the technician must inspect and correct the review. In Capture review, the model's actual per-photo inference times, per-view table, and overlays are displayed. It prefills an editable note and a provisional scratch field using the largest single-view suggestion, without adding repeated views. The optional Groq image analysis sends up to three selected photos and returns a structured editable description with a scratch or dent count only when it can identify separate marks; uncertain counts stay blank. Its counts are likewise unvalidated. The technician's confirmation checkbox is required before any AI-assisted value becomes a saved label. An independent screen-level check on the separate Kaggle cracked/intact sample raised a Broken alert on **35 of 37 cracked-screen** and **2 of 34 intact-screen** photos, with no exact image-hash overlap with the Roboflow preparation. That check does not establish accurate box locations or scratch/dent counts. The held-out Roboflow source is almost entirely defect photos, so it cannot establish the false-alert rate on clean phones. No labeled test was run on the user's iPhone 14 Plus or realme narzo 60.

The Roboflow API key used for the export is kept in the Windows user environment and in a **private laptop-only document** at `C:\Users\crazy\AppData\Local\ForgeCareAI\private\roboflow-credentials.txt`. The key is not in the dataset, project source, reports, or this research document. To reproduce on this laptop, use `requirements.txt` and then `requirements-vision.txt`; run `scripts\get_roboflow_phone_dataset.py`, `scripts\prepare_phone_damage_dataset.py`, `scripts\train_phone_damage_detector.py`, `scripts\evaluate_phone_damage_detector.py`, `scripts\evaluate_external_phone_damage.py`, `scripts\report_phone_damage.py`, and `python -m scripts.verify_phone_damage_research` in that order. The downloader reads `ROBOFLOW_API_KEY` from the process environment.

The public dataset requires CC BY 4.0 attribution. [Ultralytics states that its YOLO code and trained models use AGPL-3.0 by default](https://www.ultralytics.com/license); check those terms before distributing this detector inside a closed product. This run is a local educational research prototype.

## Earlier comparison: VisA capsules

This is a completed **manufacturing-image experiment**, separate from ForgeCare's operational phone inspection workflow. Amazon's [VisA release](https://github.com/amazon-science/spot-diff) contains real object images and anomaly masks. The extracted `capsules` category has **602 normal and 100 anomalous images**, matching the source's inventory. The dataset is [CC BY 4.0](https://github.com/amazon-science/spot-diff/blob/main/LICENSE-DATASET). Keep this attribution when presenting images.

## Where the data and models are on this laptop

All paths below are relative to `C:\Users\crazy\Documents\Codex\2026-11-03\ba`:

| Item | Path | What it proves |
| --- | --- | --- |
| 702 real source photos | `work\datasets\visa\capsules\Data\Images\Normal` and `...\Anomaly` | Public manufacturing images used for training and held-out evaluation |
| 100 source masks | `work\datasets\visa\capsules\Data\Masks\Anomaly` | VisA localization reference; these masks were not used to train the small CNN |
| Source labels | `work\datasets\visa\capsules\image_anno.csv` | Original image and mask association |
| Split and client audit | `outputs\ForgeCareAI-Release\research\visa_capsules_20261105\split_manifest.csv` | Source path, SHA-256, label, split, and Flower client ID for every image |
| Normal-only patch model | `outputs\ForgeCareAI-Release\research\visa_capsules_20261105\patch_baseline.joblib` | KMeans patch anomaly baseline and threshold |
| Federated model | `outputs\ForgeCareAI-Release\research\visa_capsules_20261105\flower_fedavg_cnn.pt` and `flower_fedavg_cnn_scripted.pt` | Saved global CNN after three real Flower FedAvg rounds; the scripted copy powers read-only test-photo inspection |
| Anomalib model | `outputs\ForgeCareAI-Release\research\visa_capsules_20261105\anomalib_padim\results\Padim\visa_capsules\v0\weights\lightning\model.ckpt` | Real PaDiM ResNet18 fit on a separate VisA subset |
| Metrics and images | `outputs\ForgeCareAI-Release\research\visa_capsules_20261105\metrics.json`, `anomalib_padim\anomalib_metrics.json`, and `anomalib_padim\results\Padim\visa_capsules\v0\images` | Measured results and example anomaly maps |

The original full VisA archive was only partially downloaded, so it was removed after the complete `capsules` category was extracted and every retained photo was decoded and hashed. The other 11 VisA categories are **not** installed. The public archive URL is linked from the VisA repository above. Dataset images are outside the release folder so the app package remains smaller; copying only the release folder will not copy the training images. In the GUI, **Models & experiments → Inspect a held-out manufacturing photo** displays any installed test image, a patch-distance overlay, and both saved model scores without writing an operational case.

## What was trained

The first pipeline used the 702-image category with a fixed stratified 421/140/141 train/validation/test split (`seed=42`). The patch model learned 96 normal-image patch prototypes from 361 training normal images and used the 95th percentile of validation-normal scores as its alert threshold. It writes two example overlays.

The CNN used **two disjoint real-image partitions** of 210 and 211 training images. Its 64×64 RGB input feeds 12- and 24-channel convolution blocks with ReLU/max-pooling and one output logit. Each `flwr.client.NumPyClient` trained locally for two epochs per round using Adam (`learning_rate=0.002`, `batch_size=32`) and binary cross-entropy. `flwr.server.strategy.FedAvg` aggregated the two returned weight updates over **three rounds**. Only the validation split selected a threshold from 0.1–0.9 for F1; 141 held-out images produced the reported test metrics. The saved manifest marks exactly which images belong to `client-1` and `client-2`. This is real federated optimization on **one laptop with in-process clients**. It does not demonstrate different factories, network isolation, or a privacy guarantee. The clients' raw images are present together on this research laptop.

The actual [Anomalib PaDiM](https://github.com/open-edge-platform/anomalib/blob/main/src/anomalib/models/image/padim/README.md) run used a pretrained ResNet18 feature extractor (`n_features=50`, `train_batch_size=8`, CPU), 100 normal train images, and 120 held-out images (80 normal, 40 anomalous) from VisA. Anomalib made its own 50/50 validation/test division within that held-out pool. Its score is therefore **not directly comparable** to the other two models' 141-image test. The generated anomaly map beside the original part image is visible in **Models & experiments**.

| Research model | Evaluation | Measured result |
| --- | --- | --- |
| Patch KMeans | 141 test images, 20 anomalous | ROC AUC 0.618; found 2/20 anomalies; 5/121 false alerts |
| Flower CNN, 3 FedAvg rounds | Same 141 test images | ROC AUC 0.587; found 13/20 anomalies; 61/121 false alerts |
| Anomalib PaDiM ResNet18 | Its separate internal holdout | Image ROC AUC 0.748; image F1 0.457 |

These are initial research results. In particular, neither of the first two meets a sensible operational alert gate. No trained research weight is approved for the live phone camera. Anomalib's image-level result does not establish pixel localization quality or phone performance. The random image split may include related-looking photos across splits; a production evaluation needs independent lots, sites, lighting, and camera hardware.

## Reproduce or verify

The prepared laptop currently has the research libraries installed in `..\..\work\forgecare-venv`. On a clean Python 3.12 computer, create a **separate** research virtual environment with `requirements-research.txt`; do not change the app's `requirements.txt` just to run experiments.

From the release folder in PowerShell:

```powershell
py -3.12 -m venv .research-venv
.\.research-venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.research-venv\Scripts\python.exe scripts\train_visa_research.py --data ..\..\work\datasets\visa\capsules --out research\visa_capsules_20261105
.\.research-venv\Scripts\python.exe scripts\train_anomalib_visa.py --manifest research\visa_capsules_20261105\split_manifest.csv --out research\visa_capsules_20261105\anomalib_padim
.\.research-venv\Scripts\python.exe scripts\verify_visa_research.py --run research\visa_capsules_20261105
```

The Anomalib run can download ResNet18 pretrained weights on first execution. Training again will overwrite the run outputs. To **show** the current evidence without retraining, run only `verify_visa_research.py` or open **Models & experiments** and **Federation** in the app. The verifier checks all 702 hashes and image decodes, exact split/client counts, metric arithmetic, and model files.

## GitHub reference decisions

| Reference | Adopted here | Boundary |
| --- | --- | --- |
| [Anomalib](https://github.com/open-edge-platform/anomalib) | Actual PaDiM fit and visual anomaly maps | Research category only |
| [Flower](https://github.com/flwrlabs/flower) | Actual local client updates and weighted FedAvg aggregation | In-process clients on one laptop |
| [FedAD](https://github.com/mirqr/FedAD) | Federated anomaly detection comparison idea | The repository is a reference; its code/model is not used in this run |
| [Maintenance Wizard AI](https://github.com/yuvrajjuv/maintenance-wizard-ai) | Equipment-health and maintenance workflow comparison | ForgeCare's queue, approval, audit, and measured time remain its own implementation |
| [Industrial Knowledge RAG](https://github.com/ten10do/industrial-knowledge-rag) | Source-backed guidance and answer abstention comparison | ForgeCare keeps its own approved-source RAG; no copied corpus or model |
| [Car Damage Detector](https://github.com/joeltadeu/car-damage-detector) | Mobile deployment and camera UX reference | Car-trained weights are not used to label phone damage |

## Future scratch and dent work

The [MSD phone-screen dataset](https://github.com/jianzhang96/MSD) is a plausible follow-on: 1,200 images of manufactured phone screens with oil, scratch, and stain defects plus only 20 extra normal images. Its 1.57 GB archive and industrial-screen setting do not provide a reliable exterior-dent dataset for the **iPhone 14 Plus** and **realme narzo 60**. The phone crack classifier above and the VisA capsule weights must not be interpreted as phone scratch/dent counts. ForgeCare continues to require technician review for those subjects.

A read-only audit of this laptop's operational database on **2026-11-05** found 158 quality-accepted iPhone captures, four iPhone human labels (one `none`, two `significant`, one `unclear`), and three iPhone cases (two closed, one open). The realme has 13 captures, all marked `blurred`, and one `unclear` human label. These are frames of individual subjects, not independent positive/negative physical-phone samples for model training. The three clearly fictional cases in **Capture review** let an operator practice correcting a draft without turning examples into real inspection or training records. The database passed SQLite integrity and foreign-key checks during this audit.
