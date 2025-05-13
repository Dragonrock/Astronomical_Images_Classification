# Astronomical Image Classification – User Manual

This repository contains the **`Astro_Classifier.ipynb`** notebook and auxiliary scripts used to build an end‑to‑end pipeline for **star vs galaxy classification** with Sloan Digital Sky Survey (SDSS) imaging.

---

## 1. Quick start

```bash
# 1. Clone the project
git clone https://github.com/Dragonrock/Astronomical_Images_Classification
cd Astronomical_Images_Classification

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Unzip any dataset files
unzip dataset
unzip astro_processing/denoised_dataset.zip
unzip astro_processing/edge_maps.zip
# 4. Launch the notebook
jupyter notebook Final.ipynb
```

All outputs are written **relative to the project root**, so you can run everything as‑is.

---

## 2. Directory layout created by the pipeline

```
.
├─ dataset/                       # Raw images & original metadata
│  ├─ STAR/  └─ GALAXY/           # JPG cut‑outs (128×128)
│  └─ metadata.csv                # One row per object
├─ astro_processing/
│  ├─ denoised_dataset/           # Noise‑reduced images
│  ├─ edge_maps/                  # Canny edge maps
│  └─ features/                   # ⚙︎ CSV feature tables
│     ├─ features_step1.csv
│     ├─ features_step2.csv
│     └─ features_step3.csv
└─ models/                        # Trained classifiers 
```

---

## 3. Chapter‑by‑chapter guide

Each **chapter** in the notebook is a self‑contained block you can *Run All* without touching the rest.
Parameters that are safe to tune are exposed as **UPPERCASE** variables at the top of the corresponding code cell.

| Chapter                               | Purpose                                                                                                                                | Key tunable parameters                                                                                  | Main outputs                                                                           |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| **0. Data Preparation**               | Fetch a balanced sample of SDSS stars & galaxies, download 128×128 JPG cut‑outs, build `metadata.csv`, and zip everything for sharing. | `TOP` (rows per class, default 2000), `PIXEL_SCALE` & `IMAGE_SIZE` in the SQL URL.                      | `dataset/STAR/`, `dataset/GALAXY/`, `dataset/metadata.csv`, `dataset/sdss_dataset.zip` |
| **1. Data Loading & Noise Reduction** | Load the raw images, perform Non‑Local Means denoising, compute Power‑Spectrum Analysis (PSA) features.                                | `PATCH_SIZE`, `PATCH_DISTANCE`, `H` (filter strength) inside `NLMeans` class; FFT window size in `PSA`. | Denoised images in `astro_processing/denoised_dataset/`, `features_step1.csv`          |
| **2. Edge Feature Extraction**        | Apply Sobel, Canny, Scharr, and Laplacian operators; measure texture statistics; merge with PSA & colour indices.                      | Edge detector thresholds (`THR_CANNY`, etc.), `GAUSSIAN_SIGMA`.                                         | Edge maps in `combined_edges/`, `features_step2.csv`                                   |
| **3. Segmentation & Extra Features**  | Run Chan–Vese, Otsu, and watershed segmentations; extract morphology (area, eccentricity, fractal dimension) and intensity stats.      | `CV_ITER`, `THRESH_OTSU`, `WATERSHED_MIN_DISTANCE`, etc.                                                | `features_step3.csv` (final master table)                                              |
| **4. Gradient‑Boosting Classifier**   | Train a GradientBoostingClassifier on the engineered features, evaluate accuracy, save scaler & model.                                 | `n_estimators`, `learning_rate`, `max_depth`                                                            | `models/gb_model.joblib`, `models/scaler.joblib` (saved via `joblib.dump`)             |
| **5. Processing a New Image**         | End‑to‑end inference pipeline: query SDSS for a fresh object, denoise, generate features, scale, and predict class label.              | Same parameters as Ch. 0 & Ch. 1 plus `QUERY` in the SQL template.                                      | Prints predicted class; optional denoised cut‑out saved alongside raw image.           |
| **6. Edge‑Detector Evaluation**       | Quantitatively compare edge maps with density, connectivity, entropy, and noise ratios.                                                | Metric thresholds inside `edge_metrics`.                                                                | In‑memory results & plots (no new files).                                              |
| **7. Classification Evaluation**      | Benchmark multiple classifiers (RBF‑SVM, PCA‑SVM, RandomForest, k‑NN, etc.) and summarise results.                                     | Hyper‑parameters inside each model definition.                                                          | Console metrics & confusion matrices; no CSVs written.                                 |

> **Tip** If you only need the trained model, jump directly to Chapter 4 after running Chapters 0–3.

---

## 4. Customising the pipeline

* **Change sample size or object classes** in Chapter 0 by editing the SQL `QUERY` and `CLASSES` dictionary.
* **Swap noise‑reduction strategy** in Chapter 1 (e.g. use median filtering instead of NL‑Means).
* **Add new feature blocks** by creating additional CSVs in `astro_processing/features` and merging them at the end of Chapter 3.

---

## 5. Saving & re‑using models

Models are saved with `joblib.dump()` under `./models/`.
Load them later with:

```python
from joblib import load
model  = load('models/gb_model.joblib')
```

---



## 6. License & citation

Code released under the MIT License.
If you use this pipeline in an academic work please cite the SDSS data release and this repository.
