# ☀️ Rooftop Solar PV Detection using AI & Satellite Imagery

## 📘 Overview

This project focuses on detecting **rooftop solar photovoltaic (PV) installations** using **AI-driven computer vision** and **satellite imagery**. It automatically identifies solar panels on rooftops, estimates their total area, and provides verification-ready outputs.

The goal is to create a scalable, accurate, and explainable solution for solar rooftop mapping and renewable energy analysis.

---

## 🧭 Key Objectives

* Fetch satellite imagery for any given geographic coordinate.
* Detect rooftop solar PV panels using a trained ML model.
* Estimate total PV area (in square meters) and model confidence.
* Generate structured JSON results with masks/bounding boxes.
* Support large-scale, automated mapping workflows.

---

## 🧩 System Workflow

```text
Input Coordinates (.xlsx)
        │
        ▼
 Imagery Fetching (Google Maps API)
        │
        ▼
  AI Model Inference
        │
        ▼
 Post-Processing (Area, Confidence, QC)
        │
        ▼
 JSON Output + Visual Overlays
```

---

## ⚙️ Tech Stack

| Component        | Technology                              |
| ---------------- | --------------------------------------- |
| Language         | Python 3.10+                            |
| ML Framework     | PyTorch / TensorFlow                    |
| Image Processing | Pillow (PIL), OpenCV                    |
| Data Handling    | pandas, openpyxl                        |
| API Integration  | requests                                |
| Map Provider     | Google Maps Static API / Tiles API      |
| Optional Data    | Google Earth Engine, ESRI World Imagery |

---

## 🚀 Setup Instructions

```bash
git clone https://github.com/<your-repo-name>.git
cd rooftop-pv-detection

python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 🔑 Configure API Key

Set your Google Maps API key as an environment variable:

```bash
export GOOGLE_MAPS_API_KEY="YOUR_API_KEY"
```

**Enable these APIs in Google Cloud Console:**

* Maps Static API ✅
* Maps Tiles API ✅
* Aerial View API (optional)

Restrict your key to your IP or domain for security.

---

## ▶️ How It Works

1. **Read Input** → An Excel file with `sample_id`, `lat`, `lon`.
2. **Fetch Imagery** → Get satellite images via Google Maps APIs.
3. **Inference** → Run detection/segmentation model to locate solar PVs.
4. **Post-Processing** → Estimate area, compute confidence, set QC status.
5. **Output** → Save structured JSON and overlay visualizations.

---

## 🧾 Example Output

```json
{
  "sample_id": 101,
  "lat": 19.0760,
  "lon": 72.8777,
  "has_pv": true,
  "confidence": 0.94,
  "pv_area_sqm_est": 24.3,
  "qc_status": "VERIFIABLE",
  "bbox_or_mask": "<encoded-mask>",
  "image_metadata": {
    "source": "GoogleStaticMaps",
    "zoom": 20,
    "capture_date": "N/A"
  }
}
```

---

## 📁 Repository Structure

```text
.
├── pipeline/
│   ├── main_inference.py        # main entry point
│   ├── fetch_image.py           # API image fetching logic
│   ├── model_inference.py       # ML model inference
│   ├── postprocess.py           # area & confidence estimation
│   └── utils/                   # helper functions
│
├── models/
│   └── solar_detector.pt        # trained model weights
│
├── artefacts/
│   └── overlays/                # verification images
│
├── data/
│   └── coordinates.xlsx         # input data
│
├── results/
│   └── predictions.json         # output results
│
├── requirements.txt
└── README.md
```

---

## 🧮 Buffer Zones (Real-World Scale)

| Buffer | Area (sq.ft) | Area (m²) | Approx Radius (m) |
| :----- | -----------: | --------: | ----------------: |
| Inner  |         1200 |     111.5 |          ≈ 5.96 m |
| Outer  |         2400 |     223.0 |          ≈ 8.43 m |

These are converted to pixel radii dynamically based on latitude and zoom using the Web Mercator projection.

---

## 🌱 Future Work

* Integrate **Google Earth Engine** for time-series and change detection.
* Add **multi-spectral analysis** for improved accuracy.
* Build a **REST API** or web dashboard for real-time inference.
* Automate retraining using feedback loops.

---

## 🧑‍💻 Contributing

Contributions are welcome! Please follow PEP8, keep commits descriptive, and ensure reproducibility.

---
