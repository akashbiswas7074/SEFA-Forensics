# Self-Eliciting Forensic Anchors (SEFA-Forensics)
> **Unlocking Intrinsic Forgery Sensitivity in Frozen Foundation Models for Universal Subtle Manipulation Detection**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Trainable Params](https://img.shields.io/badge/Trainable%20Params-%3C0.2%25-brightgreen.svg)]()
[![Explainability](https://img.shields.io/badge/Explainability-Dual%20Heatmap%20%2B%20Grounded%20Report-orange.svg)]()

---

## 📌 Executive Summary

Modern image manipulation detection faces a critical bottleneck: detectors either overfit to coarse visual artifacts or rely on heavy, compute-intensive fine-tuning. When confronted with state-of-the-art **latent diffusion inpainting** and **subtle retouching**, existing detectors collapse. Moreover, traditional detectors offer black-box predictions without actionable explanations.

**SEFA-Forensics** introduces a new paradigm:
Instead of adding heavy adapters or fine-tuning foundation models, **we treat the frozen vision foundation model (DINOv2 / ViT) as already possessing intrinsic "forensic intelligence."** 

### Key Innovations:
1. **100% Frozen Backbone ($\nabla \theta_{\text{backbone}} = 0$)**: Zero gradients flow through the heavy vision foundation model.
2. **Controlled Evidence Stress $\mathcal{S}(x)$**: Forces the model to reveal anomalous latent response residuals ($\Delta Z$) in manipulated regions, while natural image patches remain invariant.
3. **Statistical Anchor Discovery**: Automatically mines the internal units (tokens/channels) with maximal **Intrinsic Forgery Sensitivity Score (IFSS)** using only a few dozen calibration pairs.
4. **Ultra-Lightweight Router (<0.2% Parameters)**: A tiny bottleneck routing module (~4,500 parameters) trains in minutes on a standard consumer GPU (e.g., RTX 3050).
5. **Explainable Dual Output**: Yields both high-resolution pixel localization heatmaps and **natural-language justifications** grounded directly in the activated forensic anchors.

---

## 📂 Project Directory Structure

```text
SEFA-Forensics/
├── configs/
│   └── default_config.yaml           # Central hyperparameters & training configuration
├── data/
│   ├── raw/
│   │   ├── real/                     # Curated pristine natural photographs
│   │   ├── forged/                   # Subtle diffusion-inpainted images
│   │   └── masks/                    # Ground-truth binary manipulation masks
│   └── processed/
├── notebooks/
│   └── SEFA_Forensics_Walkthrough.ipynb  # Interactive step-by-step Jupyter Notebook
├── scripts/
│   ├── run_discovery.py              # CLI anchor discovery engine
│   └── run_train.py                  # CLI training & evaluation pipeline
├── src/
│   ├── models/
│   │   ├── backbone.py               # Frozen foundation backbone (DINOv2 / ViT)
│   │   ├── evidence_stress.py        # Controlled perturbation & latent residual generator
│   │   ├── anchors.py                # Forensic anchor bank & ultra-lightweight router
│   │   └── sefa_model.py             # Complete end-to-end SEFA architecture
│   ├── data/
│   │   ├── dataset.py                # Paired forensic dataset & synthetic inpainting generator
│   │   └── download_data.py          # Automated dataset downloader and curator
│   ├── engine/
│   │   ├── anchor_discovery.py       # IFSS statistical calibration engine
│   │   ├── trainer.py                # Fast router trainer with early stopping
│   │   └── evaluator.py              # AUROC, AP, F1, and localization IoU evaluator
│   ├── explainability/
│   │   ├── heatmaps.py               # Visual overlay and bounding-box extractor
│   │   └── justification.py          # Grounded natural-language diagnostic generator
│   └── utils/
│       └── metrics.py                # Standard forensic benchmark metrics
├── requirements.txt
└── README.md
```

---

## 🔬 Mathematical Formulation

### 1. Evidence Stress & Latent Residual
For an input image $x$, we apply controlled evidence stress $\mathcal{S}(x)$ designed to challenge generative diffusion coherence:
$$\mathcal{S}(x) = \text{Clamp}\left(x + \epsilon \cdot \mathcal{H}_{\text{spectral}}(x), 0, 1\right)$$
The frozen foundation model produces patch representations for clean and stressed inputs:
$$Z_l = \mathcal{F}_l(x), \quad Z_{l, \text{stress}} = \mathcal{F}_l(\mathcal{S}(x))$$
The discriminative latent residual is:
$$\Delta Z_l = |Z_l - Z_{l, \text{stress}}|$$

### 2. Intrinsic Forgery Sensitivity Score (IFSS)
On a calibration set of $M$ pairs, for each intermediate channel $c$:
$$\text{IFSS}_{l, c} = \frac{\left|\mathbb{E}[\Delta Z_{l, c}^{\text{forged}}] - \mathbb{E}[\Delta Z_{l, c}^{\text{real}}]\right|}{\sqrt{\text{Var}(\Delta Z_{l, c}^{\text{forged}}) + \text{Var}(\Delta Z_{l, c}^{\text{real}})} + \epsilon}$$
The top-$K$ channels with the highest IFSS form the **Intrinsic Forensic Anchors**.

### 3. Ultra-Lightweight Routing (<0.2% Parameters)
The extracted anchor activations $A_{\text{acts}}$ and residuals $A_{\text{res}}$ pass through a 2-layer MLP bottleneck ($K \to d_{\text{bottleneck}} \to 1$):
- **Patch Localization**: $\hat{M}(i, j) = \text{Sigmoid}(\text{Head}_{\text{patch}}(e_{i, j}))$
- **Image Forgery Logit**: Joint top-$k$ suspicious patch pooling and attention-weighted aggregation.

---

## ⚡ Quickstart

### 1. Installation
Using `uv` (recommended for ultra-fast setup):
```bash
# Create and activate virtual environment
uv venv .venv --python python3
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

### 2. Download & Curate Dataset
```bash
python3 src/data/download_data.py
```
This automatically curates real photographic baselines and creates subtle diffusion-inpainted pairs with ground-truth masks in `data/raw/`.

### 3. Run Anchor Discovery & Train Router
```bash
# 1. Discover forensic anchors
python3 scripts/run_discovery.py --data_dir data/raw --num_anchors 64

# 2. Train the lightweight router (takes < 3 minutes)
python3 scripts/run_train.py --epochs 10 --lr 0.001
```

### 4. Interactive Walkthrough in Jupyter Notebook
Open the interactive walkthrough notebook:
```bash
jupyter notebook notebooks/SEFA_Forensics_Walkthrough.ipynb
```

---

---

## 📄 Research Walkthrough & Publication PDF

The complete technical report, algorithmic mathematics, architecture diagrams, and high-resolution figures are compiled in:
- **[SEFA_Forensics_Research_Walkthrough.pdf](SEFA_Forensics_Research_Walkthrough.pdf)** (Complete 11.88 MB research walkthrough)
- **Interactive Notebook**: `notebooks/SEFA_Forensics_Walkthrough.ipynb`
- **Publication Figures**: `reports/figures/` (Figures 1 through 9)

---

## 📊 Grounded Forensic Case Diagnostic Evaluation

```text
Sample: test_sample_84
  Ground Truth:    Manipulated (Inpainted)
  Model Decision:  Manipulated (Probability: 0.9854)
  Localization:    center-central region (bounding box: [24, 44, 222, 223], covering 23.4% of image)
  Active Forensic Anchors:
    - Channel 58 (Latent Manifold Stress Sensitivity): activation = 0.074
      Evidence: Internal representation collapsed under evidence stress, confirming non-natural generative synthesis.
    - Channel 53 (Boundary Gradient Phase Incoherence): activation = 0.074
      Evidence: Spatial gradient transition seam detected along local patch perimeter, indicating inpainting mask stitching.
```

---

## 📈 Empirical Benchmark Performance

- **Classification Accuracy**: 95.71%
- **AUROC**: 0.9884 | **Average Precision (AP)**: 0.9882
- **Pixel Localization mIoU**: 84.15% | **Dice**: 90.82%
- **Social Media JPEG Q=75 Robustness**: 0.9384 AUROC (94.94% retention)
- **Train Loss Convergence**: 0.2559 (<0.30 threshold, NVIDIA RTX 3050 Ti GPU)

---

## 📄 Citation & License
MIT License. Free for academic and research exploration.

