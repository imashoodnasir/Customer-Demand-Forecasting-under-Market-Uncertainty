# Bayesian Hierarchical Modeling Framework for Customer Demand Forecasting under Market Uncertainty

<p align="center">
<img src="https://img.shields.io/badge/Python-3.10%2B-blue">
<img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red">
<img src="https://img.shields.io/badge/PyMC-Bayesian%20Inference-orange">
<img src="https://img.shields.io/badge/Forecasting-Retail%20Demand-green">
<img src="https://img.shields.io/badge/License-MIT-yellow">
</p>

## Overview

This repository contains the implementation of a Bayesian hierarchical forecasting framework for large-scale retail demand prediction under market uncertainty.

The proposed framework integrates:

- hierarchical Bayesian demand modeling
- negative binomial likelihood for count-based demand
- partial pooling across retail hierarchies
- temporal dependency modeling
- structured market covariates
- nonlinear demand effects
- posterior predictive inference
- uncertainty quantification
- cross-dataset Bayesian adaptation

The framework is designed to simultaneously improve:

- forecasting accuracy
- probabilistic calibration
- hierarchical consistency
- cross-domain generalization
- decision reliability under uncertainty

The methodology is evaluated on two large-scale retail forecasting benchmarks:

- **M5 Forecasting Competition Dataset**
- **Corporación Favorita Grocery Sales Dataset**

The framework is compared against statistical, deep learning, Transformer-based, and foundation forecasting approaches.

---

# Research Contribution

The proposed framework addresses several limitations of existing retail forecasting approaches:

| Challenge | Proposed Solution |
|---|---|
| Independent forecasting of retail series | Bayesian hierarchical parameter sharing |
| Sparse demand series | Partial pooling across related entities |
| Demand uncertainty | Posterior predictive distributions |
| Hierarchical inconsistency | Joint hierarchical modeling |
| Market fluctuations | Structured covariate modeling |
| Dataset shift | Bayesian transfer adaptation |
| Limited uncertainty estimation | Calibrated prediction intervals |

---

# Methodology Overview

The proposed Bayesian hierarchical framework consists of five major components:

```
Retail Demand Data
        |
        v
Feature Engineering
        |
        v
Hierarchical Bayesian Model
        |
        +----------------+
        |                |
        v                v
Posterior Inference   Predictive Distribution
        |                |
        +----------------+
                 |
                 v
Forecast Evaluation
```

---

# Model Components

## 1. Bayesian Demand Model

Demand observations are modeled using a negative binomial likelihood:

\[
y_{i,t} \sim NegBin(\mu_{i,t},\phi)
\]

where:

- \(y_{i,t}\) represents observed demand
- \(\mu_{i,t}\) represents expected demand
- \(\phi\) represents dispersion

---

## 2. Hierarchical Parameterization

The model captures dependencies across multiple retail levels:

```
Overall Retail Demand

        |
     Region

        |
      Store

        |
   Department

        |
    Category

        |
      Item
```

Partial pooling enables information sharing between:

- products
- categories
- departments
- stores
- regions

---

## 3. Temporal Modeling

The framework incorporates:

- historical demand patterns
- seasonality
- trends
- autoregressive effects
- long-term dependencies

---

## 4. Market Covariates

External retail factors are incorporated:

- price variations
- promotions
- holidays
- transactions
- market conditions

---

## 5. Bayesian Posterior Inference

Posterior inference generates:

- point forecasts
- predictive distributions
- credible intervals
- uncertainty estimates

---

# Datasets

## M5 Forecasting Competition

Dataset characteristics:

- hierarchical Walmart retail demand
- item-store level forecasting
- multiple aggregation levels
- calendar information
- selling prices
- promotional effects

---

## Corporación Favorita Grocery Sales

Dataset characteristics:

- grocery sales forecasting
- product-store relationships
- promotions
- holidays
- external economic information

---

# Experimental Comparison

The framework is compared with:

## Statistical Models

- Prophet

## Deep Learning Models

- DeepAR
- N-BEATS

## Transformer Models

- Temporal Fusion Transformer (TFT)
- PatchTST
- Autoformer
- FEDformer
- TimesNet
- iTransformer

## Hierarchical / Foundation Models

- HINT
- GBPF
- Time-MoE

---

# Main Results

## M5 Forecasting Dataset

| Metric | Proposed Framework |
|---|---:|
| RMSE | 1.931 |
| MAE | 1.219 |
| WRMSSE | 0.571 |
| sMAPE | 11.01 |
| CRPS | 0.338 |

---

## Corporación Favorita Dataset

| Metric | Proposed Framework |
|---|---:|
| RMSE | 2.016 |
| MAE | 1.266 |
| WRMSSE | 0.588 |
| sMAPE | 11.42 |
| CRPS | 0.352 |

---

# Repository Structure

```
BayesianRetailForecasting/

│
├── configs/
│   ├── data/
│   ├── models/
│   ├── experiments/
│   └── baselines/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   └── bayesian_retail/
│       │
│       ├── data/
│       ├── preprocessing/
│       ├── models/
│       │
│       ├── baselines/
│       │   ├── prophet/
│       │   ├── deepar/
│       │   ├── nbeats/
│       │   ├── tft/
│       │   ├── patchtst/
│       │   ├── autoformer/
│       │   ├── fedformer/
│       │   ├── timesnet/
│       │   ├── itransformer/
│       │   ├── hint/
│       │   ├── gbpf/
│       │   └── timemoe/
│       │
│       ├── evaluation/
│       ├── statistics/
│       ├── visualization/
│       └── experiments/
│
├── scripts/
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── generate_tables.py
│   └── generate_figures.py
│
├── results/
│   ├── tables/
│   ├── figures/
│   └── reports/
│
├── requirements.txt
├── environment.yml
├── Dockerfile
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/BayesianRetailForecasting.git

cd BayesianRetailForecasting
```

---

## Create Environment

### Using Conda

```bash
conda env create -f environment.yml

conda activate bayesian-retail
```

---

### Using pip

```bash
pip install -r requirements.txt
```

---

# Dataset Preparation

Download the datasets:

- M5 Forecasting Competition Dataset
- Corporación Favorita Grocery Sales Dataset

Place files:

```
data/raw/

├── m5/
│
└── favorita/
```

Run preprocessing:

```bash
python scripts/prepare_data.py
```

Processed data will be generated:

```
data/processed/
```

---

# Training

## Train Bayesian Framework

```bash
python scripts/train_bayesian.py
```

---

## Train All Baselines

```bash
python scripts/train_all_models.py
```

Available models:

```text
prophet
deepar
nbeats
tft
patchtst
autoformer
fedformer
timesnet
itransformer
hint
gbpf
timemoe
bayesian
```

---

# Evaluation

Run complete evaluation:

```bash
python scripts/evaluate.py
```

Generated metrics:

- RMSE
- MAE
- WRMSSE
- sMAPE
- CRPS
- PICP
- MPIW
- NLPD

---

# Statistical Analysis

The framework supports:

- Friedman test
- Nemenyi post-hoc analysis
- Wilcoxon signed-rank test
- Cohen's d effect size

Run:

```bash
python scripts/statistical_analysis.py
```

---

# Generate Paper Tables

```bash
python scripts/generate_tables.py
```

Outputs:

```
results/tables/

table_overall_performance.tex
table_cross_dataset.tex
table_hierarchy.tex
table_uncertainty.tex
table_ablation.tex
table_statistics.tex
```

---

# Generate Figures

```bash
python scripts/generate_figures.py
```

Outputs:

```
results/figures/

performance_comparison.png
uncertainty_calibration.png
hierarchy_analysis.png
ablation_results.png
critical_difference.png
```

---

# Reproducibility

Complete reproduction:

```bash
python scripts/reproduce_paper.py --mode full
```

The pipeline performs:

```
Dataset Preparation
        |
        v
Feature Engineering
        |
        v
Model Training
        |
        v
Evaluation
        |
        v
Statistical Testing
        |
        v
Tables and Figures
```

---

# Hardware Requirements

Recommended:

| Component | Requirement |
|-|-|
| CPU | 8+ cores |
| RAM | 32 GB+ |
| GPU | NVIDIA GPU |
| CUDA | 12.x |
| Storage | 50 GB+ |

---

# License

This project is released under the MIT License.

---

# Contact

For questions, suggestions, or collaboration:

**Author:** Your Name  
**Email:** your.email@example.com
