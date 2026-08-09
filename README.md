# ASD-Fuzzy-MLP: Uncertainty-Aware Autism Spectrum Disorder Screening

<p align="center">

<h2 align="center">
ASD-Fuzzy-MLP: An Uncertainty-Aware Interval Type-2 Bipolar Fuzzy Framework for Autism Spectrum Disorder Screening
</h2>

<p align="center">
Official Implementation
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Compatible-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Research](https://img.shields.io/badge/Research-Clinical%20AI%20%7C%20Soft%20Computing-success.svg)
![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-green.svg)

</p>

---
## 📖 Overview

**ASD-Fuzzy-MLP** is an uncertainty-aware soft-computing framework designed for automated **Autism Spectrum Disorder (ASD) screening** using questionnaire-based clinical and demographic information.

The framework integrates **Interval Type-2 Fuzzy Logic**, **Bipolar Fuzzy feature engineering**, and **prime-layered Multilayer Perceptrons (MLPs)** to model uncertainty, ambiguity, and nonlinear relationships within ASD screening data.

The proposed framework emphasizes:

- Uncertainty-aware feature representation.
- Interval Type-2 fuzzy feature expansion.
- Bipolar fuzzy feature engineering.
- Prime-layered MLP classification.
- Strict leakage-free preprocessing.
- SHAP-based Explainable Artificial Intelligence (XAI).
- Multi-seed reproducibility.
- Multi-age demographic analysis.

The framework is intended for **research and computational screening applications** and should not be considered a replacement for professional clinical assessment or diagnosis.

---
## ✨ Key Contributions

The major contributions of the proposed **ASD-Fuzzy-MLP** framework include:

- **Interval Type-2 Fuzzy Feature Engineering**  
  Models uncertainty in questionnaire-derived features using interval-valued fuzzy representations.

- **Bipolar Fuzzy Feature Expansion**  
  Represents complementary positive and negative information within the screening features.

- **Uncertainty-Aware Transformation**  
  Incorporates an uncertainty band controlled by:

  \[
  \delta = 0.15
  \]

- **Prime-Layered MLP Architecture**  
  Employs two prime-valued hidden-layer dimensions:

  \[
  (61,31)
  \]

- **Strict Leakage-Free Preprocessing**  
  Ensures that imputation, scaling, and other data-dependent transformations are learned exclusively from the training set.

- **SHAP-Based Explainable AI**  
  Provides feature-level explanations to identify influential variables contributing to ASD screening predictions.

- **Multi-Seed Reproducibility**  
  Evaluates the framework using ten predefined random seeds to assess performance stability.

- **Multi-Age Demographic Granularity**  
  Supports questionnaire-based ASD screening across multiple developmental age groups, including infants, children, adolescents, and adults.

---
# 🏗 ASD-Fuzzy-MLP Framework

<p align="center">
<img src="pipeline_architecture.png" width="700">
</p>

The proposed framework integrates uncertainty-aware fuzzy feature engineering, leakage-free preprocessing, neural classification, multi-seed evaluation, and explainable AI.

---

# 🚀 Framework Features

✅ Interval Type-2 Fuzzy Feature Expansion

✅ Bipolar Fuzzy Feature Representation

✅ Uncertainty-Aware Feature Transformation

✅ Leakage-Free Data Preprocessing

✅ Median/Mode Missing-Value Imputation

✅ Training-Only StandardScaler

✅ Prime-Layered MLP Classification

✅ ReLU-Based Nonlinear Representation

✅ Multi-Seed Reproducibility

✅ SHAP-Based Explainable AI

✅ Multi-Age ASD Screening Analysis

✅ Demographic Granularity

✅ Publication-Quality Performance Evaluation

---
# 🧠 Network Architecture

The proposed **ASD-Fuzzy-MLP** framework employs a prime-layered Multilayer Perceptron (MLP) for ASD classification.

| Layer | Configuration |
|---------|--------------|
| Input | IT2 + Bipolar Fuzzy Feature Representation |
| Hidden Layer 1 | Linear (61) |
| Activation | ReLU |
| Hidden Layer 2 | Linear (31) |
| Activation | ReLU |
| Output Layer | ASD Classification |
| Architecture | Prime-Layered MLP |

The proposed hidden-layer topology is:

(61, 31)
---
### ⚙️ Hyperparameters

The principal experimental and optimization parameters used in the ASD-Fuzzy-MLP framework are summarized below.

| Component | Value |
|------------|-------|
| Train/Test Split | Stratified 80:20 |
| Missing-Value Imputation | Median / Mode |
| Feature Scaling | StandardScaler |
| Scaling Strategy | Training Set Only |
| Fuzzy Representation | Interval Type-2 + Bipolar Fuzzy |
| Uncertainty Band | `δ = 0.15` |
| Hidden Layer 1 | `61` |
| Hidden Layer 2 | `31` |
| Activation Function | ReLU |
| Optimizer | Adam |
| Maximum Iterations | `500` |
| Evaluation Strategy | Multi-Seed |
| Number of Seeds | `10` |

## 🔁 Random Seeds

The experiments use the following predefined random seeds:

```python
SEEDS = [42, 100, 23, 7, 99, 123, 456, 789, 321, 555]

---

---
# 📊 Dataset

The **ASD-Fuzzy-MLP** framework is designed for **multi-age questionnaire-based ASD screening benchmarks**, incorporating clinical questionnaire responses and demographic information across different developmental stages.

### Developmental Age Groups

The benchmark covers multiple developmental stages:

- 👶 **Infants**
- 🧒 **Children**
- 🧑 **Adolescents**
- 👨 **Adults**

The experimental framework incorporates four age categories:

| Age Category | Developmental Group |
|---|---|
| Age 0 | Infant |
| Age 1 | Child |
| Age 2 | Adolescent |
| Age 3 | Adult |

The multi-age structure enables evaluation of the proposed uncertainty-aware framework across different developmental stages while maintaining demographic granularity.

### Dataset Characteristics

- Questionnaire-based ASD screening information.
- Clinical and demographic features.
- Multiple developmental age groups.
- Binary ASD screening classification.
- Structured tabular machine-learning format.
- Suitable for uncertainty-aware feature engineering and explainable classification.

> **Note:** The original benchmark data are not redistributed in this repository unless their respective licenses and data-sharing conditions permit redistribution.

---
