# Uncertainty-Aware Autism Spectrum Disorder Screening Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active Research](https://img.shields.io/badge/Status-Active%20Research-success.svg)]()

An uncertainty-aware, soft-computing computational framework for automated Autism Spectrum Disorder (ASD) screening using Interval Type-2 Bipolar Fuzzy feature engineering and prime-layered MLPs $(61, 31)$ for leakage-free, explainable clinical classification.

---

## 📌 Overview & Key Innovations
Early clinical screening for Autism Spectrum Disorder (ASD) often struggles with human hesitation, response ambiguity, and subjective diagnostic uncertainty. This repository provides a complete, leakage-free computational pipeline that bridges soft-computing uncertainty modeling with deep neural architectures for robust, explainable multi-age ASD screening.

### Key Innovations:
1. **Interval Type-2 (IT2) & Bipolar Fuzzy Feature Engineering:** Explicitly models epistemic uncertainty boundaries ($\delta = 0.15$) and cognitive positive/negative polarities from questionnaire scores.
2. **Leakage-Free Preprocessing Pipeline:** Strict enforcement of train-test partitioning prior to imputation and scaling to guarantee unbiased evaluation.
3. **Prime-Layered Multilayer Perceptron (MLP):** Introduces optimized prime-node hidden topologies $(61, 31)$ to disrupt structural parameter redundancies and enhance representational diversity.
4. **Explainable AI (XAI):** Integrated SHAP (Shapley Additive exPlanations) analysis to quantify feature contribution hierarchies.

---

## 📂 Repository Structure
```text
asd-fuzzy-mlp/
├── data/                       # Consolidated multi-age ASD benchmark datasets
│   ├── raw/                    # Original multi-age questionnaire datasets
│   └── processed/              # Cleaned and partitioned data artifacts
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Leakage-free imputation, scaling, and data splits
│   ├── fuzzy_engine.py         # IT2 and Bipolar Fuzzy membership feature generation
│   ├── models.py               # Prime-layered MLP and standard architectures
│   └── evaluation.py           # 10-seed experiment runner, ablation, and SHAP analysis
├── notebooks/                  # Interactive exploratory data analysis and visualization
├── outputs/                    # Generated confusion matrices, learning curves, and SHAP plots
│   ├── figures/                # High-resolution PDF/PNG visual artifacts
│   └── logs/                   # Training convergence and multi-seed metrics
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # Project documentation
