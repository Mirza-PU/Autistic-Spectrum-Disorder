# Uncertainty-Aware Autism Spectrum Disorder Screening Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Uncertainty-aware ASD screening framework using Interval Type-2 Bipolar Fuzzy feature engineering and prime-layered MLPs $(61, 31)$ for leakage-free, explainable clinical classification.

---



## 📌 Overview
Early clinical screening for Autism Spectrum Disorder (ASD) often struggles with human hesitation, response ambiguity, and subjective diagnostic uncertainty. This repository provides a complete, leakage-free computational pipeline that bridges soft-computing uncertainty modeling with deep neural architectures for robust, explainable multi-age ASD screening.

### Key Innovations:
1. **Interval Type-2 (IT2) & Bipolar Fuzzy Feature Engineering:** Explicitly models epistemic uncertainty boundaries ($\delta = 0.15$) and cognitive positive/negative polarities from questionnaire scores.
2. **Leakage-Free Preprocessing Pipeline:** Strict enforcement of train-test partitioning prior to imputation and scaling to guarantee unbiased evaluation.
3. **Prime-Layered Multilayer Perceptron (MLP):** Introduces optimized prime-node hidden topologies $(61, 31)$ to disrupt structural parameter redundancies and enhance representational diversity.
4. **Explainable AI (XAI):** Integrated SHAP (Shapley Additive exPlanations) analysis to quantify feature contribution hierarchies.

---

## 📂 Repository Structure
```text
├── data/                       # Consolidated multi-age ASD benchmark datasets
├── src/
│   ├── preprocessing.py        # Leakage-free imputation, scaling, and data splits
│   ├── fuzzy_engine.py         # IT2 and Bipolar Fuzzy membership feature generation
│   ├── models.py               # Prime-layered MLP and standard architectures
│   └── evaluation.py           # 10-seed experiment runner, ablation, and SHAP analysis
├── notebooks/                  # Interactive exploratory data analysis and visualization
├── outputs/                    # Generated confusion matrices, learning curves, and SHAP plots
├── requirements.txt            # Python dependencies
└── README.md
---
## 🚀 Installation & Setup

Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/asd-fuzzy-mlp.git](https://github.com/your-username/asd-fuzzy-mlp.git)
cd asd-fuzzy-mlp
pip install -r requirements.txt
