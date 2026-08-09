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

---
---
# =====================================================================
# Complete Python Implementation Code Snippets (For Supplementary Material)
# =====================================================================

# --- src/preprocessing.py ---
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def leakage_free_preprocessing(df: pd.DataFrame, target_column: string, test_size: float = 0.2, random_state: int = 42):
    """
    Executes a strict leakage-free preprocessing pipeline:
    1. Stratified train-test splitting before any imputation or scaling.
    2. Median/Mode imputation fitted exclusively on training partition.
    3. Robust scaling fitted exclusively on training partition.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 1. Stratified Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Identify numerical and categorical features
    num_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = X_train.select_dtypes(include=['object', 'category']).columns
    
    # 2. Imputation fitted strictly on X_train
    num_imputer = SimpleImputer(strategy='median')
    cat_imputer = SimpleImputer(strategy='most_frequent')
    
    X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
    X_test[num_cols] = num_imputer.transform(X_test[num_cols])
    
    if len(cat_cols) > 0:
        X_train[cat_cols] = cat_imputer.fit_transform(X_train[cat_cols])
        X_test[cat_cols] = cat_imputer.transform(X_test[cat_cols])
        
    # 3. Standardization fitted strictly on X_train
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test


# --- src/fuzzy_engine.py ---
import numpy as np
import pandas as pd

def generate_fuzzy_features(df: pd.DataFrame, uncertainty_delta: float = 0.15) -> pd.DataFrame:
    """
    Generates Interval Type-2 (IT2) and Bipolar Fuzzy membership features 
    from questionnaire score columns.
    """
    score_cols = [col for col in df.columns if col.startswith('A') and col[1:].isdigit()]
    
    # Normalize score columns to [0, 1]
    normalized_scores = df[score_cols].div(df[score_cols].max(axis=0))
    mean_ratio = normalized_scores.mean(axis=1)
    
    # IT2 Fuzzy Envelope
    df['IT2_Fuzzy_Upper'] = np.minimum(1.0, mean_ratio + uncertainty_delta)
    df['IT2_Fuzzy_Lower'] = np.maximum(0.0, mean_ratio - uncertainty_delta)
    
    # Bipolar Fuzzy Polarities
    df['Bipolar_Positive'] = mean_ratio
    df['Bipolar_Negative'] = -(1.0 - mean_ratio)
    
    return df


# --- src/models.py ---
from sklearn.neural_network import MLPClassifier

class PrimeFuzzyMLP(MLPClassifier):
    """
    Multilayer Perceptron classifier utilizing prime-node hidden topologies
    to disrupt structural parameter redundancies and enhance representational diversity.
    """
    def __init__(self, hidden_layer_sizes=(61, 31), max_iter=500, random_state=42, **kwargs):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=random_state,
            **kwargs
        )


# --- src/evaluation.py ---
import numpy as np
import shap

def run_ten_seed_evaluation(X, y, model_class, seeds=[42, 100, 23, 7, 99, 123, 456, 789, 321, 555]):
    """
    Executes a 10-seed experiment runner to calculate mean train and test accuracies.
    """
    train_accs, test_accs = [], []
    
    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        model = model_class(random_state=seed)
        model.fit(X_train, y_train)
        
        train_accs.append(model.score(X_train, y_train))
        test_accs.append(model.score(X_test, y_test))
        
    print(f"Mean Train Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}")
    print(f"Mean Test Accuracy:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")


def compute_shap_values(model, X_train, X_test):
    """
    Initializes SHAP KernelExplainer to analyze feature contribution hierarchies.
    """
    explainer = shap.KernelExplainer(model.predict_proba, X_train.sample(100, random_state=42))
    shap_values = explainer.shap_values(X_test.sample(50, random_state=42))
    return explainer, shap_values
