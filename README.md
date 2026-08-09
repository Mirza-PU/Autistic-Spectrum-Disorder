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

**Seeds:** `42, 100, 23, 7, 99, 123, 456, 789, 321, 555`

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
# 🔬 Data Preprocessing

The ASD-Fuzzy-MLP framework follows a strict **leakage-free preprocessing strategy** to ensure that information from the test set does not influence model training or preprocessing parameter estimation.

### Preprocessing Pipeline

- Stratified **80:20 train/test split**.
- Median imputation for numerical features.
- Mode imputation for categorical features.
- `StandardScaler` fitted exclusively on the training set.
- Training-derived preprocessing parameters applied to the test set.
- Interval Type-2 fuzzy feature transformation.
- Bipolar fuzzy feature representation.
- Uncertainty-aware feature expansion.

### 🔒 Leakage Prevention

To ensure unbiased evaluation:

- The train/test split is performed **before** data-dependent preprocessing.
- Imputation statistics are calculated using the training data only.
- Scaling parameters are learned using the training data only.
- The test set is transformed using parameters learned from the training set.
- Test data are not used during model fitting.
- Test data are not used to estimate preprocessing parameters.

This protocol ensures that the reported test performance reflects evaluation on previously unseen data.

---
# 🧩 IT2 & Bipolar Fuzzy Feature Engineering

The ASD-Fuzzy-MLP framework employs a hybrid **Interval Type-2 (IT2) and Bipolar Fuzzy feature representation** to capture uncertainty and polarity within questionnaire-derived screening information.

### Interval Type-2 Fuzzy Transformation

The IT2 fuzzy component represents uncertainty through interval-valued membership information rather than relying on a single deterministic membership value.

The uncertainty band is controlled by:

**δ = 0.15**

This transformation allows the framework to incorporate uncertainty and ambiguity present in questionnaire responses into the machine-learning representation.

### Bipolar Fuzzy Representation

The Bipolar Fuzzy component captures two complementary aspects of the input information:

- Positive membership information.
- Negative membership information.

The resulting fuzzy representation provides an uncertainty-aware feature space for the subsequent MLP classification stage.

### Combined Representation

The IT2 and Bipolar Fuzzy transformations are integrated before classification, providing the MLP with enriched features designed to represent both **uncertainty** and **positive/negative information** within the ASD screening data.

---
# 🧠 Explainable Artificial Intelligence

The ASD-Fuzzy-MLP framework incorporates **SHAP (SHapley Additive exPlanations)** to provide interpretable insights into the model's classification decisions.

### SHAP Analysis

SHAP is used to analyze the contribution of individual features to ASD screening predictions.

The explainability analysis supports:

- Global feature importance.
- Local prediction explanations.
- Feature contribution analysis.
- Positive and negative feature effects.
- Identification of influential questionnaire variables.
- Interpretation of model predictions across different age groups.
- Analysis of the relative contribution of uncertainty-aware fuzzy features.

The combination of **IT2 Bipolar Fuzzy feature engineering** and **SHAP-based XAI** provides an interpretable framework for investigating how uncertainty-aware representations contribute to ASD classification.

---
# 📈 Experimental Results

The ASD-Fuzzy-MLP framework was evaluated using the predefined **10 random seeds** to assess the stability and reproducibility of the proposed approach.

The reported multi-seed performance is summarized below:

| Metric | Mean ± Standard Deviation |
|---|---:|
| **Mean Train Accuracy** | **1.0000 ± 0.0000** |
| **Mean Test Accuracy** | **99.91% ± 0.18%** |

The results demonstrate consistently high classification performance across the predefined experimental seeds under the specified leakage-free preprocessing and evaluation protocol.

> **Note:** The reported performance should be interpreted within the context of the underlying questionnaire benchmark and experimental setting. High predictive accuracy does not independently establish clinical diagnostic validity.

---
# 📊 Evaluation Metrics

The ASD-Fuzzy-MLP framework supports comprehensive evaluation of classification performance across multiple experimental seeds.

The evaluation includes:

- **Accuracy**
- **Mean Accuracy**
- **Standard Deviation**
- **Multi-Seed Performance**
- **Classification Performance**
- **Confusion Matrix**
- **SHAP Feature Importance**
- **SHAP Summary Analysis**
- **Local Prediction Explanations**
- **Age-Specific Performance Analysis**
- **Demographic Analysis**

These evaluation measures provide a comprehensive assessment of predictive performance, model stability, and feature-level interpretability.

---
# 🔁 Reproducibility

The ASD-Fuzzy-MLP implementation is designed to support reproducible experimental evaluation.

The following settings are fixed throughout the experiments:

- Stratified **80:20 train/test split**.
- Median/mode missing-value imputation.
- Training-only `StandardScaler` fitting.
- Interval Type-2 and Bipolar Fuzzy feature transformation.
- Uncertainty parameter **δ = 0.15**.
- Prime-layered MLP topology **(61, 31)**.
- ReLU activation.
- Adam optimizer.
- Maximum iterations **500**.
- Ten predefined random seeds.
- Independent test-set evaluation.
- SHAP-based explainability.

### 🔢 Experimental Seeds

The predefined seeds used for multi-seed evaluation are:

**42, 100, 23, 7, 99, 123, 456, 789, 321, 555**

Using multiple fixed seeds allows the stability of the proposed framework to be assessed rather than relying on the result of a single random train/test configuration.

---
# 📄 Citation

If you use **ASD-Fuzzy-MLP**, its methodology, implementation, or experimental results in your research, please cite the associated publication.

### BibTeX

```bibtex
@article{hussain2026asdfuzzymlp,
  author  = {Mirza Mudassar Hussain},
  title   = {Uncertainty-Aware Autism Spectrum Disorder Screening Framework},
  journal = {Journal of Soft Computing and Clinical AI Decision Support},
  year    = {2026}
}
---
# 👨‍💻 Author

**Mirza Mudassar Hussain**

PhD Mathematics Scholar  
Institute of Mathematics  
University of the Punjab  
Lahore, Pakistan

### Research Interests

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Soft Computing
- Fuzzy Systems
- Explainable AI
- Clinical AI
- Computational Mathematics
- Uncertainty-Aware Classification

---

# 📬 Contact

For research collaboration, technical questions, or methodological discussions:

**Mirza Mudassar Hussain**

📧 **Email:** `muddasser.mh@gmail.com`

🐙 **GitHub:** https://github.com/Mirza-PU

💼 **LinkedIn:** https://www.linkedin.com/in/mirza-mudassar-hussain

🆔 **ORCID:** https://orcid.org/0009-0002-1897-5458

---
# 🤝 Contributing

Contributions are welcome from researchers and developers working in:

- Machine Learning
- Fuzzy Systems
- Soft Computing
- Clinical AI
- Explainable AI
- Uncertainty Modeling
- Computational ASD Screening

To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Implement and test your changes.
4. Document the proposed modification.
5. Commit your changes.
6. Submit a Pull Request.

Please ensure that contributions maintain the project's reproducibility, documentation standards, and leakage-free experimental methodology.

---

# 🐞 Issues

If you encounter any problems while using **ASD-Fuzzy-MLP**, please report them through the GitHub issue tracker.

### Issue Tracker

https://github.com/Mirza-PU/asd-fuzzy-mlp/issues

When submitting an issue, please provide relevant information such as:

- Python version
- Operating system
- Package versions
- Configuration file
- Error message or traceback
- Relevant dataset or preprocessing information

This information helps facilitate efficient troubleshooting and reproducible resolution of reported issues.

---
# 📜 License

This project is licensed under the **MIT License**.

The MIT License permits use, modification, distribution, and reproduction of the software subject to the terms and conditions specified in the license.

See the [`LICENSE`](LICENSE) file for the complete license text.

---
# 🙏 Acknowledgements

The author gratefully acknowledges:

- The researchers and data providers responsible for the questionnaire-based ASD screening benchmarks used in this research.
- The Python development community.
- The Scikit-Learn development team.
- The PyTorch development team.
- The NumPy and Pandas development teams.
- The SHAP development community.
- The open-source scientific computing and machine-learning community.

The author also acknowledges the broader research community working on **fuzzy systems, uncertainty-aware machine learning, explainable AI, and computational ASD screening**.

---
# ⭐ Support the Project

If you find **ASD-Fuzzy-MLP** useful in your research, please consider:

- ⭐ Starring the repository.
- 🍴 Forking the repository.
- 📚 Citing the associated publication.
- 🤝 Sharing the work with the research community.

For research collaboration, technical questions, or methodological discussions, please contact:

**Mirza Mudassar Hussain**

📧 `muddasser.mh@gmail.com`

Your support helps increase the visibility of the project and encourages further research and development.

---
# ⚠️ Disclaimer

**ASD-Fuzzy-MLP** is a research and computational screening framework.

It is **not a medical diagnostic device** and should not be used as a substitute for professional clinical assessment, diagnosis, or medical advice.

Model predictions should not replace the judgment of qualified healthcare professionals.

The reported experimental results are specific to the datasets, preprocessing procedures, and evaluation protocols used in this research.

---

<p align="center">

<strong>ASD-Fuzzy-MLP</strong><br>

Uncertainty-Aware Autism Spectrum Disorder Screening Framework

<br><br>

Developed for uncertainty-aware, explainable, and soft-computing-based computational ASD screening research.

<br><br>

© 2026 <strong>Mirza Mudassar Hussain</strong> · MIT License

</p>

