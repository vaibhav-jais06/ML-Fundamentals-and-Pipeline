<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Pipeline-8A2BE2?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="ML Pipeline Badge" />
  
  # 🚢 ML Fundamentals & Pipeline
  
  **A Comprehensive Guide to Data Manipulation, Supervised & Unsupervised Learning, Regularization & Shrinkage, Ensemble Boosting, and Model Evaluation Diagnostics**

  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" /></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" /></a>
    <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Array_Computing-013243.svg?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" /></a>
    <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg?style=flat-square" alt="Matplotlib" /></a>
    <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" /></a>
  </p>

  *From foundational mathematical array operations to building end-to-end classification pipelines, penalizing complexity with Ridge/Lasso/ElasticNet, state-of-the-art boosting ensemble models, and rigorous cross-validation diagnostics.*

  [**Core Notebooks**](#-core-notebooks) • [**Regularization & Shrinkage**](#-regularization--shrinkage-spotlight) • [**Model Evaluation**](#-model-evaluation--diagnostics-spotlight) • [**Ensemble & Boosting**](#-ensemble-learning--boosting-spotlight) • [**Pipeline Lifecycle**](#-pipeline-architecture) • [**Quickstart**](#-getting-started) • [**Results**](#-results--evaluation)

</div>

---

## 🌟 About The Project

Predicting outcomes and extracting meaningful patterns from data is at the core of Applied Machine Learning. This repository delivers a structured, progressive pathway through the ML lifecycle across six distinct pillars:

1. **Foundations of Data Engineering:** Vectorized operations, multidimensional indexing, DataFrame wrangling, and missing data imputation using **NumPy** and **Pandas**.
2. **End-to-End Supervised Pipeline:** Building a production-style classification workflow on the iconic **Titanic Survival Prediction** dataset.
3. **Supervised & Unsupervised Modeling:** Comparing parametric models (**Logistic Regression**) with non-parametric trees (**Decision Trees**), alongside unsupervised clustering (**K-Means**).
4. **Regularization & Shrinkage Methods:** Mitigating overfitting, taming multicollinearity, and driving automated feature selection via **L2 (Ridge)**, **L1 (Lasso)**, and hybrid **ElasticNet** penalties.
5. **Ensemble Methods & Boosting:** Deep-dive into sequential learning algorithms, comparing **Decision Stumps (Weak Learners)**, **AdaBoost**, and **Gradient Boosting**, complete with decision boundary plots and stage-wise error curves.
6. **Model Evaluation & Validation Diagnostics:** Rigorous evaluation methodologies using the **Wisconsin Breast Cancer** diagnostic dataset, contrasting naive baselines (`DummyClassifier`) against standard pipelines (`StandardScaler` + `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`), confusion matrices, classification reports, and 5-Fold `StratifiedKFold` cross-validation.

```
                    ┌───────────────────────────────┐
                    │  Data Science Foundations     │
                    │  (NumPy & Pandas Manipulation)│
                    └───────────────┬───────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │   End-to-End Classification   │
                    │   (EDA, Imputation, Baseline) │
                    └───────────────┬───────────────┘
                                    │
    ┌───────────────────────────────┼───────────────────────────────┐
    ▼                               ▼                               ▼
┌───────────────────────┐   ┌───────────────────────┐   ┌───────────────────────┐
│ Supervised / Unsuper. │   │ Regularization & L1/L2│   │ Ensemble Boosting     │
│ (LogReg, Trees,       │   │ (Ridge, Lasso Sparsity│   │ (AdaBoost, GradBoost, │
│  K-Means Clustering)  │   │  ElasticNet Hybrid)   │   │  Staged Loss Curves)  │
└───────────────────────┘   └───────────┬───────────┘   └───────────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────────┐
                    │     Model Evaluation & Diagnostics    │
                    │  (Dummy Baseline, PR/ROC, K-Fold CV)  │
                    └───────────────────────────────────────┘
```

> **Formal Problem Formulations:**  
> • **Titanic Survival Task:** Classify passenger survival binary state: `1` (Survived) or `0` (Did not survive) from passenger demographics, ticketing metadata, and cabin records ([Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)).  
> • **Controlled Regression & Sparsity Task:** Predict a continuous target from 10 features (5 informative signals, 5 noise features) under Gaussian perturbation to benchmark coefficient shrinkage and automated feature elimination.  
> • **Wisconsin Diagnostic Task:** Classify cellular biopsy nuclei as `1` (Benign) or `0` (Malignant) across 30 quantitative morphological features ([Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)).

---

## 📚 Core Notebooks

Each notebook in this repository is self-contained, heavily commented, and designed for hands-on experimentation.

| Notebook | Focus Area | Key Concepts & Algorithms |
| :--- | :--- | :--- |
| 📓 **[`Numpy_Pandas.ipynb`](./Numpy_Pandas.ipynb)** | **Data Foundations** | • Multi-dimensional arrays & broadcasting<br>• DataFrame filtering, grouping & aggregation<br>• Handling null values & data type transformations |
| 🚢 **[`ML_Pipeline.ipynb`](./ML_Pipeline.ipynb)** | **End-to-End Pipeline** | • Exploratory Data Analysis (EDA) with Seaborn<br>• Feature engineering & categorical encoding<br>• `DecisionTreeClassifier` training & evaluation<br>• Confusion Matrix & Classification Report |
| 🤖 **[`Week_3_ML_I_Sunday_.ipynb`](./Week_3_ML_I_Sunday_.ipynb)** | **Supervised & Unsupervised Models** | • `LogisticRegression` hyperparameter tuning<br>• `DecisionTreeClassifier` depth pruning<br>• `KMeans` clustering on synthetic 2D datasets<br>• Visualizing decision boundaries and cluster centroids |
| 🎯 **[`Regularization.ipynb`](./Regularization.ipynb)** | **Regularization & Feature Selection** | • **OLS Baseline:** Unregularized regression benchmarking<br>• **Ridge (L2):** Smooth weight shrinkage & collinearity control<br>• **Lasso (L1):** Sparse modeling & automated feature elimination<br>• **ElasticNet (L1 + L2):** Convex combination balancing sparsity & grouping<br>• Hyperparameter sensitivity sweeps ($\alpha$ & `l1_ratio`)<br>• Coefficient comparative profiling & bar chart visualizations |
| ⚡ **[`Boosting.ipynb`](./Boosting.ipynb)** | **Ensemble Learning & Boosting** | • **Weak Learners:** Training single-split Decision Stumps<br>• **AdaBoost:** Iterative sample re-weighting & decision boundary<br>• **Gradient Boosting:** Residual fitting & learning rate tuning<br>• Stage-wise accuracy curves (`staged_score`)<br>• Feature importance analysis across models |
| 🧪 **[`Model_Evalution.ipynb`](./Model_Evalution.ipynb)** | **Model Evaluation & Cross-Validation** | • Baseline benchmarking (`DummyClassifier`)<br>• Full metric suite: Precision, Recall, F1, Accuracy<br>• Scaled `Pipeline` with `StandardScaler` & `LogisticRegression`<br>• Confusion Matrix visualization (`ConfusionMatrixDisplay`)<br>• Multi-model comparison (LogReg, Decision Tree, Random Forest)<br>• 5-Fold `StratifiedKFold` Cross-Validation |

---

## 🎯 Regularization & Shrinkage Spotlight

The **[`Regularization.ipynb`](./Regularization.ipynb)** notebook provides an in-depth, hands-on masterclass on preventing overfitting, penalizing complexity, handling multicollinearity, and performing automated feature selection using **L2 Ridge**, **L1 Lasso**, and hybrid **ElasticNet** models.

```
[Ordinary Least Squares] ──► [Ridge (L2 Penalty)] ──► [Lasso (L1 Sparsity)] ──► [ElasticNet (Hybrid)]
   (OLS MSE: ~325.12)           (Smooth Shrinkage)       (7 Features Zeroed out)     (Combined L1 + L2)
```

<details>
<summary><b>🔍 Expand to View the 13-Part Curriculum in Regularization.ipynb</b></summary>
<br>

1. **Part 1: Import Regression & Regularization Libraries** — NumPy, Pandas, Matplotlib, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, and evaluation metrics (`mean_squared_error`, `r2_score`).
2. **Part 2: Controlled Synthetic Dataset Generation** — Creating a reproducible regression dataset via `make_regression(n_samples=200, n_features=10, n_informative=5, noise=20, random_state=42)`.
3. **Part 3: Semantic Feature Assignment** — Mapping realistic domain attributes: `Area`, `Bedrooms`, `Age`, `Location`, `Parking`, `Garden`, `Floor`, `Distance`, `School`, `Transport`.
4. **Part 4: Train / Test Partitioning** — Splitting into 160 training and 40 testing samples (80/20 partition, `random_state=42`).
5. **Part 5: Ordinary Least Squares (OLS) Baseline** — Fitting standard `LinearRegression` with unconstrained parameter optimization.
6. **Part 6: OLS Evaluation & Metric Logging** — Measuring baseline performance (MSE: 325.12, $R^2$: 0.949) and extracting unpenalized weight coefficients.
7. **Part 7: Ridge Regression (L2 Regularization)** — Training `Ridge(alpha=50)` with quadratic weight penalty to shrink coefficients smoothly.
8. **Part 8: Lasso Regression (L1 Regularization)** — Fitting `Lasso(alpha=10)` with absolute-value penalty to drive uninformative weights to exact zero.
9. **Part 9: Sparsity & Zero-Coefficient Detection** — Quantifying feature elimination: verifying that 7 out of 10 coefficients are driven to exact `0.000`.
10. **Part 10: ElasticNet Regularization (L1 + L2 Hybrid)** — Balancing L1 sparsity and L2 grouping with `ElasticNet(alpha=10, l1_ratio=0.5)`.
11. **Part 11: Multi-Model Benchmark & Tabular Comparison** — Consolidating MSE, $R^2$, and coefficient profiles into structured comparative DataFrames.
12. **Part 12: Visualizing Weight Profiles** — High-resolution comparative bar chart comparing feature coefficients across all four regression paradigms.
13. **Part 13: Hyperparameter Sweeps & Sensitivity** — Analyzing the impact of $\alpha$ on Lasso sparsity and `l1_ratio` on ElasticNet feature retention.

</details>

### Mathematical Formulations

Regularization injects a penalty term $\Omega(w)$ into the Ordinary Least Squares objective to control weight magnitude:

$$\mathcal{L}_{\text{OLS}}(w) = \frac{1}{2n}\sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2$$

* **Ridge Regression ($L_2$ Penalty):** Minimizes residual sum of squares plus the squared Euclidean norm of coefficients:
  $$\mathcal{L}_{\text{Ridge}}(w) = \mathcal{L}_{\text{OLS}}(w) + \alpha \sum_{j=1}^p w_j^2$$
* **Lasso Regression ($L_1$ Penalty):** Minimizes residual sum of squares plus the Manhattan ($L_1$) norm, inducing sharp geometric corners that force non-informative coefficients to exactly zero:
  $$\mathcal{L}_{\text{Lasso}}(w) = \mathcal{L}_{\text{OLS}}(w) + \alpha \sum_{j=1}^p |w_j|$$
* **ElasticNet ($L_1 + L_2$ Hybrid):** Convex combination parameterized by mixing ratio $\rho \in [0, 1]$ (`l1_ratio`):
  $$\mathcal{L}_{\text{ElasticNet}}(w) = \mathcal{L}_{\text{OLS}}(w) + \alpha \left[ \rho \sum_{j=1}^p |w_j| + \frac{1-\rho}{2} \sum_{j=1}^p w_j^2 \right]$$

### Quantitative Performance Benchmark

#### 🎯 Test Set Metrics (40 Samples / 20% Hold-out):
| Model | Penalty Type | Hyperparameters | Test MSE | Test $R^2$ Score | Zero Coefficients | Feature Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 📉 **Linear Regression** | None (OLS) | Unregularized | **325.12** | **0.949** | 0 / 10 (0%) | Retains all features; susceptible to noise |
| 🛡️ **Ridge Regression** | $L_2$ Penalty | $\alpha = 50.0$ | 866.39 | 0.863 | 0 / 10 (0%) | Smooth shrinkage; shrinks weights towards 0 without setting to 0 |
| ✂️ **Lasso Regression** | $L_1$ Penalty | $\alpha = 10.0$ | 761.83 | 0.880 | **7 / 10 (70%)** | **Automated Selection:** Retains only dominant signals (`Location`, `Distance`, `Garden`) |
| ⚖️ **ElasticNet** | $L_1 + L_2$ Hybrid | $\alpha = 10.0, \rho = 0.5$ | 4973.97 | 0.215 | 3 / 10 (30%) | Balances feature grouping with selective sparsity |

#### 📊 Feature Coefficient Comparison:
| Feature | Linear Regression | Ridge ($\alpha=50$) | Lasso ($\alpha=10$) | ElasticNet ($\alpha=10, \rho=0.5$) | Feature Status in Lasso |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Location`** | 79.956 | 59.982 | **70.505** | 12.212 | 🌟 **Retained** (Dominant signal) |
| **`Distance`** | 40.753 | 30.495 | **29.188** | 5.646 | 🌟 **Retained** (True informative) |
| **`Garden`** | 36.273 | 29.524 | **28.814** | 6.834 | 🌟 **Retained** (True informative) |
| **`Floor`** | 8.599 | 7.579 | **0.000** | 1.651 | ✂️ **Pruned to Zero** |
| **`Age`** | 7.122 | 3.333 | **0.000** | 0.000 | ✂️ **Pruned to Zero** |
| **`Area`** | 1.518 | 2.568 | **0.000** | 0.372 | ✂️ **Pruned to Zero** |
| **`School`** | 1.208 | 1.263 | **0.000** | 0.000 | ✂️ **Pruned to Zero** |
| **`Parking`** | 1.093 | -0.190 | **0.000** | 0.000 | ✂️ **Pruned to Zero** |
| **`Transport`** | -1.104 | -1.868 | **0.000** | -0.422 | ✂️ **Pruned to Zero** |
| **`Bedrooms`** | -1.805 | -3.003 | **0.000** | -0.771 | ✂️ **Pruned to Zero** |

#### 🎛️ Hyperparameter Sensitivity Sweeps:

* **Effect of Alpha ($\alpha$) on Lasso Sparsity:**
  | Regularization Strength ($\alpha$) | Zero Coefficients | Active Features | Sparsity Ratio | Modeling Impact |
  | :---: | :---: | :---: | :---: | :--- |
  | `0.01` | 0 / 10 | 10 / 10 | 0% | Near-OLS behavior; negligible penalty |
  | `0.10` | 0 / 10 | 10 / 10 | 0% | Mild shrinkage across coefficients |
  | `1.00` | 1 / 10 | 9 / 10 | 10% | Begins pruning weakest collinear feature |
  | `10.00` | **7 / 10** | **3 / 10** | **70%** | **Optimal selective sparsity:** Retains only genuine informative features |
  | `100.00` | 10 / 10 | 0 / 10 | 100% | Severe over-regularization; predicts global mean |

* **Effect of L1 Ratio ($\rho$) on ElasticNet Sparsity ($\alpha=10$):**
  | L1 Ratio ($\rho$) | Zero Coefficients | Active Features | Sparsity Character |
  | :---: | :---: | :---: | :--- |
  | `0.10` | 0 / 10 | 10 / 10 | Dominantly Ridge-like (all weights non-zero) |
  | `0.30` | 2 / 10 | 8 / 10 | Light sparsity with strong weight shrinkage |
  | `0.50` | 3 / 10 | 7 / 10 | Balanced combination of L1 and L2 penalties |
  | `0.70` | 5 / 10 | 5 / 10 | Moderate sparsity favoring L1 feature selection |
  | `0.90` | 6 / 10 | 4 / 10 | Strongly Lasso-like with sparse selection |

---

## 🧪 Model Evaluation & Diagnostics Spotlight

The **[`Model_Evalution.ipynb`](./Model_Evalution.ipynb)** notebook provides an in-depth, 15-part masterclass on assessing machine learning models reliably, detecting overfitting, and avoiding data leakage.

```
[Dummy Baseline] ──► [Scaled Pipeline] ──► [Multi-Model Benchmark] ──► [5-Fold Stratified CV]
  (Acc: ~62.74%)       (Acc: ~98.25%)      (LogReg vs Tree vs Forest)    (Stability Analysis)
```

<details>
<summary><b>🔍 Expand to View the 15-Part Curriculum in Model_Evalution.ipynb</b></summary>
<br>

1. **Part 1: Import Libraries** — Core scientific libraries, scikit-learn metrics, model selection tools, and pipelines.
2. **Part 2: Load Built-in Dataset** — Loading the Wisconsin Breast Cancer diagnostic dataset (569 samples, 30 features).
3. **Part 3: Understand the Data** — Inspecting feature dimensions, column types, and sample attributes.
4. **Part 4: Understand the Target Variable** — Mapping binary targets: `0` (Malignant) vs. `1` (Benign).
5. **Part 5: Check Class Balance** — Analyzing target distribution (62.74% benign, 37.26% malignant).
6. **Part 6: Train / Test Split** — Partitioning into training (80%) and test (20%) sets with `random_state=42`.
7. **Part 7: Baseline Model Benchmarking** — Establishing minimum performance threshold with `DummyClassifier(strategy="most_frequent")` (~62.74%).
8. **Part 8: Scaled Pipeline Construction** — Assembling a leak-free pipeline with `StandardScaler` and `LogisticRegression(max_iter=3000)`.
9. **Part 9: Four Core Metrics** — Calculating Accuracy, Precision, Recall, and F1-Score.
10. **Part 10: Detailed Classification Report** — Per-class precision, recall, f1-score, and support metrics.
11. **Part 11: Confusion Matrix Visualization** — Plotting True Positives, True Negatives, False Positives, and False Negatives using `ConfusionMatrixDisplay`.
12. **Part 12: Multi-Model Comparison** — Evaluating Logistic Regression, Decision Tree (`max_depth=4`), and Random Forest (`n_estimators=100`).
13. **Part 13: Comparative Metric Visualization** — Visualizing comparative bar charts across models and metrics.
14. **Part 14: Introduction to Cross-Validation** — 5-Fold Stratified Cross-Validation on training folds to measure variance and stability.
15. **Part 15: Cross-Validation Benchmark** — Side-by-side mean CV accuracy and standard deviation across all models.

</details>

### Diagnostic Performance Benchmark

#### 🎯 Hold-Out Test Set Performance (20% Split):
| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| 🥇 **Logistic Regression (Pipeline)** | **98.25%** | **98.61%** | 98.61% | **98.61%** |
| 🥈 **Random Forest (100 Trees)** | 95.61% | 95.89% | **97.22%** | 96.55% |
| 🥉 **Decision Tree (Depth 4)** | 93.86% | 95.77% | 94.44% | 95.10% |
| ⚪ *Dummy Baseline (Most Frequent)* | *62.74%* | *62.74%* | *100.0%* | *77.10%* |

#### 🔄 5-Fold Stratified Cross-Validation (Training Folds):
| Model | Mean CV Accuracy | CV Std Dev ($\sigma$) | Generalization & Stability |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (Pipeline)** | **97.80%** | **± 0.0098** | 🟢 Extremely stable and robust across folds |
| **Random Forest Classifier** | 96.26% | ± 0.0179 | 🟢 Strong ensemble variance reduction |
| **Decision Tree Classifier** | 92.97% | ± 0.0149 | 🟡 Higher sensitivity to data partitioning |

---

## ⚡ Ensemble Learning & Boosting Spotlight

The **[`Boosting.ipynb`](./Boosting.ipynb)** notebook provides an in-depth, 12-part masterclass on boosting mechanics:

```
[Weak Learner (Stump)] ──► [AdaBoost Classifier] ──► [Gradient Boosting] ──► [Comparative Evaluation]
     (Acc: ~82.4%)               (Acc: ~88-92%)              (Acc: ~90-94%)           (Feature Importance)
```

<details>
<summary><b>🔍 Expand to View the 12-Part Curriculum in Boosting.ipynb</b></summary>
<br>

1. **Part 1: Synthetic Dataset Generation** — Creating reproducible classification datasets using `make_classification`.
2. **Part 2: 2D Feature Visualization** — Plotting class separation and feature overlap with Matplotlib.
3. **Part 3: Weak Learner Baseline** — Training a 1-depth Decision Stump (`max_depth=1`) establishing a baseline (~82.4% accuracy).
4. **Part 4: AdaBoost Classifier** — Sequential ensembling with `AdaBoostClassifier` using multiple weak learners.
5. **Part 5: Decision Boundary Visualization** — High-resolution 2D mesh grid plots illustrating non-linear decision boundaries.
6. **Part 6: Iterative Learning Curves** — Tracking test set accuracy progression after each boosting iteration (`staged_score`).
7. **Part 7: Learner Weights & Errors** — Analyzing individual estimator weights ($\alpha_m$) and sample error rates.
8. **Part 8: Learning Rate Sensitivity** — Benchmarking learning rates (`0.1`, `0.5`, `1.0`, `1.5`) against overfitting/underfitting.
9. **Part 9: Gradient Boosting Classifier** — Fitting pseudo-residuals with `GradientBoostingClassifier`.
10. **Part 10: Model Comparison Benchmark** — Direct side-by-side performance comparison: Weak Learner vs. AdaBoost vs. Gradient Boosting.
11. **Part 11: Feature Importance Extraction** — Comparing Gini importance scores across tree ensembles.
12. **Part 12: Student Hands-on Experiment** — Interactive tuning playground for hyperparameters (n_estimators, learning_rate, max_depth).

</details>

---

## 🛠️ Pipeline Architecture

Our machine learning pipeline follows an industry-standard modular architecture:

<details>
<summary><b>🔍 1. Setup & Environment Configuration</b></summary>
Load essential analytical and visualization packages: <code>pandas</code>, <code>numpy</code>, <code>matplotlib</code>, <code>seaborn</code>, and <code>scikit-learn</code> modules.
</details>

<details>
<summary><b>📊 2. Exploratory Data Analysis (EDA)</b></summary>
Investigate distributions, correlation matrices, and survival discrepancies across demographic segments (e.g., gender, passenger class, ticket fare).
</details>

<details>
<summary><b>🧹 3. Preprocessing & Feature Engineering</b></summary>
Transform raw data into model-ready tensors:
<ul>
  <li>Impute missing numerical values using median/mean statistics</li>
  <li>One-hot / label encode categorical attributes (e.g., Sex, Embarked)</li>
  <li>Prune uninformative identifiers (e.g., PassengerId, Ticket, Cabin noise)</li>
  <li>Scale features via <code>StandardScaler</code> within leak-free <code>Pipeline</code> abstractions</li>
</ul>
</details>

<details>
<summary><b>🤖 4. Model Training & Ensemble Construction</b></summary>
Train baseline models, regularized regressions, and advanced ensembles:
<ul>
  <li><code>DummyClassifier</code> for minimal empirical performance threshold</li>
  <li><code>LinearRegression</code> unpenalized OLS baseline</li>
  <li><code>Ridge</code> (L2 penalty) for continuous weight shrinkage and collinearity stabilization</li>
  <li><code>Lasso</code> (L1 penalty) for automatic feature elimination and sparse modeling</li>
  <li><code>ElasticNet</code> (hybrid L1 + L2) for balanced shrinkage and feature grouping</li>
  <li><code>DecisionTreeClassifier</code> with controlled <code>max_depth</code></li>
  <li><code>LogisticRegression</code> with convergence optimization and feature scaling</li>
  <li><code>RandomForestClassifier</code> for bagging variance reduction</li>
  <li><code>AdaBoostClassifier</code> & <code>GradientBoostingClassifier</code> with tuned estimators</li>
  <li><code>KMeans</code> unsupervised clustering for customer / passenger grouping</li>
</ul>
</details>

<details>
<summary><b>📈 5. Model Evaluation & Diagnostics</b></summary>
Validate models with rigorous metrics: Confusion Matrices, Accuracy, Precision, Recall, F1-Score, MSE, $R^2$ Score, Sparsity counts, 5-Fold Stratified Cross-Validation, and stage-wise error curves.
</details>

---

## 💻 Code Sneak Peek

### 1. Decision Tree Classification (Titanic Pipeline)
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize model with reproducible seed
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Fit on preprocessed training data
model.fit(X_train, y_train)

# Predict & evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))
```

### 2. Regularization & Feature Selection (Ridge, Lasso, ElasticNet)
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

# 1. Unregularized OLS Baseline
ols = LinearRegression().fit(X_train, y_train)

# 2. Ridge Regression (L2 penalty: smooth shrinkage)
ridge = Ridge(alpha=50.0).fit(X_train, y_train)

# 3. Lasso Regression (L1 penalty: automatic feature selection)
lasso = Lasso(alpha=10.0).fit(X_train, y_train)
eliminated = (lasso.coef_ == 0).sum()
print(f"Lasso eliminated {eliminated} / {len(lasso.coef_)} features to exactly 0.0")

# 4. ElasticNet (Convex combination of L1 and L2 penalties)
elastic = ElasticNet(alpha=10.0, l1_ratio=0.5).fit(X_train, y_train)
```

### 3. Boosting Ensembles (AdaBoost & Gradient Boosting)
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Base Weak Learner (Decision Stump)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)

# 2. AdaBoost Ensemble
adaboost = AdaBoostClassifier(estimator=stump, n_estimators=50, learning_rate=1.0, random_state=42)
adaboost.fit(X_train, y_train)

# 3. Gradient Boosting Classifier
grad_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
grad_boost.fit(X_train, y_train)

print(f"AdaBoost Accuracy: {adaboost.score(X_test, y_test) * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {grad_boost.score(X_test, y_test) * 100:.2f}%")
```

### 4. Scaled Pipeline & Stratified K-Fold Cross-Validation
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

# 1. Assemble leak-proof pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=3000))
])

# 2. Fit and evaluate on hold-out set
pipeline.fit(X_train, y_train)
print(f"Test Accuracy: {pipeline.score(X_test, y_test) * 100:.2f}%")

# 3. 5-Fold Stratified Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")

print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (± {cv_scores.std() * 100:.2f}%)")
```

---

## 📊 Results & Evaluation

A summary of diagnostic metrics utilized across our models:

| Metric | Focus | Interpretation |
| :--- | :--- | :--- |
| **🎯 Accuracy Score** | Overall correctness | Ratio of correct predictions over total predictions. |
| **🧮 Confusion Matrix** | Error localization | Pinpoints Type I errors (False Positives) and Type II errors (False Negatives). |
| **⚖️ F1-Score** | Precision & Recall balance | Crucial for assessing performance on imbalanced classes. |
| **🔄 Cross-Validation (CV)** | Generalization stability | Evaluates variance across k folds to guard against train-test split bias. |
| **📉 Staged Accuracy** | Convergence tracking | Evaluates test accuracy iteration-by-iteration to detect the onset of overfitting. |
| **🌲 Feature Importance** | Interpretability | Quantifies each feature's contribution to reducing Gini impurity or variance. |
| **📐 Mean Squared Error (MSE)** | Residual variance | Quantifies continuous prediction error penalties in regression models. |
| **📈 $R^2$ Score (Goodness-of-Fit)** | Variance explained | Proportion of target variance accounted for by features. |
| **✂️ Sparsity (Zero Coefficients)** | Feature pruning | Quantifies structural weight elimination via $L_1$ penalty. |

---

## 🚀 Getting Started

Follow these steps to set up and run the notebooks on your local machine:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline.git
cd ML-Fundamentals-and-Pipeline
```

### 2️⃣ Create & Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to any of the notebooks (`Numpy_Pandas.ipynb`, `ML_Pipeline.ipynb`, `Week_3_ML_I_Sunday_.ipynb`, `Regularization.ipynb`, `Boosting.ipynb`, or `Model_Evalution.ipynb`) to run the interactive cells.

---

## 📂 Repository Structure

```
ML-Fundamentals-and-Pipeline/
├── .gitignore                    # Git ignore configurations (checkpoints, envs)
├── requirements.txt              # Pinned core project dependencies
├── README.md                     # Comprehensive project documentation & guides
├── Numpy_Pandas.ipynb            # NumPy & Pandas fundamentals
├── ML_Pipeline.ipynb             # Titanic survival end-to-end pipeline
├── Week_3_ML_I_Sunday_.ipynb     # Supervised (Trees, LogReg) & Unsupervised (K-Means)
├── Regularization.ipynb          # Ridge (L2), Lasso (L1 feature selection), ElasticNet
├── Boosting.ipynb                # Ensemble learning: AdaBoost & Gradient Boosting
└── Model_Evalution.ipynb         # Model evaluation, diagnostics & Stratified K-Fold CV
```

---

## 🔮 Roadmap & Future Enhancements

- [x] **Core Foundations:** Vectorized NumPy & Pandas data wrangling workflows.
- [x] **End-to-End Pipeline:** Titanic classification with EDA, imputation, and evaluation.
- [x] **Unsupervised Learning:** K-Means clustering and centroid analysis.
- [x] **Regularization & Shrinkage:** Ridge (L2), Lasso (L1 automated feature elimination), and ElasticNet with alpha tuning and coefficient profiling.
- [x] **Ensemble Boosting:** AdaBoost and Gradient Boosting with staged scoring and boundary visuals.
- [x] **Model Evaluation & Diagnostics:** Stratified K-Fold CV, DummyClassifier baseline, confusion matrices, and multi-model benchmarking.
- [ ] **Advanced Ensemble Models:** Implement XGBoost, LightGBM, and CatBoost pipelines.
- [ ] **Automated Hyperparameter Tuning:** Integrate `GridSearchCV`, `RandomizedSearchCV`, and Optuna.
- [ ] **Interactive Web App:** Build a Streamlit / Gradio dashboard for interactive model predictions.

---

<div align="center">
  <b>🌟 Contributions, issues, and feature requests are welcome!</b><br>
  Feel free to open an issue or submit a pull request on the <a href="https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline/issues">GitHub Issues page</a>.
  <br><br>
  <a href="#top">🔼 Back to top</a>
</div>
