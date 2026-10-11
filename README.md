<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Pipeline-8A2BE2?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="ML Pipeline Badge" />
  
  # 🚢 ML Fundamentals & Pipeline
  
  **A Comprehensive Guide to Data Manipulation, Supervised & Unsupervised Learning, and Ensemble Boosting**

  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" /></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" /></a>
    <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Array_Computing-013243.svg?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" /></a>
    <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg?style=flat-square" alt="Matplotlib" /></a>
    <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" /></a>
  </p>

  *From foundational mathematical array operations to building end-to-end classification pipelines and state-of-the-art boosting ensemble models.*

  [**Core Notebooks**](#-core-notebooks) • [**Ensemble & Boosting**](#-ensemble-learning--boosting-spotlight) • [**Pipeline Lifecycle**](#-pipeline-architecture) • [**Quickstart**](#-getting-started) • [**Results**](#-results--evaluation)

</div>

---

## 🌟 About The Project

Predicting outcomes and extracting meaningful patterns from data is at the core of Applied Machine Learning. This repository delivers a structured, progressive pathway through the ML lifecycle across four distinct pillars:

1. **Foundations of Data Engineering:** Vectorized operations, multidimensional indexing, DataFrame wrangling, and missing data imputation using **NumPy** and **Pandas**.
2. **End-to-End Supervised Pipeline:** Building a production-style classification workflow on the iconic **Titanic Survival Prediction** dataset.
3. **Supervised & Unsupervised Modeling:** Comparing parametric models (**Logistic Regression**) with non-parametric trees (**Decision Trees**), alongside unsupervised clustering (**K-Means**).
4. **Ensemble Methods & Boosting:** Deep-dive into sequential learning algorithms, comparing **Decision Stumps (Weak Learners)**, **AdaBoost**, and **Gradient Boosting**, complete with decision boundary plots and stage-wise error curves.

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
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
     ┌─────────────────────────────┐ ┌─────────────────────────────┐
     │  Supervised & Unsupervised  │ │  Ensemble Methods           │
     │  (LogReg, Trees, K-Means)   │ │  (AdaBoost, Gradient Boost) │
     └─────────────────────────────┘ └─────────────────────────────┘
```

> **Formal Problem Formulation (Titanic Task):**  
> • **Task (T):** Classify passenger survival binary state: `1` (Survived) or `0` (Did not survive).  
> • **Experience (E):** Passenger demographics, ticketing metadata, and cabin records ([Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)).  
> • **Performance (P):** Accuracy score, Precision, Recall, F1-Score, and ROC-AUC.

---

## 📚 Core Notebooks

Each notebook in this repository is self-contained, heavily commented, and designed for hands-on experimentation.

| Notebook | Focus Area | Key Concepts & Algorithms |
| :--- | :--- | :--- |
| 📓 **[`Numpy_Pandas.ipynb`](./Numpy_Pandas.ipynb)** | **Data Foundations** | • Multi-dimensional arrays & broadcasting<br>• DataFrame filtering, grouping & aggregation<br>• Handling null values & data type transformations |
| 🚢 **[`ML_Pipeline.ipynb`](./ML_Pipeline.ipynb)** | **End-to-End Pipeline** | • Exploratory Data Analysis (EDA) with Seaborn<br>• Feature engineering & categorical encoding<br>• `DecisionTreeClassifier` training & evaluation<br>• Confusion Matrix & Classification Report |
| 🤖 **[`Week_3_ML_I_Sunday_.ipynb`](./Week_3_ML_I_Sunday_.ipynb)** | **Supervised & Unsupervised Models** | • `LogisticRegression` hyperparameter tuning<br>• `DecisionTreeClassifier` depth pruning<br>• `KMeans` clustering on synthetic 2D datasets<br>• Visualizing decision boundaries and cluster centroids |
| ⚡ **[`Boosting.ipynb`](./Boosting.ipynb)** | **Ensemble Learning & Boosting** | • **Weak Learners:** Training single-split Decision Stumps<br>• **AdaBoost:** Iterative sample re-weighting & decision boundary<br>• **Gradient Boosting:** Residual fitting & learning rate tuning<br>• Stage-wise accuracy curves (`staged_score`)<br>• Feature importance analysis across models |

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
</ul>
</details>

<details>
<summary><b>🤖 4. Model Training & Ensemble Construction</b></summary>
Train baseline models and advanced ensembles:
<ul>
  <li><code>DecisionTreeClassifier</code> with controlled <code>max_depth</code></li>
  <li><code>LogisticRegression</code> with convergence optimization</li>
  <li><code>AdaBoostClassifier</code> & <code>GradientBoostingClassifier</code> with tuned estimators</li>
  <li><code>KMeans</code> unsupervised clustering for customer / passenger grouping</li>
</ul>
</details>

<details>
<summary><b>📈 5. Model Evaluation & Diagnostics</b></summary>
Validate models with rigorous metrics: Confusion Matrices, Accuracy, Precision, Recall, F1-Score, and stage-wise error curves.
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

### 2. Boosting Ensembles (AdaBoost & Gradient Boosting)
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

---

## 📊 Results & Evaluation

A summary of diagnostic metrics utilized across our models:

| Metric | Focus | Interpretation |
| :--- | :--- | :--- |
| **🎯 Accuracy Score** | Overall correctness | Ratio of correct predictions over total predictions. |
| **🧮 Confusion Matrix** | Error localization | Pinpoints Type I errors (False Positives) and Type II errors (False Negatives). |
| **⚖️ F1-Score** | Precision & Recall balance | Crucial for assessing performance on imbalanced classes. |
| **📉 Staged Accuracy** | Convergence tracking | Evaluates test accuracy iteration-by-iteration to detect the onset of overfitting. |
| **🌲 Feature Importance** | Interpretability | Quantifies each feature's contribution to reducing Gini impurity or variance. |

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
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 4️⃣ Launch Jupyter Notebook
```bash
jupyter notebook
```
Navigate to any of the notebooks (`Numpy_Pandas.ipynb`, `ML_Pipeline.ipynb`, `Week_3_ML_I_Sunday_.ipynb`, or `Boosting.ipynb`) to run the interactive cells.

---

## 📂 Repository Structure

```
ML-Fundamentals-and-Pipeline/
├── .gitignore                    # Git ignore configurations (checkpoints, envs)
├── README.md                     # Project documentation & guides
├── Numpy_Pandas.ipynb            # NumPy & Pandas fundamentals
├── ML_Pipeline.ipynb             # Titanic survival end-to-end pipeline
├── Week_3_ML_I_Sunday_.ipynb     # Supervised (Trees, LogReg) & Unsupervised (K-Means)
└── Boosting.ipynb                # Ensemble learning: AdaBoost & Gradient Boosting
```

---

## 🔮 Roadmap & Future Enhancements

- [x] **Core Foundations:** Vectorized NumPy & Pandas data wrangling workflows.
- [x] **End-to-End Pipeline:** Titanic classification with EDA, imputation, and evaluation.
- [x] **Unsupervised Learning:** K-Means clustering and centroid analysis.
- [x] **Ensemble Boosting:** AdaBoost and Gradient Boosting with staged scoring and boundary visuals.
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
