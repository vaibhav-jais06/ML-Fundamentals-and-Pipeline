<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-3D%20Pipeline-8A2BE2?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="ML Pipeline Badge" />
  
  # 🌐 ML Fundamentals & 3D Interactive Pipeline

  **A Masterclass in Applied Machine Learning: Exploratory Data Engineering, Supervised Classification, Regularization & Sparsity, Ensemble Boosting, Cross-Validation Diagnostics, and 3D Interactive Customer Segmentation**

  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-3776AB.svg?style=flat-square&logo=python&logoColor=white" alt="Python" /></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" /></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg?style=flat-square&logo=pandas&logoColor=white" alt="Pandas" /></a>
    <a href="https://numpy.org/"><img src="https://img.shields.io/badge/NumPy-Array_Computing-013243.svg?style=flat-square&logo=numpy&logoColor=white" alt="NumPy" /></a>
    <a href="https://matplotlib.org/"><img src="https://img.shields.io/badge/Matplotlib-Visualization-11557c.svg?style=flat-square" alt="Matplotlib" /></a>
    <a href="https://seaborn.pydata.org/"><img src="https://img.shields.io/badge/Seaborn-Statistical_Plots-4C72B0.svg?style=flat-square" alt="Seaborn" /></a>
    <a href="https://plotly.com/"><img src="https://img.shields.io/badge/Plotly-3D_Interactive-3F4F75.svg?style=flat-square&logo=plotly&logoColor=white" alt="Plotly" /></a>
    <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat-square&logo=jupyter&logoColor=white" alt="Jupyter" /></a>
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome" />
    <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square" alt="License" />
  </p>

  <p align="center">
    <a href="https://htmlpreview.github.io/?https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline/blob/main/assets/interactive_3d_clusters.html"><img src="https://img.shields.io/badge/🎮_Launch-3D_Interactive_Visualizer-8A2BE2?style=for-the-badge&logo=plotly&logoColor=white" alt="Launch 3D Visualizer" /></a>
    <a href="https://colab.research.google.com/github/vaibhav-jais06/ML-Fundamentals-and-Pipeline/blob/main/KMeans_Cluster_Analysis.ipynb"><img src="https://img.shields.io/badge/Colab-Run_In_Browser-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Run In Colab" /></a>
  </p>

  *From foundational tensor indexing and feature wrangling to production-style classification pipelines, complexity penalties with Ridge/Lasso/ElasticNet, state-of-the-art ensemble boosting, leak-free stratified cross-validation, and 3D unsupervised customer segmentation via K-Means and DBSCAN.*

  [**3D Visualizer**](#-3d-interactive-manifold--cluster-diagnostics) • [**Core Notebooks**](#-core-notebooks) • [**Customer Segmentation**](#-unsupervised-customer-segmentation-spotlight) • [**Regularization & Shrinkage**](#-regularization--shrinkage-spotlight) • [**Model Evaluation**](#-model-evaluation--diagnostics-spotlight) • [**Ensemble & Boosting**](#-ensemble-learning--boosting-spotlight) • [**Architecture**](#-3d-pipeline-architecture) • [**Code Sneak Peek**](#-code-sneak-peek) • [**Quickstart**](#-getting-started)

</div>

---

## 🏗️ 3D Pipeline Architecture

<p align="center">
  <img src="./assets/pipeline_3d_architecture.svg" alt="3D Architecture Pipeline Topology" width="100%" />
</p>

```mermaid
flowchart TD
    classDef base fill:#312e81,stroke:#818cf8,stroke-width:2px,color:#fff;
    classDef supervised fill:#78350f,stroke:#fbbf24,stroke-width:2px,color:#fff;
    classDef regular fill:#064e3b,stroke:#34d399,stroke-width:2px,color:#fff;
    classDef ensemble fill:#164e63,stroke:#22d3ee,stroke-width:2px,color:#fff;
    classDef cv fill:#1e3a8a,stroke:#60a5fa,stroke-width:2px,color:#fff;
    classDef clustering fill:#701a75,stroke:#f472b6,stroke-width:2px,color:#fff;

    P1["📦 Pillar 1: Tensor & Array Foundations<br/>(NumPy Vectorization & Pandas Wrangling)"]:::base
    P2["🚢 Pillar 2: End-to-End Supervised ML<br/>(Titanic EDA, Imputation & Decision Trees)"]:::supervised
    P3["🎯 Pillar 3: Regularization & Shrinkage<br/>(Ridge L2, Lasso L1 Sparsity, ElasticNet)"]:::regular
    P4["⚡ Pillar 4: Sequential Ensemble Boosting<br/>(Decision Stumps, AdaBoost, Gradient Boosting)"]:::ensemble
    P5["🧪 Pillar 5: Model Evaluation & 5-Fold CV<br/>(Dummy Baseline, PR/ROC, Stratified K-Fold)"]:::cv
    P6["🌐 Pillar 6: 3D Latent Customer Segmentation<br/>(Elbow Inertia, Silhouette Peak 0.5547, K=5, DBSCAN)"]:::clustering

    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> P5
    P5 --> P6
```

---

## 🌐 3D Interactive Manifold & Cluster Diagnostics

Experience our unsupervised retail customer segmentation in full **3D interactive space** ($N=200$ consumers). By projecting customer attributes across three quantitative dimensions—`Age` ($X$), `Annual Income` ($Y$), and `Spending Score` ($Z$)—the natural clustering topology becomes immediately apparent.

<p align="center">
  <a href="https://htmlpreview.github.io/?https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline/blob/main/assets/interactive_3d_clusters.html">
    <img src="./assets/customer_clusters_3d_preview.png" alt="3D Customer Segmentation Manifold" width="90%" style="border-radius:12px; box-shadow: 0 10px 30px rgba(0,0,0,0.5);" />
  </a>
</p>

<div align="center">
  <a href="https://htmlpreview.github.io/?https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline/blob/main/assets/interactive_3d_clusters.html"><b>👉 Click Here to Open Live 3D Interactive Canvas in Browser (Full 360° Rotation)</b></a><br>
  <i>(Or open <a href="./assets/interactive_3d_clusters.html"><code>assets/interactive_3d_clusters.html</code></a> locally in Chrome/Firefox/Edge)</i>
</div>

<br>

### 🎮 Interactive Controls & 3D Diagnostics Features:
* **🔄 360° Free Space Rotation:** Click and drag anywhere on the canvas to rotate along Pitch, Yaw, and Roll to inspect cluster boundaries from any viewing angle.
* **🔍 Dynamic Panning & Zoom:** Scroll to zoom directly into dense cluster intersections between *Young Trendsetters* and *High-Value VIPs*.
* **🏷️ Real-Time Multi-Attribute Tooltips:** Hover over any 3D data point to inspect `CustomerID`, `Age`, `Gender`, `Annual Income ($k)`, `Spending Score (1-100)`, and assigned `Customer Persona`.
* **👁️ Interactive Legend Isolation:** Click any persona in the legend to isolate that specific market segment or double-click to solo.

### 📐 3D Mathematical Centroid Tensors:
The cluster centroids represent the center-of-mass coordinates $\boldsymbol{\mu}_k \in \mathbb{R}^3$ in three-dimensional space:

$$\boldsymbol{\mu}_k = \left( \frac{1}{|C_k|} \sum_{i \in C_k} \text{Age}_i, \quad \frac{1}{|C_k|} \sum_{i \in C_k} \text{Income}_i, \quad \frac{1}{|C_k|} \sum_{i \in C_k} \text{Spending}_i \right)$$

| Cluster | Persona Title | 3D Centroid Coordinate $(\text{Age}, \text{Income}, \text{Spending})$ | Volume Share | Commercial Strategy |
| :---: | :--- | :---: | :---: | :--- |
| **0** | **Standard (Mainstream)** | **$(42.72\text{ yrs}, \; \$55.30\text{k}, \; 49.52/100)$** | **40.5%** | Baseline loyalty programs & recurring value bundles |
| **1** | **VIP / High-Value Champions** | **$(32.69\text{ yrs}, \; \$86.54\text{k}, \; 82.13/100)$** | **19.5%** | White-glove concierge, luxury previews, VIP access |
| **2** | **Young Trendsetters** | **$(25.27\text{ yrs}, \; \$25.73\text{k}, \; 79.36/100)$** | **11.0%** | Social commerce, influencer collabs, BNPL financing |
| **3** | **Careful Savers** | **$(41.11\text{ yrs}, \; \$88.20\text{k}, \; 17.11/100)$** | **17.5%** | Premium durability warranties, high-yield rewards |
| **4** | **Sensible Budget Shoppers** | **$(45.22\text{ yrs}, \; \$26.30\text{k}, \; 20.91/100)$** | **11.5%** | Price-slashing clearance, discount coupons, essentials |

---

## 🌟 About The Project

Predicting outcomes, extracting latent structure, and optimizing decision boundaries is the core discipline of Applied Machine Learning. This repository delivers an industry-grade, highly pedagogical curriculum structured across **seven interconnected pillars**:

1. **Foundations of Data Engineering:** Vectorized math, multidimensional indexing, DataFrame wrangling, slicing, and missing data imputation using **NumPy** and **Pandas**.
2. **End-to-End Supervised Pipeline:** Building a production-style classification workflow on the iconic **Titanic Survival Prediction** dataset with EDA, feature imputation, encoding, and decision trees.
3. **Supervised & Unsupervised Modeling Foundations:** Comparing parametric models (**Logistic Regression**) with non-parametric decision boundaries (**Decision Trees**), alongside foundational unsupervised clustering (**K-Means**).
4. **Regularization & Shrinkage Methods:** Taming overfitting, suppressing multicollinearity, and driving automated feature selection via **L2 (Ridge)**, **L1 (Lasso)**, and hybrid **ElasticNet** penalties.
5. **Ensemble Methods & Sequential Boosting:** Mastering weak learner theory, decision stumps, iterative sample re-weighting via **AdaBoost**, and pseudo-residual gradient optimization via **Gradient Boosting** with staged loss curves.
6. **Model Evaluation & Validation Diagnostics:** Rigorous evaluation methodologies using the **Wisconsin Breast Cancer** diagnostic dataset, contrasting naive baselines (`DummyClassifier`) against standard pipelines (`StandardScaler` + `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`), confusion matrices, classification reports, and 5-Fold `StratifiedKFold` cross-validation.
7. **Unsupervised Customer Segmentation & Density Clustering:** Full-lifecycle behavioral segmentation on the **Mall Customers** dataset, leveraging feature standardization, Elbow curve optimization (WCSS / Inertia), Silhouette score validation, 2D centroid mapping, **Plotly 3D/2D interactive exploration**, and density-based clustering via **DBSCAN**.

> **Formal Real-World Problem Formulations:**  
> • **Titanic Survival Classification Task:** Predict binary passenger survival state: $y \in \{0, 1\}$ (0 = Deceased, 1 = Survived) from passenger socio-demographics, fare classes, and cabin records ([Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)).  
> • **Wisconsin Diagnostic Oncology Task:** Classify cellular biopsy nuclei as benign ($1$) or malignant ($0$) across 30 morphological real-valued features ([Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)).  
> • **Controlled Sparsity & Shrinkage Task:** Predict continuous targets from 10 features (5 true informative signals, 5 Gaussian noise features) to benchmark mathematical coefficient shrinkage and $L_1$ variable selection.  
> • **Mall Customer Behavioral Segmentation Task:** Partition $N=200$ retail consumers into actionable commercial personas based on Annual Income ($k\$$) and Spending Score ($1..100$) using unsupervised Euclidean & Density clustering ([Mall Customers Dataset](./Mall_Customers.csv)).

---

## 📚 Core Notebooks

Every notebook in this repository is completely self-contained, heavily documented with mathematical formulations, and optimized for both local Jupyter environments and Google Colab.

| Notebook | Focus Area | Key Algorithms & Methods | Dataset / Problem |
| :--- | :--- | :--- | :--- |
| 📓 **[`Numpy_Pandas.ipynb`](./Numpy_Pandas.ipynb)** | **Data Foundations** | • Multi-dimensional tensor broadcasting<br>• Vectorized filtering & aggregation<br>• Missing value imputation & type casting | Iris & Synthetic Tensors |
| 🚢 **[`ML_Pipeline.ipynb`](./ML_Pipeline.ipynb)** | **End-to-End Pipeline** | • Exploratory Data Analysis (EDA) with Seaborn<br>• Missing data handling & categorical encoding<br>• `DecisionTreeClassifier` depth tuning<br>• Confusion Matrix & Classification Report | [Titanic Survival Dataset](https://www.kaggle.com/c/titanic/data) |
| 🤖 **[`Week_3_ML_I_Sunday_.ipynb`](./Week_3_ML_I_Sunday_.ipynb)** | **Supervised & Unsupervised** | • `LogisticRegression` hyperparameter tuning<br>• Decision boundary visualization<br>• `DecisionTreeClassifier` depth pruning<br>• Foundational `KMeans` on synthetic 2D clusters | Scikit-Learn Blobs, Moons & Iris |
| 🎯 **[`Regularization.ipynb`](./Regularization.ipynb)** | **Regularization & Feature Selection** | • **OLS Baseline:** Unregularized regression benchmarking<br>• **Ridge ($L_2$):** Smooth coefficient shrinkage & collinearity control<br>• **Lasso ($L_1$):** Exact zero sparsity & automated feature elimination<br>• **ElasticNet:** Convex combination balancing sparsity & grouping<br>• Hyperparameter sensitivity sweeps ($\alpha$ & `l1_ratio`) | Synthetic Gauss-Markov Regression |
| ⚡ **[`Boosting.ipynb`](./Boosting.ipynb)** | **Ensemble Learning & Boosting** | • **Weak Learners:** Single-split Decision Stumps (`max_depth=1`)<br>• **AdaBoost:** Iterative sample weight updating & adaptive voting<br>• **Gradient Boosting:** Pseudo-residuals & shrinkage learning rate<br>• Stage-wise accuracy tracking (`staged_score`)<br>• Feature importance extraction | Synthetic Classification Boundary |
| 🧪 **[`Model_Evalution.ipynb`](./Model_Evalution.ipynb)** | **Evaluation & Cross-Validation** | • Baseline benchmarking (`DummyClassifier`)<br>• Precision, Recall, F1, Accuracy metric suite<br>• Leak-free `Pipeline` (`StandardScaler` + `LogisticRegression`)<br>• Multi-model comparison (LogReg, Tree, Random Forest)<br>• 5-Fold `StratifiedKFold` Cross-Validation | [Wisconsin Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) |
| 🛍️ **[`KMeans_Cluster_Analysis.ipynb`](./KMeans_Cluster_Analysis.ipynb)** <br> *(also [`KMeans_Cluser_analysis.ipynb`](./KMeans_Cluser_analysis.ipynb))* | **3D Customer Segmentation & Diagnostics** | • Standard feature scaling (`StandardScaler`)<br>• **Elbow Method:** Inertia (WCSS) sweeps ($K=1..10$)<br>• **Silhouette Analysis:** Quantitative validation ($K=2..8$)<br>• **K-Means Model:** 5 customer personas & centroid mapping<br>• **DBSCAN:** Density-based clustering & noise detection<br>• **Plotly Express 3D:** Interactive customer exploration | [Mall Customers Dataset](./Mall_Customers.csv) (200 records) |

---

## 🛍️ Unsupervised Customer Segmentation Spotlight

The **[`KMeans_Cluster_Analysis.ipynb`](./KMeans_Cluster_Analysis.ipynb)** notebook provides an end-to-end commercial case study in customer segmentation, evaluating how unsupervised algorithms uncover latent behavioral archetypes from real customer data.

```
[Feature Standardization] ──► [Elbow & Silhouette Validation] ──► [K-Means (K=5)] ──► [3D Plotly & DBSCAN]
  (Zero Mean, Unit Var)          (Inertia + Silhouette Peak)        (5 Clear Personas)     (360° Free Space Canvas)
```

<details>
<summary><b>🔍 Expand to View the 15-Part Curriculum in KMeans_Cluster_Analysis.ipynb</b></summary>
<br>

1. **Part 1: Environment Setup & Library Imports** — Pandas, NumPy, Matplotlib, Seaborn, Plotly Express, and Scikit-Learn clustering modules (`KMeans`, `DBSCAN`, `StandardScaler`, `silhouette_score`).
2. **Part 2: Robust Universal Dataset Loading** — Seamless execution across local directories, Google Colab, and remote GitHub fallbacks.
3. **Part 3: Dataset Inspection & Statistical Profiling** — Checking schema types, dimension verification ($200 \times 5$), missing value scans, and duplicate checks.
4. **Part 4: Exploratory Data Analysis (EDA) — Univariate Distributions** — Kernel density estimation and histogram plotting for `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`.
5. **Part 5: Bivariate Analysis & Demographics** — Evaluating interactions between Income, Spending, and Gender via Seaborn scatter diagnostics.
6. **Part 6: Feature Selection for Clustering** — Isolating primary behavioral drivers: `Annual Income (k$)` and `Spending Score (1-100)`.
7. **Part 7: Feature Standardization (`StandardScaler`)** — Rescaling heterogeneous units to ensure equal geometric weighting in Euclidean distance calculations.
8. **Part 8: Optimal K Determination via Elbow Method** — Iterating $K \in [1, 10]$ and plotting Within-Cluster Sum of Squares (Inertia) to locate the inflection point.
9. **Part 9: Quantitative Validation via Silhouette Score Analysis** — Evaluating silhouette coefficients for $K \in [2, 8]$ to verify cluster compactness and inter-cluster separation.
10. **Part 10: Training the Optimal K-Means Model ($K=5$)** — Fitting the final model with 10 random restarts (`n_init=10`) and assigning cluster labels.
11. **Part 11: 2D Static Cluster Visualization with Centroids** — High-resolution Seaborn scatter plot with inverse-transformed centroid coordinates marked with prominent crosses.
12. **Part 12: Quantitative Cluster Profiling** — Aggregating mean age, annual income, spending scores, and customer counts per cluster.
13. **Part 13: Business Persona Profiling & Marketing Strategies** — Translating mathematical clusters into actionable retail personas (VIP Champions, Trendsetters, Savers, Budget Shoppers, Mainstream).
14. **Part 14: Interactive 3D & 2D Cluster Exploration (Plotly)** — Rich web-based 3D scatter plot with multi-attribute hover tooltips (`CustomerID`, `Age`, `Gender`, `Segment`) and 360-degree rotation.
15. **Part 15: Density-Based Clustering Comparison (DBSCAN)** — Benchmarking K-Means against `DBSCAN(eps=0.35, min_samples=5)` to identify arbitrary density formations and isolate outliers/noise.

</details>

### Mathematical Formulations

Unsupervised clustering relies on distance metrics in standardized Euclidean space:

* **Within-Cluster Sum of Squares (Inertia / WCSS):**  
  Measures the internal coherence and compactness of clusters:
  $$J = \sum_{k=1}^K \sum_{x_i \in C_k} \| x_i - \boldsymbol{\mu}_k \|^2$$
  where $\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$ is the centroid of cluster $C_k$.

* **Silhouette Coefficient:**  
  Quantifies how well an individual point $i$ is matched to its own cluster compared to neighboring clusters:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, +1]$$
  where $a(i)$ is the mean intra-cluster distance of point $i$, and $b(i)$ is the mean nearest-cluster distance. A global average score close to $+1$ indicates dense, well-isolated clusters.

* **DBSCAN Density Reachability:**  
  Identifies core samples in dense regions where the $\varepsilon$-neighborhood contains at least $\text{MinPts}$ points:
  $$\mathcal{N}_\varepsilon(p) = \{ q \in D \mid \text{dist}(p, q) \le \varepsilon \}, \quad |\mathcal{N}_\varepsilon(p)| \ge \text{MinPts}$$
  Points not reachable from any core sample are categorized as noise ($y = -1$).

### Quantitative Optimization Benchmark

To objectively determine the optimal number of clusters, both Inertia (WCSS) and Silhouette scores were evaluated across $K \in [2, 8]$ on standardized features:

| Number of Clusters ($K$) | Inertia (WCSS) | Silhouette Score | Geometric & Separation Evaluation |
| :---: | :---: | :---: | :--- |
| **$K = 2$** | 269.69 | 0.3213 | Broad macro-split; high within-cluster variance |
| **$K = 3$** | 157.70 | 0.4666 | Captures low, medium, and high spenders partially |
| **$K = 4$** | 108.92 | 0.4939 | Under-segments the middle-income demographic |
| **$K = 5$** | **65.57** | **0.5547** | 🏆 **Optimal Global Inflection:** Peak Silhouette Score & Clear Elbow Point |
| **$K = 6$** | 55.06 | 0.5399 | Begins fragmenting natural dense clusters |
| **$K = 7$** | 44.86 | 0.5281 | Artificial sub-clustering with diminishing returns |
| **$K = 8$** | 37.23 | 0.4552 | Significant drop in silhouette separation |

### Customer Persona Profiles & Commercial Strategies

| Cluster ID | Segment Persona | Customer Count | Share (%) | Avg Age | Avg Income | Avg Spending | Strategic Business Playbook |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **0** | **Standard / Mainstream** | **81** | 40.5% | 42.7 yrs | \$55.3k | 49.5 / 100 | **Core Customer Base:** Moderate income and spending. Retain through personalized loyalty points, bundle discounts, and seasonal offers. |
| **1** | **VIP Champions / High-Value** | **39** | 19.5% | 32.7 yrs | **\$86.5k** | **82.1 / 100** | **Highest Lifetime Value (LTV):** High income and high propensity to spend. Target with luxury releases, premium memberships, private shopping events, and concierge service. |
| **2** | **Young Trendsetters** | **22** | 11.0% | **25.3 yrs** | \$25.7k | **79.4 / 100** | **High Engagement, Budget-Constrained:** Young demographic driven by lifestyle and trends. Engage via social commerce, influencer campaigns, flash drops, and Buy-Now-Pay-Later (BNPL). |
| **3** | **Careful Savers** | **35** | 17.5% | 41.1 yrs | **\$88.2k** | 17.1 / 100 | **Affluent but Conservative:** High income with minimal discretionary spending. Win over with quality guarantees, durability assurances, premium utility, and financial investment products. |
| **4** | **Sensible Budget Shoppers** | **23** | 11.5% | 45.2 yrs | \$26.3k | 20.9 / 100 | **Price-Conscious:** Low income and low spending. Attract with clearance sales, discount coupons, budget essentials, and cashback rewards. |

---

## 🎯 Regularization & Shrinkage Spotlight

The **[`Regularization.ipynb`](./Regularization.ipynb)** notebook provides an in-depth masterclass on preventing overfitting, penalizing complexity, handling multicollinearity, and performing automated feature selection using **L2 Ridge**, **L1 Lasso**, and hybrid **ElasticNet** models.

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
12. **Part 12: Visualizing Weight Profiles** — Comparative bar chart visualizing feature coefficients across all four regression paradigms.
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

#### 🎯 Test Set Metrics (40 Hold-out Samples):
| Model | Penalty Type | Hyperparameters | Test MSE | Test $R^2$ Score | Zero Coefficients | Feature Selection Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 📉 **Linear Regression** | None (OLS) | Unregularized | **325.12** | **0.949** | 0 / 10 (0%) | Retains all features; susceptible to collinear noise |
| 🛡️ **Ridge Regression** | $L_2$ Penalty | $\alpha = 50.0$ | 866.39 | 0.863 | 0 / 10 (0%) | Smooth shrinkage; shrinks weights towards 0 without setting to 0 |
| ✂️ **Lasso Regression** | $L_1$ Penalty | $\alpha = 10.0$ | 761.83 | 0.880 | **7 / 10 (70%)** | **Automated Selection:** Retains only dominant signals (`Location`, `Distance`, `Garden`) |
| ⚖️ **ElasticNet** | $L_1 + L_2$ Hybrid | $\alpha = 10.0, \rho = 0.5$ | 4973.97 | 0.215 | 3 / 10 (30%) | Balances feature grouping with selective sparsity |

#### 📊 Feature Coefficient Evolution:
| Feature | Linear Regression | Ridge ($\alpha=50$) | Lasso ($\alpha=10$) | ElasticNet ($\alpha=10, \rho=0.5$) | Feature Status in Lasso |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`Location`** | 79.956 | 59.982 | **70.505** | 12.212 | 🌟 **Retained** (Dominant true signal) |
| **`Distance`** | 40.753 | 30.495 | **29.188** | 5.646 | 🌟 **Retained** (True informative) |
| **`Garden`** | 36.273 | 29.524 | **28.814** | 6.834 | 🌟 **Retained** (True informative) |
| **`Floor`** | 8.599 | 7.579 | **0.000** | 1.651 | ✂️ **Pruned to Zero** |
| **`Age`** | 7.122 | 3.333 | **0.000** | 0.000 | ✂️ **Pruned to Zero** |
| **`Area`** | 1.518 | 2.568 | **0.000** | 0.372 | ✂️ **Pruned to Zero** |
| **`School`** | 1.208 | 1.263 | **0.000** | 0.000 | ✂️ **Pruned to Zero** |
| **`Parking`** | 1.093 | -0.190 | **0.000** | 0.000 | ✂️ **Pruned to Zero** |
| **`Transport`** | -1.104 | -1.868 | **0.000** | -0.422 | ✂️ **Pruned to Zero** |
| **`Bedrooms`** | -1.805 | -3.003 | **0.000** | -0.771 | ✂️ **Pruned to Zero** |

---

## 🧪 Model Evaluation & Diagnostics Spotlight

The **[`Model_Evalution.ipynb`](./Model_Evalution.ipynb)** notebook provides an in-depth, 15-part masterclass on assessing machine learning models reliably, detecting overfitting, and preventing data leakage.

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
| Model | Accuracy | Precision | Recall | F1-Score | Evaluation & Diagnostic Insight |
| :--- | :---: | :---: | :---: | :---: | :--- |
| 🥇 **Logistic Regression (Pipeline)** | **98.25%** | **98.61%** | 98.61% | **98.61%** | Highest precision and harmonic F1; optimal linear separation |
| 🥈 **Random Forest (100 Trees)** | 95.61% | 95.89% | **97.22%** | 96.55% | Strong recall; bagging mitigates individual tree variance |
| 🥉 **Decision Tree (Depth 4)** | 93.86% | 95.77% | 94.44% | 95.10% | Fully interpretable white-box rules; sensitive to split thresholds |
| ⚪ *Dummy Baseline (Most Frequent)* | *62.74%* | *62.74%* | *100.0%* | *77.10%* | Empirical floor baseline; zero discriminative power |

#### 🔄 5-Fold Stratified Cross-Validation (Training Folds):
| Model | Mean CV Accuracy | CV Std Dev ($\sigma$) | Generalization & Stability |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (Pipeline)** | **97.80%** | **± 0.0098** | 🟢 Extremely stable and robust across folds |
| **Random Forest Classifier** | 96.26% | ± 0.0179 | 🟢 Strong ensemble variance reduction |
| **Decision Tree Classifier** | 92.97% | ± 0.0149 | 🟡 Higher sensitivity to data partitioning |

---

## ⚡ Ensemble Learning & Boosting Spotlight

The **[`Boosting.ipynb`](./Boosting.ipynb)** notebook provides an in-depth masterclass on boosting mechanics and sequential ensemble learning:

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
12. **Part 12: Interactive Hyperparameter Experimentation** — Tuning playground for `n_estimators`, `learning_rate`, and tree depth.

</details>

---

## 💻 Code Sneak Peek

### 1. 3D Customer Segmentation & Manifold Projection (Plotly Express)
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# 1. Load retail customer dataset
df = pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit optimal K-Means model (K=5)
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# 3. Render 3D Interactive Manifold (360° Free Space Canvas)
fig = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster",
    hover_data=["CustomerID", "Age", "Gender"],
    title="3D Customer Segmentation Manifold"
)
fig.show()
```

### 2. Regularization & Feature Selection (Ridge, Lasso, ElasticNet)
```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 1. Ordinary Least Squares Baseline
ols = LinearRegression().fit(X_train, y_train)

# 2. Ridge Regression (L2 penalty: continuous shrinkage)
ridge = Ridge(alpha=50.0).fit(X_train, y_train)

# 3. Lasso Regression (L1 penalty: automated variable elimination)
lasso = Lasso(alpha=10.0).fit(X_train, y_train)
eliminated = (lasso.coef_ == 0).sum()
print(f"Lasso zeroed out {eliminated} / {len(lasso.coef_)} coefficients.")

# 4. ElasticNet (Convex combination of L1 and L2 penalties)
elastic = ElasticNet(alpha=10.0, l1_ratio=0.5).fit(X_train, y_train)
```

### 3. Boosting Ensembles (AdaBoost & Gradient Boosting)
```python
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# 1. Weak Base Learner (Decision Stump)
stump = DecisionTreeClassifier(max_depth=1, random_state=42)

# 2. Sequential AdaBoost Ensemble
adaboost = AdaBoostClassifier(estimator=stump, n_estimators=50, learning_rate=1.0, random_state=42)
adaboost.fit(X_train, y_train)

# 3. Gradient Boosting Classifier (Pseudo-Residual Fitting)
grad_boost = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
grad_boost.fit(X_train, y_train)

print(f"AdaBoost Accuracy: {adaboost.score(X_test, y_test) * 100:.2f}%")
print(f"Gradient Boosting Accuracy: {grad_boost.score(X_test, y_test) * 100:.2f}%")
```

### 4. Leak-Proof Scaled Pipeline & Stratified Cross-Validation
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

## 📊 Evaluation & Diagnostics Matrix

A comprehensive reference matrix of the mathematical diagnostic metrics used across this repository:

| Metric | Paradigm | Analytical Focus | Interpretation & Decision Rule |
| :--- | :---: | :--- | :--- |
| **🎯 Accuracy Score** | Supervised | Global correctness | Ratio of correct predictions over total instances; unreliable under class imbalance. |
| **🧮 Confusion Matrix** | Supervised | Error localization | Pinpoints Type I errors (False Positives) and Type II errors (False Negatives). |
| **⚖️ Precision & Recall** | Supervised | Trade-off tuning | Precision safeguards against false alarms; Recall prevents missed detections. |
| **📈 F1-Score** | Supervised | Harmonic balance | Harmonic mean of Precision and Recall; essential for imbalanced class assessment. |
| **🔄 Stratified K-Fold CV** | Supervised | Stability & variance | Partitions dataset into $k$ balanced folds to detect split bias and measure variance ($\sigma$). |
| **📉 Staged Loss Curves** | Ensembles | Convergence tracking | Tracks iteration-by-iteration performance to diagnose underfitting vs overfitting onset. |
| **🌲 Gini Feature Importance** | Tree Models | Interpretability | Quantifies relative variance or impurity reduction contributed by each attribute. |
| **📐 Mean Squared Error (MSE)** | Regression | Residual penalty | Quadratically penalizes large continuous prediction residuals. |
| **📊 $R^2$ Score** | Regression | Variance explained | Proportion of dependent variable variance captured by regression features. |
| **✂️ Sparsity Ratio** | Regularization | Model compression | Fraction of feature coefficients driven to exact zero via $L_1$ penalty. |
| **📉 Inertia (WCSS)** | Unsupervised | Cluster compactness | Sum of squared Euclidean distances to assigned centroids; used in Elbow heuristic. |
| **✨ Silhouette Coefficient** | Unsupervised | Separation & cohesion | Ranges from $-1$ to $+1$; values $>0.5$ indicate well-defined, dense cluster structures. |
| **📍 Density Core Ratio** | Unsupervised | Outlier detection | Fraction of samples residing within $\varepsilon$-density boundaries vs isolated noise. |

---

## 🚀 Getting Started

Follow these steps to set up and run the repository on your local workstation:

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

### 4️⃣ Launch Jupyter Notebook or JupyterLab
```bash
jupyter notebook
```
Navigate to any of the notebooks to interact with the visualizations:
* **3D Customer Segmentation:** [`KMeans_Cluster_Analysis.ipynb`](./KMeans_Cluster_Analysis.ipynb)
* **Regularization & Shrinkage:** [`Regularization.ipynb`](./Regularization.ipynb)
* **Model Evaluation & CV:** [`Model_Evalution.ipynb`](./Model_Evalution.ipynb)
* **Ensemble Boosting:** [`Boosting.ipynb`](./Boosting.ipynb)
* **End-to-End Pipeline:** [`ML_Pipeline.ipynb`](./ML_Pipeline.ipynb)

---

## 📂 Repository Structure

```
ML-Fundamentals-and-Pipeline/
├── .gitignore                      # Git ignore rules (checkpoints, virtual environments, caches)
├── requirements.txt                # Pinned production dependencies (scikit-learn, plotly, etc.)
├── README.md                       # Comprehensive documentation, curriculum & benchmarks
├── Mall_Customers.csv              # Benchmark dataset for customer segmentation (200 records)
├── assets/                         # 3D Visual assets, interactive exports & SVG topologies
│   ├── pipeline_3d_architecture.svg    # 3D Isometric multilayer pipeline SVG diagram
│   ├── customer_clusters_3d_preview.png # High-resolution 3D customer manifold render
│   └── interactive_3d_clusters.html    # Standalone 3D interactive Plotly Web Visualizer
├── Numpy_Pandas.ipynb              # Pillar 1: Tensor operations, broadcasting & DataFrame wrangling
├── ML_Pipeline.ipynb               # Pillar 2: Titanic survival classification end-to-end pipeline
├── Week_3_ML_I_Sunday_.ipynb       # Pillar 3: Supervised (Trees, LogReg) & Unsupervised 2D clustering
├── Regularization.ipynb            # Pillar 4: Ridge (L2), Lasso (L1 sparsity), ElasticNet hybrid
├── Boosting.ipynb                  # Pillar 5: Decision Stumps, AdaBoost & Gradient Boosting
├── Model_Evalution.ipynb           # Pillar 6: Model evaluation, diagnostics & Stratified K-Fold CV
├── KMeans_Cluster_Analysis.ipynb   # Pillar 7: Customer segmentation, Elbow, Silhouette & DBSCAN
└── KMeans_Cluser_analysis.ipynb    # Backward-compatible notebook alias
```

---

## 🔮 Roadmap & Future Horizons

- [x] **Data Engineering Foundations:** Vectorized NumPy & Pandas data wrangling workflows.
- [x] **End-to-End Pipeline:** Titanic classification with EDA, imputation, and evaluation.
- [x] **Classical Machine Learning:** Logistic Regression, Decision Trees, and 2D clustering.
- [x] **Regularization & Shrinkage:** Ridge ($L_2$), Lasso ($L_1$ feature elimination), and ElasticNet with sensitivity sweeps.
- [x] **Ensemble Boosting:** AdaBoost and Gradient Boosting with staged scoring and boundary visuals.
- [x] **Model Evaluation & Diagnostics:** Stratified K-Fold CV, DummyClassifier baseline, and confusion matrix diagnostics.
- [x] **3D Unsupervised Customer Segmentation:** Mall customer clustering, WCSS Elbow curve, Silhouette validation ($K=2..8$), persona profiling, and 3D interactive Web Visualizer.
- [ ] **Advanced Gradient Boosters:** Implement XGBoost, LightGBM, and CatBoost pipelines.
- [ ] **Automated Hyperparameter Optimization:** Integrate Optuna, `GridSearchCV`, and Bayesian Optimization.
- [ ] **Interactive Web Application:** Build a Streamlit / Gradio dashboard for real-time model inference and customer segment exploration.

---

<div align="center">
  <b>🌟 Contributions, issues, and feature requests are warmly welcomed!</b><br>
  Feel free to open an issue or submit a pull request on the <a href="https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline/issues">GitHub Issues page</a>.
  <br><br>
  <a href="#top">🔼 Back to top</a>
</div>
