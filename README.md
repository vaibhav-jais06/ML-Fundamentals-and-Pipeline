<div align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Pipeline-8A2BE2?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  
  # 🚢 ML Fundamentals & Pipeline
  
  **A Comprehensive Guide to Data Manipulation & Titanic Survival Prediction**

  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8+-blue.svg?style=flat-square&logo=python&logoColor=white" /></a>
    <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.0+-F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white" /></a>
    <a href="https://pandas.pydata.org/"><img src="https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg?style=flat-square&logo=pandas&logoColor=white" /></a>
    <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=flat-square&logo=jupyter&logoColor=white" /></a>
  </p>

  *Master the foundations of Data Science and apply them to build an end-to-end Machine Learning model from scratch.*

  [**Core Notebooks**](#-core-notebooks) • [**Pipeline Architecture**](#-pipeline-architecture) • [**Installation**](#-getting-started) • [**Results**](#-results--evaluation)

</div>

---

## 🌟 About The Project

Predicting outcomes based on historical data is the heartbeat of Machine Learning. This repository serves as a two-part educational journey:
1. Mastering the essential tools of the trade (**NumPy & Pandas**).
2. Tackling the legendary **Titanic Survival Prediction** challenge using Scikit-Learn.

> **Task (T):** Classify whether a passenger survived `(1)` or did not survive `(0)`.  
> **Experience (E):** Historical Titanic passenger records with known outcomes (including Age, Sex, Ticket Class, etc.).  
> **Performance (P):** Model accuracy (percentage correctly classified) and F1-Score.  
> **Dataset Source:** [Kaggle Titanic Data](https://www.kaggle.com/c/titanic/data)

---

## 📚 Core Notebooks

This repository is split into two modular notebooks designed for progressive learning.

| Notebook | Description | Key Topics Covered |
| :--- | :--- | :--- |
| 📓 **[`Numpy_Pandas.ipynb`](./Numpy_Pandas.ipynb)** | **The Fundamentals.** A comprehensive primer on data manipulation basics. | • Array operations (NumPy)<br>• DataFrames (Pandas)<br>• Data Cleaning & Aggregation |
| 🚢 **[`ML_Pipeline.ipynb`](./ML_Pipeline.ipynb)** | **The ML Pipeline.** The core end-to-end Titanic survival prediction model. | • Exploratory Data Analysis (EDA)<br>• Feature Engineering<br>• Model Training (Decision Tree)<br>• Evaluation Metrics |
| 🤖 **[`Week_3_ML_I_Sunday_.ipynb`](./Week_3_ML_I_Sunday_.ipynb)** | **Advanced ML Models.** Deep dive into additional supervised and unsupervised ML models. | • Decision Tree Model<br>• Logistic Regression Model<br>• Unsupervised Machine Learning |

---

## 🛠️ Pipeline Architecture

The standard lifecycle of our ML project is encapsulated below. *Click to expand each step!*

<details>
<summary><b>🔍 1. Setup & Import Libraries</b></summary>
We bring in powerful data science tools: <code>pandas</code> for data manipulation, <code>numpy</code> for math, <code>matplotlib/seaborn</code> for visuals, and <code>scikit-learn</code> for ML models.
</details>

<details>
<summary><b>📊 2. Exploratory Data Analysis (EDA)</b></summary>
Visualizing the data! We look at survival rates based on gender, passenger class, and age distributions to uncover hidden patterns.
</details>

<details>
<summary><b>🧹 3. Data Preprocessing</b></summary>
Real-world data is messy. In this step, we:
<ul>
  <li>Impute missing values (like ages)</li>
  <li>Encode categorical variables (like gender) into numbers</li>
  <li>Drop irrelevant columns</li>
</ul>
</details>

<details>
<summary><b>🤖 4. Model Training</b></summary>
We utilize a robust <code>DecisionTreeClassifier</code> to learn the rules of survival based on the processed features.
</details>

<details>
<summary><b>📈 5. Model Evaluation</b></summary>
We don't just train; we test! We use Confusion Matrices, Classification Reports, and Accuracy Scores to grade our AI.
</details>

---

## 🚀 Getting Started

Want to run this on your own machine? Follow these interactive steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline.git
cd ML-Fundamentals-and-Pipeline
```

### 2️⃣ Install Dependencies
*Pro Tip: We highly recommend using a virtual environment!*
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3️⃣ Launch the Interface
Launch Jupyter Notebook to explore the code interactively:
```bash
jupyter notebook
```
*Then, simply select either `ML_Pipeline.ipynb` or `Numpy_Pandas.ipynb` from the browser interface.*

---

## 💻 Code Sneak Peek

Curious about how we train the model? Here is a quick look at the core ML code:

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Initialize the AI Model
model = DecisionTreeClassifier(random_state=42)

# 2. Teach the AI (Training)
model.fit(X_train, y_train)

# 3. Make Predictions
predictions = model.predict(X_test)

# 4. Score the Model
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
```

---

## 📊 Results & Evaluation

Our final model yields solid results across standard classification metrics:

| Metric | Importance | What it tells us |
|:---|:---|:---|
| **🎯 Accuracy Score** | Gives a quick glance at model performance. | Overall correctness of the model. |
| **🧮 Confusion Matrix**| Shows exactly *where* the model is making mistakes. | True/False Positives & Negatives. |
| **⚖️ F1-Score** | Crucial for imbalanced datasets. | The balance of Precision & Recall. |

---

## 🚀 Future Enhancements

We are always looking to improve the pipeline! Planned updates include:
- [ ] **Advanced Models:** Implementing Random Forest, XGBoost, and Support Vector Machines (SVM).
- [ ] **Hyperparameter Tuning:** Utilizing `GridSearchCV` or `RandomizedSearchCV` to optimize parameters.
- [ ] **Feature Engineering:** Extracting new features like "Family Size" or "Titles" from passenger names to boost accuracy.

---

<div align="center">
  <b>🌟 Contributions, issues, and feature requests are welcome!</b><br>
  Feel free to check the <a href="https://github.com/vaibhav-jais06/ML-Fundamentals-and-Pipeline/issues">issues page</a>.
  <br><br>
  <a href="#top">🔼 Back to top</a>
</div>
