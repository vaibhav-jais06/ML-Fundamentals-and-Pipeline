<div align="center">
  
# 🚢 End-to-End Machine Learning Pipeline
### *Titanic Survival Prediction Model*

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)

*An educational and highly structured approach to building a Machine Learning model from scratch.*

[**Explore the Code**](#-pipeline-architecture) • [**Getting Started**](#-getting-started) • [**Results**](#-results--evaluation)

</div>

---

## 📑 Table of Contents
- [🎯 Problem Statement](#-problem-statement)
- [📚 Included Notebooks](#-included-notebooks)
- [🛠️ Pipeline Architecture](#-pipeline-architecture)
- [🚀 Getting Started](#-getting-started)
- [📊 Results & Evaluation](#-results--evaluation)
- [💻 Code Snippets](#-code-sneak-peek)

---

## 🎯 Problem Statement
Predicting outcomes based on historical data is at the heart of Machine Learning.

> **Task (T):** Classify whether a passenger survived `(1)` or did not survive `(0)`.  
> **Experience (E):** Historical Titanic passenger records with known outcomes.  
> **Performance (P):** Model accuracy (the percentage of passengers correctly classified).  

---

## 📚 Included Notebooks
- **[`ML_Pipeline.ipynb`](./ML_Pipeline.ipynb)**: The core end-to-end Titanic survival prediction pipeline.
- **[`Numpy_Pandas.ipynb`](./Numpy_Pandas.ipynb)**: A primer on data manipulation basics using NumPy and Pandas.

---

## 🛠️ Pipeline Architecture

This repository encapsulates the standard lifecycle of an ML project. Click to expand each step!

<details>
<summary><b>1. Setup & Import Libraries</b> <i>(Click to expand)</i></summary>
<br>
We bring in powerful data science tools: <code>pandas</code> for data manipulation, <code>numpy</code> for math, <code>matplotlib/seaborn</code> for visuals, and <code>scikit-learn</code> for our ML models.
</details>

<details>
<summary><b>2. Exploratory Data Analysis (EDA)</b> <i>(Click to expand)</i></summary>
<br>
Visualizing the data! We look at survival rates based on gender, passenger class, and age distributions to find hidden patterns.
</details>

<details>
<summary><b>3. Data Preprocessing</b> <i>(Click to expand)</i></summary>
<br>
Real-world data is messy. Here, we:
<ul>
  <li>Impute missing values (like ages)</li>
  <li>Encode categorical variables (like gender) into numbers</li>
  <li>Drop irrelevant columns</li>
</ul>
</details>

<details>
<summary><b>4. Model Training</b> <i>(Click to expand)</i></summary>
<br>
We utilize a robust <code>DecisionTreeClassifier</code> to learn the rules of survival based on the processed features.
</details>

<details>
<summary><b>5. Model Evaluation</b> <i>(Click to expand)</i></summary>
<br>
We don't just train; we test! We use Confusion Matrices, Classification Reports, and Accuracy Scores to grade our AI.
</details>

---

## 🚀 Getting Started

Want to run this on your own machine? Follow these interactive steps:

### 1️⃣ Clone the Repo
```bash
git clone https://github.com/vaibhav-jais06/Machine-Learning-Pipline.git
cd Machine-Learning-Pipline
```

### 2️⃣ Install Dependencies
*Pro Tip: Use a virtual environment!*
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3️⃣ Run the Notebook
```bash
jupyter notebook ML_Pipeline.ipynb
```

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

| Metric | Definition | Importance |
|--------|------------|------------|
| **Accuracy Score** | Overall correctness | Gives a quick glance at model performance. |
| **Confusion Matrix** | True/False Positives & Negatives | Shows exactly *where* the model is making mistakes. |
| **F1-Score** | Balance of Precision & Recall | Crucial for imbalanced datasets. |

---

<div align="center">
  <b>Contributions, issues, and feature requests are welcome!</b> <br>
  Feel free to check the <a href="https://github.com/vaibhav-jais06/Machine-Learning-Pipline/issues">issues page</a>.
  
  <br><br>
  
  <a href="#top">🔼 Back to top</a>
</div>
