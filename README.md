# 🚗 Vehicle Style Classification – Machine Learning

A multi-class classification project that predicts a car's vehicle style (Sedan, SUV, Coupe, Hatchback, etc.) from its technical specifications using scikit-learn.

---

## 🧠 Overview

Given a dataset of car features (engine specs, transmission, drivetrain, fuel efficiency), the goal is to predict which vehicle style category a car belongs to — without using price (MSRP), which is intentionally excluded to ensure the model learns from physical/technical attributes only.

**Best result:** **84.8% accuracy** with a **0.83 macro F1-score**

---

## 📊 Dataset

- **Source:** Car Features and MSRP (CooperUnion)
- **File:** `data.csv`
- **Target:** `Vehicle Style` (multi-class: 16 categories)
- **Excluded:** MSRP, Make, Model, Year, Market Category (to prevent leakage/memorization)

### Features Used

| Feature | Type |
|---------|------|
| Engine Fuel Type | Categorical |
| Engine HP | Numeric |
| Engine Cylinders | Numeric |
| Transmission Type | Categorical |
| Driven Wheels | Categorical |
| Number of Doors | Numeric |
| Vehicle Size | Categorical |
| Highway MPG | Numeric |
| City MPG | Numeric |
| Popularity | Numeric |

---

## 🤖 Models Compared

Five classifiers were trained and evaluated using consistent preprocessing pipelines:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Winner selected by **Macro F1-score** (chosen over accuracy to account for class imbalance).

---

## 📈 Results Summary

| Metric | Value |
|--------|-------|
| Best Accuracy | 84.8% |
| Best Macro F1 | 0.83 |
| Evaluation Split | 80% train / 20% test (stratified) |

Full per-class classification reports, confusion matrices, and ROC curves (One-vs-Rest, top 6 classes) are available in the notebook.

---

## 🏗️ Project Structure

```
Vehicle-classificationML/
├── SE447_Project2_VehicleStyle_Classification.ipynb   # Full notebook
├── data.csv                                            # Dataset
└── SE447_ProjectReportIEEE.pdf                        # IEEE-format report
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
```

### Run the Notebook

```bash
jupyter notebook SE447_Project2_VehicleStyle_Classification.ipynb
```

> **Note:** The notebook was originally developed in Google Colab. If running locally, update the `DATA_PATH` variable to point to `data.csv` directly instead of a Drive path.

---

## 🔧 Preprocessing Pipeline

A unified `sklearn.Pipeline` is used for all models:

- **Numeric features:** Median imputation → StandardScaler
- **Categorical features:** Most-frequent imputation → OneHotEncoder
- **Label encoding:** LabelEncoder for the target variable
- **Train/test split:** 80/20 with `stratify=y` to preserve class distribution

---

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** – models, pipelines, evaluation
- **pandas / NumPy** – data processing
- **Matplotlib** – confusion matrices, ROC curves
- **Google Colab** – development environment

---

## 👤 Author

**Zakariya Ba Alawi**  
Software Engineering Student — Alfaisal University  
[LinkedIn](https://linkedin.com/in/zakariya-s-ba-alawi-a17977276) · [GitHub](https://github.com/zkr-523)
