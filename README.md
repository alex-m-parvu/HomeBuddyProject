
# Predictive Pipeline for Lead Appointment Setting

This project defines a predictive modeling pipeline designed to process lead generation datasets, apply feature engineering, train a machine learning model, and evaluate its performance. It primarily focuses on predicting whether an appointment will be set for a given lead.

---

## 📦 Dependencies

The code relies on the following major libraries:

- `pandas`
- `scikit-learn`
- `matplotlib`
- Custom modules from `DS_Helpers`:
  - `filters`
  - `data`
  - `metrics`
  - `models`

---

## 🔧 Machine Learning Pipeline

```python
pipe = make_pipeline(
    simple_preprocessing,
    StandardScaler(),
    log_scaler,
    RandomForestClassifier(n_estimators=200, n_jobs=-1)
)
```

### Components:

- **`simple_preprocessing`**: Custom preprocessing from `DS_Helpers`.
- **`StandardScaler`**: Standardizes features by removing the mean and scaling to unit variance.
- **`log_scaler`**: Custom transformation to reduce the effect of outliers.
- **`RandomForestClassifier`**: Scikit-learn ensemble classifier using 200 trees and all CPU cores.

---

## 🔄 `pred_pipe` Class

### Purpose

Encapsulates the end-to-end modeling workflow:
- Data loading and cleaning
- Merging auxiliary datasets
- Feature engineering
- Train-test split
- Training and evaluation

---

### 📥 Initialization

```python
pred = pred_pipe(
    leads='data/leads_dataset.csv',
    infutor_enrichment_dataset='data/infutor_enrichment_dataset.csv',
    zip_code_dataset='data/zip_code_dataset.csv',
    country_code='US'
)
```

#### What Happens:

1. **Loads** the lead, enrichment, and zip code datasets.
2. **Filters columns** based on missing data thresholds.
3. **Merges** the datasets on phone number and ZIP code.
4. **Adds** a new feature: days to the nearest holiday (based on lead creation date).
5. **Cleans** the final dataset by dropping rows with missing target values and ID fields.

---

### ✂️ Train-Test Split

```python
pred.train_test_split(test_size=0.3, random_state=42, embedding_dims=8)
```

#### Steps:

- Splits dataset into training and test sets.
- Extracts datetime features (year, month, weekday, etc.).
- Replaces infrequent categorical values with `"other"`.
- Applies embedding transformations to categorical features via helper methods.

---

### 🧠 Training the Model

```python
pred.train(pipe=pipe)
```

Trains the `RandomForestClassifier` pipeline on the processed training dataset.

---

### 📊 Evaluation

#### 1. **Precision-Recall Curve**

```python
pred.evaluate_model()
```

- Plots the precision-recall curve for model performance visualization.

#### 2. **Classification Metrics**

```python
pred.evaluate_model_metrics()
```

- Outputs standard classification metrics using a custom helper:
  - Accuracy
  - Precision
  - Recall
  - F1-score, etc.

---

## 📁 File Organization

```
project/
│
├── data/
│   ├── leads_dataset.csv
│   ├── infutor_enrichment_dataset.csv
│   └── zip_code_dataset.csv
│
├── DS_Helpers/
│   ├── filters.py
│   ├── data.py
│   ├── metrics.py
│   └── models.py
│
├── model.py  # The code provided
└── README.md
```

---

## 🧠 Notes

- Embedding and "other" category handling is crucial for managing high-cardinality categorical data.
- Holiday proximity can be a significant temporal feature in appointment likelihood.
- Designed to scale and parallelize with `RandomForestClassifier(n_jobs=-1)`.

---

## ✅ To Run the Pipeline

```python
pred = pred_pipe()
pred.train_test_split()
pred.train()
pred.evaluate_model()
pred.evaluate_model_metrics()
```
