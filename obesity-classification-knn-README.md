# Obesity Classification with K-Nearest Neighbors

## Problem

Given a person's age, gender, height, weight, and BMI, predict their obesity category: **Underweight**, **Normal Weight**, **Overweight**, or **Obese**.

This is a multi-class classification problem. Real-world applications include early screening tools in healthcare where quick, data-driven assessments could flag patients for follow-up.

---

## Dataset

- **Source:** [Obesity Classification Dataset](https://www.kaggle.com/datasets/sujithmandala/obesity-classification-dataset) — Kaggle
- **Size:** 1,000 records, 6 features
- **Target column:** `Label` (4 classes)

| Feature | Type | Description |
|--------|------|-------------|
| Age | Numeric | Age in years |
| Gender | Categorical | Male / Female |
| Height | Numeric | Height in cm |
| Weight | Numeric | Weight in kg |
| BMI | Numeric | Body Mass Index |

---

## Approach

### 1. Exploratory Data Analysis
- Inspected data types, shape, and class distribution
- Confirmed no missing values or duplicate rows
- Plotted boxplots for all numeric features to detect outliers

### 2. Data Cleaning
- Detected outliers in `Age` using the IQR method and clipped them to the boundary values
- Applied `pd.get_dummies()` with `drop_first=True` to encode the `Gender` column

### 3. Train/Test Split
- Split data 80/20 using `train_test_split` with `random_state=42` for reproducibility
- Split was performed **before** scaling to prevent data leakage

### 4. Feature Scaling
- Applied `StandardScaler` — critical for KNN because the algorithm measures distance between data points
- Used `fit_transform` on training data only, and `transform` (no refit) on test data to avoid leaking test statistics into the model

### 5. Finding the Best K
- Iterated K from 1 to 20, recording accuracy and error rate for each
- Plotted accuracy vs K and error rate vs K to visually identify the optimal value
- Selected **K = 7** as the best performing value

### 6. Evaluation
- Retrained the final model using K = 7
- Evaluated using accuracy score and a confusion matrix heatmap

---

## Results

| Metric | Value |
|--------|-------|
| Best K | 7 |
| Test Accuracy | 63.64% |

### Confusion Matrix Observations

- **Normal Weight** was predicted most reliably, though it was occasionally confused with Underweight
- **Obese** was frequently misclassified as Overweight — the model struggled with the boundary between these two classes
- **Overweight** was most often confused with Normal Weight, and occasionally with Underweight
- **Underweight** was only confused with Normal Weight

The confusion pattern makes intuitive sense — obesity categories are defined by numerical BMI ranges with no natural gap between them. A person at BMI 29.9 (Overweight) and BMI 30.1 (Obese) are nearly identical, making the boundary hard for any model to learn cleanly.

---

## Why 63.64%? — Honest Reflection

The modest accuracy is expected given the following:

1. **BMI is derived from Height and Weight** — all three were included as features, introducing redundant information. In a future iteration I would drop either BMI or the Height/Weight pair to reduce noise.
2. **Small dataset** — 1,000 records is relatively small for a 4-class problem. KNN is particularly sensitive to this because it relies on having enough nearby neighbours to vote reliably.
3. **Natural class overlap** — obesity categories are based on fixed BMI cut-offs. Points near the boundaries will always be ambiguous regardless of the algorithm used.

---

## What I Would Improve

- Drop `BMI` and keep only `Height` and `Weight` to eliminate feature redundancy
- Try a larger K range (up to 50) to see if the elbow point shifts
- Experiment with weighted KNN (`weights='distance'`) so closer neighbours have more influence
- Test other classification algorithms (Random Forest, Logistic Regression) and compare results
- Use cross-validation instead of a single train/test split for a more reliable accuracy estimate

---

## Tools & Libraries

`Python` · `pandas` · `scikit-learn` · `matplotlib` · `seaborn`

---

## Files

| File | Description |
|------|-------------|
| `obesity-classification.ipynb` | Full notebook — EDA, preprocessing, modelling, evaluation |

---

## How to Run

1. Open the notebook on [Kaggle](https://www.kaggle.com) — dataset is pre-attached
2. Run all cells in order
3. The final cell outputs the confusion matrix for the best K model

---

*Part of my NQF7 Machine Learning coursework portfolio. Built while learning classification algorithms and best practices in ML pipelines.*
