# KNN Heart Disease Prediction

Lucas Kiukawa · Eduexplora Stanford Artificial Intelligence program · Brazil, January 2020

A k-nearest neighbors (KNN) classifier that predicts whether a patient has cardiovascular disease from routine clinical measurements: age, weight, blood pressure, cholesterol, glucose, smoking and physical activity.

## Data

The [Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) on Kaggle: 70,000 patients, each with 11 features and a binary target, `cardio` (1 = has cardiovascular disease). Download `cardio_train.csv` and put it next to the script. The data is not included in this repository.

The model uses eight of the features:

| Column | Meaning |
| --- | --- |
| `age` | Age, in days |
| `weight` | Weight, in kg |
| `ap_hi` | Systolic blood pressure |
| `ap_lo` | Diastolic blood pressure |
| `cholesterol` | 1 normal, 2 above normal, 3 well above normal |
| `gluc` | Glucose: 1 normal, 2 above normal, 3 well above normal |
| `smoke` | Smokes (0 or 1) |
| `active` | Physically active (0 or 1) |

## Method

1. Split the patients 80/20 into training and test sets (stratified, fixed random seed).
2. Min-max scale every feature to 0-1, fit on the training set only, so no single measurement dominates the distance between patients.
3. Train a baseline KNN with k = 5 and report its accuracy and recall.
4. Sweep every odd k from 3 to 499 (odd values avoid tied votes) and plot test accuracy against k, saved as `accuracy_vs_k.png`.
5. Choose k with a grid search over odd k from 3 to 99, using 5-fold cross-validation scored on recall: for a screening tool, missing a patient who has the disease costs more than a false alarm.

## Run

```bash
pip install -r requirements.txt
python knn_heart.py                      # reads cardio_train.csv in this folder
python knn_heart.py path/to/cardio_train.csv
```

The k sweep trains 249 models, so the full run takes a few minutes.

## Changes from the original Colab notebook

The project started as a Google Colab notebook. To run it as a script:

- Added the missing `accuracy_score` import.
- Passed `recall_score` its arguments in the right order (`y_true`, then `y_pred`).
- Ran the grid search on the training set, with scaling inside a pipeline, so k is chosen on scaled data and the test set stays unseen.
- Printed the results and saved the plot to a file instead of relying on notebook output.
