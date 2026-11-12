*Read this in other languages: [Español](README_es.md)*

# Hotel Booking Demand: Multiclass Market Segment Classification

## 1. Project Overview

This repository contains an end-to-end Machine Learning pipeline that predicts the **market_segment** of a hotel booking **at the moment the booking is made**. Using only information available at booking time (stay details, party composition, lead time, rate, room and deposit type, guest history), the model classifies each reservation into one of seven market segments.

Accurate segment classification helps hospitality businesses audit segment labels, fill in missing ones, tailor pricing and marketing, and forecast demand by segment.

---

## 2. Business Context

In the hospitality industry, understanding the origin of each booking is crucial. Market segments differ in cancellation behavior, lead time, length of stay and price sensitivity, so revenue management decisions depend on knowing which segment a booking belongs to.

---

## 3. Dataset Information

The project uses the public **Hotel Booking Demand** dataset (Antonio, Almeida & Nunes, 2019), with bookings from a City Hotel in Lisbon and a Resort Hotel in the Algarve, arriving between 2015 and 2017.

### Target Variable

**market_segment** (7 classes, strongly imbalanced):

| Segment | Share |
|---------|------:|
| Online TA | 59.1% |
| Offline TA/TO | 15.9% |
| Direct | 13.5% |
| Groups | 5.7% |
| Corporate | 4.8% |
| Complementary | 0.8% |
| Aviation | 0.26% |

### Feature Scope

The model is designed for **booking-time prediction**, so two groups of columns are deliberately excluded:

| Excluded columns | Reason |
|------------------|--------|
| `is_canceled`, `reservation_status`, `reservation_status_date`, `assigned_room_type`, `booking_changes` | Only known after the booking is made (future information) |
| `distribution_channel`, `agent`, `company` | Channel identifiers that encode the segment almost one-to-one |

---

## 4. Methodology and Architecture

The workflow emphasizes leakage prevention, honest evaluation and reproducibility.

### 4.1 Data Cleaning

- Removal of 31,994 exact duplicate records before any split.
- Removal of classes with fewer than 10 records (`Undefined`) to allow stratified splitting.
- Final dataset: 87,394 bookings.

### 4.2 Evaluation Strategy

Stratified split into **Train (64%)**, **Validation (16%)** and **Test (20%)**. Models are compared and tuned on Validation only; Test is used once, for the final evaluation.

### 4.3 Preprocessing and Feature Engineering

Custom Scikit-Learn transformers integrated into a single pipeline:

- **FeatureEngineer:** total nights, total guests and a kids flag; cyclical (sine/cosine) encoding of arrival month and arrival week.
- **IQRTransformer:** clips outliers to `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`, learned on training data only. Columns with zero IQR (binary flags, sparse counts) are left untouched.
- **RareCategoryEncoder:** groups categories below 1% frequency into `"Other"` before one-hot encoding.

Numerical and categorical columns are detected automatically with `make_column_selector`.

### 4.4 Class Imbalance

**Moderate SMOTE oversampling** inside an imbalanced-learn `Pipeline`: every minority class is raised to at least 15% of the majority class. Resampling happens only during `fit()`, so validation and test always keep the real class distribution.

### 4.5 Model Benchmarking

Five algorithms are trained with the same pipeline and ranked on validation by **macro F1-Score** (every segment weighs the same); weighted F1 is reported alongside:

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM

### 4.6 Hyperparameter Optimization

The best model is selected automatically and tuned with **Optuna** (TPE sampler, 25 trials, macro F1 on validation). Tuned parameters are adopted only if they beat the benchmark configuration. The final model is retrained on Train + Validation before the test evaluation.

---

## 5. Results

### Validation benchmark

| Model | Macro F1 | Weighted F1 |
|-------|---------:|------------:|
| **Random Forest** | **0.754** | **0.805** |
| XGBoost | 0.747 | 0.803 |
| LightGBM | 0.744 | 0.795 |
| Gradient Boosting | 0.648 | 0.735 |
| Logistic Regression | 0.505 | 0.661 |

Optuna raised the Random Forest validation macro F1 from 0.754 to 0.759.

### Final test set (17,479 unseen bookings)

| Segment | Precision | Recall | F1 |
|---------|----------:|-------:|---:|
| Online TA | 0.84 | 0.94 | 0.89 |
| Groups | 0.87 | 0.91 | 0.89 |
| Complementary | 0.80 | 0.90 | 0.85 |
| Offline TA/TO | 0.85 | 0.72 | 0.78 |
| Corporate | 0.75 | 0.76 | 0.75 |
| Aviation | 0.68 | 0.71 | 0.70 |
| Direct | 0.68 | 0.43 | 0.53 |
| **Macro avg** | **0.78** | **0.77** | **0.77** |
| **Weighted avg** | **0.82** | **0.83** | **0.82** |

Accuracy: 0.83. One-vs-rest ROC-AUC ranges from 0.881 (Direct) to 0.999 (Complementary).

**Key findings**

- Segments with a distinctive booking profile (Groups, Complementary, Aviation) are recognized well even with few examples.
- The main challenge is **Direct vs. Online TA**: almost half of the direct bookings are predicted as online agency bookings. Without channel information, both look like the same leisure guest, which makes this the natural limit of booking-time prediction.

---

## 6. Project Structure

```
├── data/
│   └── hotel_bookings.csv
├── notebooks/
│   └── Hotel Booking Demand.ipynb
├── requirements.txt
├── README.md
└── README_es.md
```

---

## 7. Technical Stack

| Category | Technologies |
|-----------|-------------|
| Language | Python 3.12 |
| Data Manipulation & Analysis | Pandas, NumPy |
| Machine Learning Framework | Scikit-Learn |
| Imbalanced Data | imbalanced-learn (SMOTE) |
| Advanced Ensembles | XGBoost, LightGBM |
| Hyperparameter Tuning | Optuna |
| Data Visualization | Matplotlib, Seaborn |

---

## 8. Execution Guide

1. Clone the repository and install the dependencies:

```bash
git clone https://github.com/arguar13/Hotel-Booking-Segment-Classification.git
cd Hotel-Booking-Segment-Classification
pip install -r requirements.txt
```

2. Open `notebooks/Hotel Booking Demand.ipynb` and run all cells. The notebook finds `data/hotel_bookings.csv` whether it is launched from the project root or from `notebooks/`.

A full run (benchmark + 25 Optuna trials) takes roughly 30 minutes on a standard laptop.

---

## License

Educational and portfolio Data Science project. Dataset obtained from Kaggle and used for non-commercial purposes.

---

## Author

**Armando Guarnera**  
Data Scientist  
Argentina
