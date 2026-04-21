# Smart Parking Availability Prediction using Machine Learning

## 1. Overview

This project predicts the number of available parking spots for a given street segment at a specific time using machine learning.

The system combines:

* Temporal patterns (hour, day)
* Spatial behavior (segment-specific trends)
* Recent historical data (lag features)

Final performance:

* **MAE ≈ 1.17 (≈ 1 parking spot error)**

---

## 2. Problem Statement

Urban parking is inefficient due to:

* Lack of real-time availability awareness
* High search time for drivers
* Congestion caused by parking search

Goal:

> Predict parking availability given location and time.

---

## 3. Dataset

Dataset: SFpark parking dataset

Characteristics:

* ~5 million rows
* Time interval: ~5 minutes
* Duration: ~40 days
* Features:

  * timestamp
  * segmentid (street segment)
  * capacity
  * occupied

Derived:

* availability = capacity − occupied

---

## 4. Key Observations (EDA)

### 4.1 Temporal Patterns

* Peak occupancy: **7–8 PM**
* Lowest occupancy: **3–4 AM**
* Strong daily cycles

### 4.2 Weekly Patterns

* Highest demand: **Friday**
* Lowest demand: **Sunday**

### 4.3 Segment Behavior

* Large variation across segments
* Some always near full, some rarely used
* Distribution roughly normal (balanced dataset)

---

## 5. Feature Engineering

### 5.1 Basic Features

* hour
* dayofweek
* capacity

### 5.2 Target

* availability = capacity − occupied

---

### 5.3 Temporal Features (Most Important)

#### Lag Features

* lag_1 → availability at previous timestep
* lag_2 → availability at t−2

#### Rolling Feature

* rolling_mean_3 → average of last 3 values

Reason:

> Parking availability is highly dependent on recent history.

---

## 6. Modeling Approach

### Model Used:

* Random Forest Regressor

### Why Random Forest?

* Handles non-linear relationships
* Captures feature interactions
* Works well without heavy preprocessing
* Suitable for tabular + time-derived features

---

## 7. Data Splitting

Used **time-based split (80/20)**:

* Train → past data
* Test → future data

Reason:

> Prevent data leakage and simulate real-world prediction

---

## 8. Model Performance

| Stage             | MAE      |
| ----------------- | -------- |
| Baseline (no lag) | 2.24     |
| + Lag features    | 1.40     |
| + Rolling mean    | 1.37     |
| + Model tuning    | **1.17** |

Improvement:

* ~48% reduction in error

---

## 9. Feature Importance

Top contributors:

* lag_1, lag_2, rolling_mean_3 (~70%)
* hour, capacity (~30%)

Low impact:

* dayofweek
* segmentid (raw)

Insight:

> Recent history dominates prediction

---

## 10. System Design

### Challenge

User cannot provide lag values manually.

### Solution

System automatically:

* Fetches historical data
* Computes lag features internally

---

### Prediction Flow

User Input:

* segmentid
* hour
* day

System:

* retrieves relevant past data
* computes lag features
* predicts availability

---

## 11. CLI Tool

Two modes:

### Simple Mode (recommended)

* minimal input
* system computes everything

### Advanced Mode

* manual control over lag inputs

---

## 12. Limitations

* No real-time data (uses historical proxy)
* Dataset limited to ~40 days
* No seasonal (yearly) modeling
* Assumes stable patterns

---

## 13. Future Improvements

* Real-time data integration
* API / web interface
* Use gradient boosting (XGBoost, LightGBM)
* Segment embeddings or clustering
* Time-series models (LSTM, Prophet)

---

## 14. Key Learnings

* Temporal dependency is critical in time-series problems
* Feature engineering > model complexity
* Data representation matters (IDs vs behavior)
* System design must align with real-world usage

---

## 15. Conclusion

This project demonstrates a complete ML pipeline:

* Data analysis
* Feature engineering
* Model training
* Evaluation
* Deployment (CLI tool)

Final system:

> Predicts parking availability with high accuracy (~1 spot error)

---

## 16. How to Run

### Train model

```
python train_model.py
```

### Run prediction CLI

```
python predict.py
```
