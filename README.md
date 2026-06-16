# laughing-octo-chainsaw
just doing a personal tasks

# Customer Churn Prediction Framework

An end-to-end machine learning pipeline built to predict customer churn, optimize retention strategies, and evaluate the operational trade-offs of ensemble modeling versus standalone architectures.

## 1. Business Problem & Objective
Customer churn directly impacts revenue and increases customer acquisition costs. This project develops a predictive framework to flag high-risk customers using historical data, enabling proactive marketing interventions. The target variable is binary: `1` for Churn, `0` for Stay.

## 2. Project Architecture & Directory Structure
This repository follows an industry-standard production layout for clean data separation and model reproducibility:

```text
laughing-octo-chainsaw/
├── data/
│   ├── processed/              # Cleaned, engineered, and scaled data arrays
│   └── raw/                    # Original, untouched source data files
├── models/                     # Saved binary deployment assets (Pickle files)
│   ├── baseline_logistic_regression.pkl
│   ├── ensemble_voting.pkl
│   ├── scaler.pkl
│   ├── tuned_lightgbm.pkl
│   └── tuned_xgboost.pkl
├── notebook/                   # Sequentially ordered development scripts
│   ├── 1.0_data_exploration_and_prep.ipynb
│   └── 2.0_model_training_and_ensemble.ipynb
├── .gitignore                  # Prevents committing heavy data/local configs
├── README.md                   # Technical documentation and executive summary
└── requirements.txt            # Environment dependencies for reproducibility
```

## 3. Data Pipeline & Engineering Wins
* **Feature Processing:** Handled dirty data inputs, applied standard scaling via Scikit-Learn pipelines, and preserved explicit feature mapping for complete transparency.
* **Data Leakage Mitigation:** Enforced strict separation between training and evaluation phases by executing scaling transformations cleanly across data splits.
* **Dimensionality:** Optimized modeling around 10 high-impact predictive features.

## 4. Model Performance & Evaluation
We trained and hyperparameter-tuned a collection of diverse architectures, ultimately evaluating a Soft Voting Ensemble to weigh combined probabilities.

| Model | Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline Logistic Regression** | 0.921 | 0.95 | 0.82 | 0.88 |
| **Tuned XGBoost** | 0.919 | 0.95 | 0.81 | 0.88 |
| **Tuned LightGBM (Champion)** | **0.921** | **0.95** | **0.82** | **0.88** |
| **Soft Voting Ensemble** | 0.921 | 0.95 | 0.82 | 0.88 |

### Champion Model Performance Breakdown
The selected **LightGBM** model achieved a tight performance profile on the holdout test set:
* **True Negatives (Correctly predicted to stay):** 1,167 customers
* **True Positives (Correctly flagged as churning):** 548 customers
* **False Negatives (Missed churn risks):** 120 customers
* **False Positives (False alarms):** 27 customers

> 💡 **Production Engineering Decision:** 
> While the Soft Voting Ensemble was fully evaluated, empirical testing revealed **0% performance lift** over the standalone **LightGBM** model. Both achieved an identical accuracy of `0.921` and shared the exact same confusion matrix. 
> 
> To optimize for production latency, minimize server memory allocation, and keep maintenance overhead low, the **standalone LightGBM model was selected as the production champion.**

## 5. How to Reproduce This Project
1. Clone the repository:
   ```bash
   git clone https://github.com/claziq/laughing-octo-chainsaw
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
