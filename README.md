# 💳 Credit Card Fraud Detection

## Objective
Detect fraudulent credit card transactions using machine learning on a highly imbalanced dataset (~0.17% fraud).

## Dataset
- **Source:** Kaggle – Credit Card Fraud Detection by ULB ML Group  
- **Size:** 284,807 transactions (492 fraud)
- **Features:** V1–V28 (PCA), Time, Amount, Class (0 = No fraud, 1 = Fraud)

## Workflow

1. Standardized the `Amount` feature and removed `Time`.
2. Visualized class imbalance; applied **SMOTE** to training set.
3. Trained three models:
   - Logistic Regression (balanced class weights)
   - Random Forest (balanced)
   - XGBoost (weight tuned)
4. Evaluated on test set using:
   - Precision / Recall / F1-score
   - Confusion Matrix
   - ROC-AUC

## Evaluation Results

| Model              | Precision (fraud) | Recall (fraud) | F1-score | ROC-AUC |
|--------------------|-------------------|----------------|----------|---------|
| Logistic Regression| 0.81              | 0.74           | 0.77     | 0.95    |
| Random Forest      | 0.84              | 0.79           | 0.81     | 0.98    |
| XGBoost            | 0.88              | 0.83           | 0.85     | 0.99    |

## Visuals
- `ccf_class_dist.png` – Original distribution  
- `ccf_smote_dist.png` – After SMOTE  
- Confusion Matrices and ROC Curves for each model (`*_cm.png`, `*_roc.png`)

## Output Files
- `credit_card_fraud_detection.ipynb` – Full notebook  
- `best_ccf_model.pkl` – Serialized best model (XGBoost)  
- `README.md` – This documentation  
- `.png` graphics – for demo/report purposes

## Summary
With SMOTE and weighted modelling, the XGBoost model achieved an ROC-AUC of 0.99 and strong F1-scores, showing successful handling of extreme imbalance in fraud detection.