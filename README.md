# Customer Churn Prediction

This project focuses on building and optimizing a machine learning model to predict customer churn. The core algorithm used is **XGBoost** (Gradient Boosting), fine-tuned using hyperparameter optimization and classification threshold tuning to effectively handle imbalanced data.

## 🚀 Key Features of the Solution
* **Class Imbalance Handling**: Utilized the `scale_pos_weight` parameter to balance the loss function based on the class ratio (~2.77).
* **Hyperparameter Tuning**: Automated search for optimal tree structures and learning rates using `RandomizedSearchCV`.
* **Threshold Optimization (Youden's J Statistic)**: Shifted the default `0.5` classification threshold using the ROC curve to maximize the `Recall` score for the minority class (churned customers).

## 📊 Model Evaluation Results

The model demonstrates strong discriminative power (high AUC-ROC). By shifting the classification threshold, false negatives (missing customers who are about to churn) were significantly reduced.

### Classification Report
```text
              precision    recall  f1-score   support

           0       0.91      0.76      0.83      1036
           1       0.54      0.80      0.65       373

    accuracy                           0.77      1409
   macro avg       0.73      0.78      0.74      1409
weighted avg       0.81      0.77      0.78      1409
```

### Confusion Matrix Breakdown
* **True Negatives (Stayed, Predicted Stayed)**: 788
* **False Positives (False Alarms)**: 248
* **False Negatives (Missed Churn)**: 76 — *Successfully minimized using the customized threshold!*
* **True Positives (Churn, Predicted Churn)**: 297

## 💾 Model Serialization and Inference

The final trained model and the mathematically derived optimal threshold are saved together in a single file `CustomerChurnXGBoost.pkl` for production deployment.



## 🛠 Tech Stack
* Python 3
* XGBoost
* Scikit-learn
* Pandas & NumPy
* Matplotlib & Seaborn
