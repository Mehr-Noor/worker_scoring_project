# فایل: src/model.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from xgboost import XGBClassifier

from src.data_processing import generate_synthetic_data, preprocess_data
from src.scoring import decision_logic
from src.fairness import demographic_parity, disparate_impact


# --------------------------------------------------
# Config
# --------------------------------------------------
DECISION_THRESHOLD = 0.4
RANDOM_STATE = 42


# --------------------------------------------------
# Train & Evaluate
# --------------------------------------------------
def train_models():

    # 1️⃣ Generate synthetic data
    raw_data = generate_synthetic_data()
    raw_data.to_csv("data/synthetic_data.csv", index=False)
    print("Data saved to data/synthetic_data.csv")

    # 2️⃣ Preprocess
    df = preprocess_data(raw_data)

    X = df.drop(columns=["score", "decision"])
    y = df["decision"].map({"Fail": 0, "Pass": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ==================================================
    # Logistic Regression (interpretable baseline)
    # ==================================================
    lr_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )

    lr_model.fit(X_train, y_train)

    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs >= DECISION_THRESHOLD).astype(int)

    print("\n================ Logistic Regression ================")
    print(f"Decision threshold: {DECISION_THRESHOLD}")
    print("Accuracy:", accuracy_score(y_test, lr_preds))
    print("F1 Score:", f1_score(y_test, lr_preds))
    print("ROC AUC:", roc_auc_score(y_test, lr_probs))
    print("Confusion Matrix:\n", confusion_matrix(y_test, lr_preds))

    # Save model + threshold
    joblib.dump(
        {
            "model": lr_model,
            "threshold": DECISION_THRESHOLD
        },
        "src/logistic_model.pkl"
    )

    # ==================================================
    # XGBoost (performance-oriented)
    # ==================================================
    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=RANDOM_STATE
    )

    xgb_model.fit(X_train, y_train)

    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = (xgb_probs >= DECISION_THRESHOLD).astype(int)

    print("\n================ XGBoost ============================")
    print(f"Decision threshold: {DECISION_THRESHOLD}")
    print("Accuracy:", accuracy_score(y_test, xgb_preds))
    print("F1 Score:", f1_score(y_test, xgb_preds))
    print("ROC AUC:", roc_auc_score(y_test, xgb_probs))
    print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_preds))

    joblib.dump(
        {
            "model": xgb_model,
            "threshold": DECISION_THRESHOLD
        },
        "src/xgb_model.pkl"
    )

    # ==================================================
    # Fairness & Bias Evaluation (on raw data)
    # ==================================================
    print("\n================ FAIRNESS CHECK ====================")

    fairness_gender = demographic_parity(raw_data, "gender")
    print("Demographic Parity (Gender):", fairness_gender)

    di_gender = disparate_impact(
        raw_data,
        group_col="gender",
        reference_group="Male"
    )
    print("Disparate Impact (Gender):", di_gender)

    fairness_region = demographic_parity(raw_data, "region")
    print("Demographic Parity (Region):", fairness_region)

    print("\n✔ Training, evaluation, and fairness analysis completed.")

    return lr_model, xgb_model


# --------------------------------------------------
# Entry point
# --------------------------------------------------
if __name__ == "__main__":
    train_models()
