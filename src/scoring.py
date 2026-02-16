# فایل: src/scoring.py
import joblib
import pandas as pd

def load_models():
    lr_model = joblib.load("src/logistic_model.pkl")
    xgb_model = joblib.load("src/xgb_model.pkl")
    return lr_model, xgb_model

def compute_score(model, X):
    """
    محاسبه score پیش‌بینی شده توسط مدل
    """
    proba = model.predict_proba(X)[:,1]  # احتمال Pass
    return proba

def decision_logic(score):
    """
    تبدیل score به decision:
    - <0.4: Fail / High Risk
    - 0.4-0.7: Medium Risk
    - >0.7: Pass / Low Risk
    """
    if score < 0.4:
        return "Fail", "High Risk"
    elif score < 0.7:
        return "Medium Risk", "Medium Risk"
    else:
        return "Pass", "Low Risk"

if __name__ == "__main__":
    from src.data_processing import preprocess_data, generate_synthetic_data
    data = generate_synthetic_data(5)
    df = preprocess_data(data)
    lr_model, xgb_model = load_models()
    
    for i, row in df.iterrows():
        score = compute_score(lr_model, row.values.reshape(1, -1))[0]
        dec, risk = decision_logic(score)
        print(f"Candidate {i}: Score={score:.2f}, Decision={dec}, Risk={risk}")
