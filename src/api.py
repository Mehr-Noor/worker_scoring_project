# فایل: src/api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import pandas as pd
import joblib

from src.data_processing import preprocess_data, generate_synthetic_data
from src.scoring import decision_logic
from src.fairness import demographic_parity, disparate_impact

app = FastAPI(
    title="Worker Scoring API",
    description="ML system for scoring hospital candidates with fairness evaluation",
    version="1.0.0"
)

# -------------------------------
# Request models
# -------------------------------
class Candidate(BaseModel):
    features: Dict[str, float]

class BatchCandidates(BaseModel):
    candidates: List[Candidate]

# -------------------------------
# Load models at startup
# -------------------------------
lr_pipeline = joblib.load("src/logistic_model.pkl")
xgb_pipeline = joblib.load("src/xgb_model.pkl")

# -------------------------------
# Prediction endpoints
# -------------------------------
@app.post("/predict/candidate")
def predict_candidate(candidate: Candidate):
    df = pd.DataFrame([candidate.features])
    lr_model = lr_pipeline["model"]
    threshold = lr_pipeline["threshold"]
    prob = lr_model.predict_proba(df)[:, 1][0]
    decision = "Pass" if prob >= threshold else "Fail"
    return {"probability": prob, "decision": decision}

@app.post("/predict/batch")
def predict_batch(batch: BatchCandidates):
    df = pd.DataFrame([c.features for c in batch.candidates])
    lr_model = lr_pipeline["model"]
    threshold = lr_pipeline["threshold"]
    probs = lr_model.predict_proba(df)[:, 1]
    decisions = ["Pass" if p >= threshold else "Fail" for p in probs]
    return {"probabilities": probs.tolist(), "decisions": decisions}

# -------------------------------
# Fairness endpoint
# -------------------------------
@app.get("/fairness")
def get_fairness():
    """
    Compute Demographic Parity & Disparate Impact on synthetic data.
    """
    try:
        # Generate or load dataset
        data = generate_synthetic_data()
        # Fairness metrics
        dp_gender = demographic_parity(data, "gender")
        di_gender = disparate_impact(data, "gender", reference_group="Male")
        dp_region = demographic_parity(data, "region")
        return {
            "demographic_parity": {
                "gender": dp_gender,
                "region": dp_region
            },
            "disparate_impact": {
                "gender": di_gender
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
