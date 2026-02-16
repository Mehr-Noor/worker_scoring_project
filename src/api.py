# فایل: src/api.py

from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from io import BytesIO

from src.data_processing import preprocess_data
from src.scoring import decision_logic

app = FastAPI(
    title="Worker Scoring & Decision API",
    description="ML-based scoring and decision system for hospital workforce staffing",
    version="1.0"
)

# --------------------------------------------------
# Load trained models
# --------------------------------------------------
lr_bundle = joblib.load("src/logistic_model.pkl")
lr_model = lr_bundle["model"]
decision_threshold = lr_bundle["threshold"]


# --------------------------------------------------
# Pydantic input schema (IMPORTANT)
# --------------------------------------------------
class CandidateInput(BaseModel):
    age: int = Field(..., example=30)
    gender: str = Field(..., example="Female")
    icu_experience: int = Field(..., example=3)
    cpr_skill: int = Field(..., example=4)
    language_level: str = Field(..., example="B1")
    previous_score: int = Field(..., example=85)
    region: str = Field(..., example="North")


# --------------------------------------------------
# Root
# --------------------------------------------------
@app.get("/")
def root():
    return {"message": "Worker Scoring API is running"}


# --------------------------------------------------
# Real-time inference
# --------------------------------------------------
@app.post("/predict")
def predict_candidate(candidate: CandidateInput):
    try:
        df = pd.DataFrame([candidate.dict()])
        df_processed = preprocess_data(df)

        prob = lr_model.predict_proba(df_processed.values)[:, 1][0]

        decision, risk = decision_logic(prob)

        return {
            "score": round(float(prob), 3),
            "decision": decision,
            "risk_level": risk
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# --------------------------------------------------
# Batch inference
# --------------------------------------------------
@app.post("/predict_batch")
def predict_batch(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        df = pd.read_csv(BytesIO(contents))

        df_processed = preprocess_data(df)
        probs = lr_model.predict_proba(df_processed.values)[:, 1]

        decisions = [decision_logic(p) for p in probs]

        df["score"] = probs
        df["decision"] = [d[0] for d in decisions]
        df["risk_level"] = [d[1] for d in decisions]

        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
