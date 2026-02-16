import pytest
import pandas as pd
from src.model import train_models
from src.data_processing import generate_synthetic_data

def test_model_training_shapes():
    lr_model, xgb_model = train_models()
    assert lr_model is not None
    assert xgb_model is not None

def test_synthetic_data():
    data = generate_synthetic_data()
    assert isinstance(data, pd.DataFrame)
    # حداقل یک نمونه Pass و Fail
    assert "Pass" in data["decision"].values
    assert "Fail" in data["decision"].values
