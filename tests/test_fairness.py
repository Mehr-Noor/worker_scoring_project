# فایل: tests/test_fairness.py

import pytest
import pandas as pd
from src.fairness import demographic_parity, disparate_impact

@pytest.fixture
def sample_data():
    # داده نمونه کوچک برای تست fairness
    return pd.DataFrame([
        {"gender": "Male", "region": "North", "decision": "Pass"},
        {"gender": "Female", "region": "North", "decision": "Fail"},
        {"gender": "Male", "region": "South", "decision": "Pass"},
        {"gender": "Female", "region": "South", "decision": "Pass"},
        {"gender": "Female", "region": "North", "decision": "Pass"},
    ])

def test_demographic_parity(sample_data):
    dp_gender = demographic_parity(sample_data, "gender")
    dp_region = demographic_parity(sample_data, "region")

    assert isinstance(dp_gender, dict)
    assert isinstance(dp_region, dict)
    assert "Male" in dp_gender
    assert "Female" in dp_gender

def test_disparate_impact(sample_data):
    di_gender = disparate_impact(sample_data, "gender", reference_group="Male")
    assert isinstance(di_gender, dict)
    # Male/reference should always be 1
    assert di_gender["Male"] == 1.0
