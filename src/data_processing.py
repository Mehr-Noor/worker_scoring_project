# فایل: src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(42)

def generate_synthetic_data(n_candidates=500):
    """
    تولید داده‌های مصنوعی برای نیروی کار بیمارستان‌ها
    """
    data = pd.DataFrame({
        "age": np.random.randint(22, 60, size=n_candidates),
        "gender": np.random.choice(["Male", "Female"], size=n_candidates),
        "icu_experience": np.random.randint(0, 10, size=n_candidates),
        "cpr_skill": np.random.randint(0, 5, size=n_candidates),
        "language_level": np.random.choice(["A1", "B1", "C1"], size=n_candidates),
        "previous_score": np.random.randint(50, 100, size=n_candidates),
        "region": np.random.choice(["North", "South", "East", "West"], size=n_candidates)
    })

    # امتیاز مصنوعی (score)
    data["score"] = (
        0.3*data["icu_experience"] +
        0.2*data["cpr_skill"] +
        0.3*data["previous_score"]/20 +
        np.random.normal(0, 2, n_candidates)
    )
    
    threshold = data["score"].quantile(0.6)  # 40٪ Pass
    # تصمیم Pass/Fail بر اساس score
    data["decision"] = np.where(data["score"] >= threshold, "Pass", "Fail")
    
    # ذخیره CSV
    data.to_csv("data/synthetic_data.csv", index=False)
    print("Data saved to data/synthetic_data.csv")
    
    return data

def preprocess_data(df):
    """
    پردازش اولیه داده‌ها:
    - تبدیل categorical به numeric
    - scaling برای مدل
    """
    df_processed = df.copy()
    
    # One-hot encoding برای gender, language_level, region
    categorical_cols = ["gender", "language_level", "region"]
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df_processed[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))
    
    df_processed = pd.concat([df_processed.drop(columns=categorical_cols), encoded_df], axis=1)
    
    # Scaling ستون‌های عددی
    scaler = StandardScaler()
    numeric_cols = ["age", "icu_experience", "cpr_skill", "previous_score", "score"]
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed

# اجرا برای تست
if __name__ == "__main__":
    data = generate_synthetic_data()
    processed_data = preprocess_data(data)
    print(processed_data.head())
