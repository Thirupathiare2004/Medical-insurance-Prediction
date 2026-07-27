import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib, os, json

import os
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR   = os.path.join(BASE_DIR, 'datasets')
os.makedirs(MODELS_DIR, exist_ok=True)

def encode_and_save(df, cat_cols, target, name):
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))
        encoders[c] = {int(i): str(v) for i, v in enumerate(le.classes_)}
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    print(f"[{name}] MAE={mae:.0f}  RMSE={rmse:.0f}  R²={r2:.4f}")
    joblib.dump(model, f'{MODELS_DIR}/{name}_model.pkl')
    joblib.dump(list(X.columns), f'{MODELS_DIR}/{name}_features.pkl')
    with open(f'{MODELS_DIR}/{name}_encoders.json', 'w') as f:
        json.dump(encoders, f)
    with open(f'{MODELS_DIR}/{name}_metrics.json', 'w') as f:
        json.dump({'mae': round(mae,2), 'rmse': round(rmse,2), 'r2': round(r2,4)}, f)
    print(f"  → Saved {name}_model.pkl")

# ── MEDICAL ───────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{DATA_DIR}/medical_insurance.csv')
cat = ['sex','smoker','alcohol','exercise','diabetes','bp','cholesterol',
       'heart_disease','stress_level','occupation','region','pollution_level']
encode_and_save(df, cat, 'charges', 'medical')

# ── CAR ───────────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{DATA_DIR}/car_insurance.csv')
cat = ['gender','vehicle_type','city_type','anti_theft','multi_car']
encode_and_save(df, cat, 'premium', 'car')

# ── HOME ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{DATA_DIR}/home_insurance.csv')
cat = ['location_risk','construction_type','security_system',
       'smoke_detectors','flood_zone','garage']
encode_and_save(df, cat, 'premium', 'home')

# ── LIFE ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{DATA_DIR}/life_insurance.csv')
cat = ['gender','smoker','occupation_risk','family_history','chronic_illness','alcohol','exercise_freq']
encode_and_save(df, cat, 'premium', 'life')

# ── TRAVEL ────────────────────────────────────────────────────────────────────
df = pd.read_csv(f'{DATA_DIR}/travel_insurance.csv')
cat = ['destination_risk','travel_purpose','pre_existing_condition',
       'coverage_type','flight_delay','adventure_sports']
encode_and_save(df, cat, 'premium', 'travel')

print("\n✅  All 5 models trained and saved!")