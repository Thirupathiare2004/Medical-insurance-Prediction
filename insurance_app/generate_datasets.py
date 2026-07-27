import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'datasets')
os.makedirs(DATA_DIR, exist_ok=True)

np.random.seed(42)
N = 5000

# ── 1. CAR INSURANCE ──────────────────────────────────────────────────────────
age = np.random.randint(18, 75, N)
gender = np.random.choice(['male', 'female'], N)
vehicle_age = np.random.randint(0, 20, N)
vehicle_type = np.random.choice(['sedan', 'suv', 'hatchback', 'truck', 'sports'], N)
annual_mileage = np.random.randint(5000, 50000, N)
accidents_history = np.random.randint(0, 5, N)
traffic_violations = np.random.randint(0, 8, N)
license_years = np.clip(age - 18, 0, 50)
city_type = np.random.choice(['urban', 'suburban', 'rural'], N)
credit_score = np.random.randint(300, 850, N)
anti_theft = np.random.choice(['yes', 'no'], N)
multi_car = np.random.choice(['yes', 'no'], N)

base = 800
charges = (base
    + age * 5
    + vehicle_age * (-15)
    + annual_mileage * 0.02
    + accidents_history * 400
    + traffic_violations * 150
    + (vehicle_type == 'sports').astype(int) * 600
    + (vehicle_type == 'suv').astype(int) * 200
    + (city_type == 'urban').astype(int) * 300
    + (credit_score < 600).astype(int) * 400
    + (anti_theft == 'no').astype(int) * 100
    + np.random.normal(0, 200, N))
charges = np.clip(charges, 300, 8000).astype(int)

car_df = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'vehicle_type': vehicle_type, 'annual_mileage': annual_mileage,
    'accidents_history': accidents_history, 'traffic_violations': traffic_violations,
    'license_years': license_years, 'city_type': city_type,
    'credit_score': credit_score, 'anti_theft': anti_theft,
    'multi_car': multi_car, 'premium': charges
})
car_df.to_csv(os.path.join(DATA_DIR, 'car_insurance.csv'), index=False)
print("Car dataset created:", car_df.shape)

# ── 2. HOME INSURANCE ─────────────────────────────────────────────────────────
home_age = np.random.randint(1, 100, N)
home_value = np.random.randint(100000, 1500000, N)
square_feet = np.random.randint(600, 6000, N)
location_risk = np.random.choice(['low', 'medium', 'high'], N)
construction_type = np.random.choice(['brick', 'wood', 'concrete', 'steel'], N)
security_system = np.random.choice(['yes', 'no'], N)
smoke_detectors = np.random.choice(['yes', 'no'], N)
flood_zone = np.random.choice(['yes', 'no'], N)
roof_age = np.random.randint(0, 30, N)
owner_age = np.random.randint(22, 80, N)
claims_history = np.random.randint(0, 4, N)
garage = np.random.choice(['yes', 'no'], N)

base = 500
charges = (base
    + home_value * 0.001
    + square_feet * 0.05
    + home_age * 8
    + roof_age * 15
    + (location_risk == 'high').astype(int) * 600
    + (location_risk == 'medium').astype(int) * 300
    + (construction_type == 'wood').astype(int) * 400
    + (flood_zone == 'yes').astype(int) * 700
    + (security_system == 'no').astype(int) * 150
    + claims_history * 300
    + np.random.normal(0, 200, N))
charges = np.clip(charges, 300, 8000).astype(int)

home_df = pd.DataFrame({
    'home_age': home_age, 'home_value': home_value, 'square_feet': square_feet,
    'location_risk': location_risk, 'construction_type': construction_type,
    'security_system': security_system, 'smoke_detectors': smoke_detectors,
    'flood_zone': flood_zone, 'roof_age': roof_age, 'owner_age': owner_age,
    'claims_history': claims_history, 'garage': garage, 'premium': charges
})
home_df.to_csv(os.path.join(DATA_DIR, 'home_insurance.csv'), index=False)
print("Home dataset created:", home_df.shape)

# ── 3. LIFE INSURANCE ─────────────────────────────────────────────────────────
age = np.random.randint(18, 70, N)
gender = np.random.choice(['male', 'female'], N)
bmi = np.round(np.random.uniform(16, 45, N), 1)
smoker = np.random.choice(['yes', 'no'], N, p=[0.2, 0.8])
occupation_risk = np.random.choice(['low', 'medium', 'high'], N)
coverage_amount = np.random.randint(100000, 5000000, N)
policy_term = np.random.choice([10, 15, 20, 25, 30], N)
family_history = np.random.choice(['yes', 'no'], N, p=[0.3, 0.7])
chronic_illness = np.random.choice(['yes', 'no'], N, p=[0.2, 0.8])
alcohol = np.random.choice(['none', 'moderate', 'heavy'], N)
exercise_freq = np.random.choice(['none', 'light', 'moderate', 'intense'], N)
annual_income = np.random.randint(200000, 5000000, N)

base = 2000
charges = (base
    + age * 80
    + (gender == 'male').astype(int) * 500
    + bmi * 30
    + (smoker == 'yes').astype(int) * 3000
    + (occupation_risk == 'high').astype(int) * 1500
    + (occupation_risk == 'medium').astype(int) * 700
    + coverage_amount * 0.001
    + policy_term * 50
    + (family_history == 'yes').astype(int) * 800
    + (chronic_illness == 'yes').astype(int) * 2000
    + (alcohol == 'heavy').astype(int) * 1000
    + np.random.normal(0, 500, N))
charges = np.clip(charges, 500, 30000).astype(int)

life_df = pd.DataFrame({
    'age': age, 'gender': gender, 'bmi': bmi, 'smoker': smoker,
    'occupation_risk': occupation_risk, 'coverage_amount': coverage_amount,
    'policy_term': policy_term, 'family_history': family_history,
    'chronic_illness': chronic_illness, 'alcohol': alcohol,
    'exercise_freq': exercise_freq, 'annual_income': annual_income,
    'premium': charges
})
life_df.to_csv(os.path.join(DATA_DIR, 'life_insurance.csv'), index=False)
print("Life dataset created:", life_df.shape)

# ── 4. TRAVEL INSURANCE ───────────────────────────────────────────────────────
age = np.random.randint(18, 80, N)
trip_duration = np.random.randint(1, 90, N)
destination_risk = np.random.choice(['low', 'medium', 'high', 'extreme'], N)
trip_cost = np.random.randint(5000, 500000, N)
num_travelers = np.random.randint(1, 8, N)
travel_purpose = np.random.choice(['leisure', 'business', 'adventure', 'medical'], N)
pre_existing_condition = np.random.choice(['yes', 'no'], N, p=[0.25, 0.75])
coverage_type = np.random.choice(['basic', 'standard', 'premium'], N)
medical_coverage = np.random.randint(50000, 1000000, N)
baggage_coverage = np.random.randint(5000, 100000, N)
flight_delay = np.random.choice(['yes', 'no'], N)
adventure_sports = np.random.choice(['yes', 'no'], N, p=[0.3, 0.7])

base = 200
charges = (base
    + trip_duration * 15
    + (destination_risk == 'extreme').astype(int) * 2000
    + (destination_risk == 'high').astype(int) * 1000
    + (destination_risk == 'medium').astype(int) * 400
    + trip_cost * 0.005
    + num_travelers * 100
    + age * 5
    + (pre_existing_condition == 'yes').astype(int) * 800
    + (coverage_type == 'premium').astype(int) * 500
    + (coverage_type == 'standard').astype(int) * 200
    + (adventure_sports == 'yes').astype(int) * 600
    + (travel_purpose == 'adventure').astype(int) * 400
    + np.random.normal(0, 150, N))
charges = np.clip(charges, 100, 15000).astype(int)

travel_df = pd.DataFrame({
    'age': age, 'trip_duration': trip_duration, 'destination_risk': destination_risk,
    'trip_cost': trip_cost, 'num_travelers': num_travelers, 'travel_purpose': travel_purpose,
    'pre_existing_condition': pre_existing_condition, 'coverage_type': coverage_type,
    'medical_coverage': medical_coverage, 'baggage_coverage': baggage_coverage,
    'flight_delay': flight_delay, 'adventure_sports': adventure_sports, 'premium': charges
})
travel_df.to_csv(os.path.join(DATA_DIR, 'travel_insurance.csv'), index=False)
print("Travel dataset created:", travel_df.shape)

print("\nAll datasets generated successfully!")