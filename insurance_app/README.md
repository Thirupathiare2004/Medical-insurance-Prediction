# InsureAI — Multi-Insurance Cost Predictor
### Major Project: Medical Insurance Cost Prediction using Python Flask & Random Forest

---

## 🚀 Quick Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate synthetic datasets (car, home, life, travel)
python generate_datasets.py

# 3. Train all 5 Random Forest models
python train_models.py

# 4. Run the Flask app
python app.py

# 5. Open browser at:  http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
insurance_app/
├── app.py                  # Flask backend + REST API
├── train_models.py         # Random Forest training for all 5 insurances
├── generate_datasets.py    # Synthetic dataset generator
├── requirements.txt        # Python dependencies
├── datasets/
│   ├── medical_insurance.csv   # Your original dataset (5000 records)
│   ├── car_insurance.csv       # Generated (5000 records)
│   ├── home_insurance.csv      # Generated (5000 records)
│   ├── life_insurance.csv      # Generated (5000 records)
│   └── travel_insurance.csv    # Generated (5000 records)
├── models/
│   ├── medical_model.pkl   # Trained RF model
│   ├── car_model.pkl
│   ├── home_model.pkl
│   ├── life_model.pkl
│   └── travel_model.pkl
└── templates/
    └── index.html          # Full-stack UI (single-page app)
```

---

## 🎯 Insurance Types & Features

| Type    | Input Features                               | R² Score |
|---------|----------------------------------------------|----------|
| Medical | Age, BMI, smoker, conditions, lifestyle...   | 0.876    |
| Car     | Vehicle type, accidents, mileage, credit...  | 0.872    |
| Home    | Property value, location risk, roof age...   | 0.879    |
| Life    | Coverage amount, term, health history...     | 0.905    |
| Travel  | Destination risk, trip duration, sports...   | 0.917    |

---

## 🤖 ML Algorithm: Random Forest

- **Algorithm**: Random Forest Regressor
- **Trees**: 150 estimators
- **Max Depth**: 15
- **Features**: Automatic label encoding for categorical variables
- **Extras**: Top-5 feature importance shown per prediction

## 🌟 Extra Features (beyond basic prediction)
- Real-time risk level badge (Low / Medium / High)
- Top cost drivers with importance bar charts
- Monthly premium breakdown
- Model accuracy metrics (MAE, RMSE, R²)
- Personalized tips to reduce premium
- Animated, responsive UI with tab navigation

---

## API Endpoints

| Method | Endpoint                  | Description            |
|--------|---------------------------|------------------------|
| GET    | `/`                       | Main UI                |
| POST   | `/api/predict/<type>`     | Get premium prediction |
| GET    | `/api/metrics/<type>`     | Model performance      |

`<type>` = `medical` | `car` | `home` | `life` | `travel`
