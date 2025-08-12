import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from fpdf import FPDF
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ======================
# STEP 1: Load & Prepare Data
# ======================
df = pd.read_csv("HousingData.csv")  # Your CSV file
df.replace('NA', np.nan, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(df.mean(), inplace=True)

X = df.drop("MEDV", axis=1)
y = df["MEDV"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ======================
# STEP 2: Train Models
# ======================
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    results[name] = {"MSE": mse, "RMSE": rmse, "R2": r2}

# Choose best model based on R²
best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = models[best_model_name]

# ======================
# STEP 3: Save Model & Scaler
# ======================
joblib.dump(best_model, "boston_house_price_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print(f"✅ Saved best model: {best_model_name} as boston_house_price_model.pkl")
print("✅ Saved scaler as scaler.pkl")



# Feature importance (for tree models)
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importance, y=X.columns)
    plt.title(f"{best_model_name} - Feature Importance")
    plt.savefig("plots/feature_importance.png")
    plt.close()


# ======================
# STEP 6: Predict New Price Example
# ======================
# Load saved model & scaler
loaded_model = joblib.load("boston_house_price_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")

# Example new data
new_data = pd.DataFrame([{
    "CRIM": 0.1, "ZN": 18.0, "INDUS": 2.31, "CHAS": 0, "NOX": 0.538,
    "RM": 6.575, "AGE": 65.2, "DIS": 4.09, "RAD": 1, "TAX": 296,
    "PTRATIO": 15.3, "B": 396.9, "LSTAT": 4.98
}])

new_data_scaled = loaded_scaler.transform(new_data)
predicted_price = loaded_model.predict(new_data_scaled)
print(f"💰 Predicted House Price: ${predicted_price[0]*1000:.2f}")
