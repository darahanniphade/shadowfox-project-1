# Load saved model & scaler
import joblib
import pandas as pd
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