# shadowfox-project-1
🏡 Boston House Price Prediction
Author: Darshan Anil Niphade

This project predicts Boston house prices using multiple regression techniques. It utilizes features such as crime rate, average number of rooms, accessibility to highways, property tax rate, and other relevant socio-economic indicators. The solution includes data preprocessing, model training, evaluation, visualization, and saving the model for future predictions.

boston.csv                     # Dataset (Boston Housing Data)
boston_house_price_model.pkl   # Best trained model (saved)
scaler.pkl                     # Scaler used for feature normalization
Boston_House_Price_Report.pdf  # Generated report with plots and results
plots/                         # Folder containing plots
├── actual_vs_predicted.png    # Plot of actual vs predicted prices
├── feature_importance.png     # Plot showing feature importances
main.py                        # Main script with training & prediction code
README.md                      # This file


#⚙️ Tools & Technologies Used
Python 3.9+

Pandas – Data manipulation and analysis

NumPy – Numerical operations

Matplotlib & Seaborn – Data visualization


#🚀 Features
Cleans and preprocesses Boston Housing Dataset

Handles missing values and normalizes features

Trains and evaluates regression model

Generates and saves visual plots

Produces a detailed PDF report with results & insights

Saves model (.pkl) and scaler for future use

Predicts house prices for new data

#📊 Model & Evaluation
The model used is a Random Forest Regressor (you can change it to Linear Regression, Gradient Boosting, etc. in main.py).
Evaluation metrics include:

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

R² Score

Scikit-learn – Machine learning algorithms and preprocessing

Pickle – Saving & loading models and scalers
