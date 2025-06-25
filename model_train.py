# model_train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load your dataset
df = pd.read_csv("2d_materials.csv")

X = df[["surface_area", "bandgap"]]  # Features
y = df["H2_adsorption_energy"]       # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("MAE:", mean_absolute_error(y_test, model.predict(X_test)))

# Save model
import joblib
joblib.dump(model, "rf_model.joblib")
