import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load Corrected Datasets
df_glucose = pd.read_csv(r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/glucose_dataset_resampled.csv")
df_systolic = pd.read_csv(r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/systolic_bp_dataset_resampled.csv")
df_diastolic = pd.read_csv(r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/diastolic_bp_dataset_resampled.csv")
df_spo2 = pd.read_csv(r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/spo2_dataset_updated.csv")

# Feature Selection - Corrected column names
glucose_features = ["Glucose"]
systolic_features = ["Systolic"]
diastolic_features = ["Diastolic"]
spo2_features = ["SpO2"]

# Target Labels - Corrected column names
glucose_label = "Risk_Level"
systolic_label = "Systolic_Risk"
diastolic_label = "Diastolic_Risk"
spo2_label = "Risk_Level"

# Train-Test Split Function
def split_data(df, features, label):
    X = df[features]
    y = df[label]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Splitting Data
X_train_g, X_test_g, y_train_g, y_test_g = split_data(df_glucose, glucose_features, glucose_label)
X_train_sy, X_test_sy, y_train_sy, y_test_sy = split_data(df_systolic, systolic_features, systolic_label)
X_train_dy, X_test_dy, y_train_dy, y_test_dy = split_data(df_diastolic, diastolic_features, diastolic_label)
X_train_s, X_test_s, y_train_s, y_test_s = split_data(df_spo2, spo2_features, spo2_label)

# Scaling Data
scalers = {
    "Glucose": StandardScaler(),
    "Systolic": StandardScaler(),
    "Diastolic": StandardScaler(),
    "SpO2": StandardScaler()
}

X_train_g = scalers["Glucose"].fit_transform(X_train_g)
X_test_g = scalers["Glucose"].transform(X_test_g)
X_train_sy = scalers["Systolic"].fit_transform(X_train_sy)
X_test_sy = scalers["Systolic"].transform(X_test_sy)
X_train_dy = scalers["Diastolic"].fit_transform(X_train_dy)
X_test_dy = scalers["Diastolic"].transform(X_test_dy)
X_train_s = scalers["SpO2"].fit_transform(X_train_s)
X_test_s = scalers["SpO2"].transform(X_test_s)

# Train Random Forest Models
models = {}
for category, X_train, y_train in zip([
    "Glucose", "Systolic", "Diastolic", "SpO2"],
    [X_train_g, X_train_sy, X_train_dy, X_train_s],
    [y_train_g, y_train_sy, y_train_dy, y_train_s]):
    model = RandomForestClassifier(
        n_estimators=150, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42
    )
    model.fit(X_train, y_train)
    models[category] = model
    
    # Save model and scaler
    with open(f"C:/Users/Anand Raj/OneDrive/Desktop/project/ml_model/{category}_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)
    with open(f"C:/Users/Anand Raj/OneDrive/Desktop/project/ml_model/{category}_scaler.pkl", "wb") as scaler_file:
        pickle.dump(scalers[category], scaler_file)

# Evaluation
for category, X_test, y_test in zip([
    "Glucose", "Systolic", "Diastolic", "SpO2"],
    [X_test_g, X_test_sy, X_test_dy, X_test_s],
    [y_test_g, y_test_sy, y_test_dy, y_test_s]):
    y_pred = models[category].predict(X_test)
    print(f"{category} Classification Report:")
    print(classification_report(y_test, y_pred))

# Load models and scalers for prediction
loaded_models = {}
loaded_scalers = {}
for category in ["Glucose", "Systolic", "Diastolic", "SpO2"]:
    with open(f"C:/Users/Anand Raj/OneDrive/Desktop/project/ml_model/{category}_model.pkl", "rb") as model_file:
        loaded_models[category] = pickle.load(model_file)
    with open(f"C:/Users/Anand Raj/OneDrive/Desktop/project/ml_model/{category}_scaler.pkl", "rb") as scaler_file:
        loaded_scalers[category] = pickle.load(scaler_file)

# Prediction Function
def predict_risk(glucose, systolic, diastolic, spo2):
    glucose_scaled = loaded_scalers["Glucose"].transform([[glucose]])
    glucose_pred = loaded_models["Glucose"].predict(glucose_scaled)[0]
    systolic_scaled = loaded_scalers["Systolic"].transform([[systolic]])
    systolic_pred = loaded_models["Systolic"].predict(systolic_scaled)[0]
    diastolic_scaled = loaded_scalers["Diastolic"].transform([[diastolic]])
    diastolic_pred = loaded_models["Diastolic"].predict(diastolic_scaled)[0]
    spo2_scaled = loaded_scalers["SpO2"].transform([[spo2]])
    spo2_pred = loaded_models["SpO2"].predict(spo2_scaled)[0]

    return {
        "Glucose Risk": glucose_pred,
        "Systolic BP Risk": systolic_pred,
        "Diastolic BP Risk": diastolic_pred,
        "SpO2 Risk": spo2_pred
    }

# Example Prediction
example_input = predict_risk(100, 120, 80, 99)
print("Example Risk Classification:", example_input)
