import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

who_guidelines = {
    "Glucose": {
        0: "Possibly Hypoglycemia",
        1: "Lower Normal Range",
        2: "Normal",
        3: "Pre-Diabetes",
        4: "Diabetes/ Hyperglycemia"
    },
    "Systolic": {
        1: "Low BP/ Hypotension",
        2: "Normal",
        3: "Elevated BP",
        4: "Hypertension Stage 1",
        5: "Hypertension Stage 2"
    },
    "Diastolic": {
        1: "Low BP/ Hypotension",
        2: "Normal",
        3: "Elevated BP",
        4: "Hypertension"
    },
    "SpO2": {
        0: "Severe Hypoxia",
        1: "Moderate Hypoxia",
        2: "Mild Hypoxia",
        3: "Normal"
    }}

base_path = os.path.join(os.path.dirname(__file__), "models")

trained_models = None
trained_scalers = None

def load_data_and_train_models():
    df_glucose = pd.read_csv(os.path.join(base_path, "glucose_dataset.csv"))
    df_systolic = pd.read_csv(os.path.join(base_path, "systolic_bp_dataset.csv"))
    df_diastolic = pd.read_csv(os.path.join(base_path, "diastolic_bp_dataset.csv"))
    df_spo2 = pd.read_csv(os.path.join(base_path, "spo2_dataset.csv"))

    datasets = {
        "Glucose": (df_glucose, ["Glucose"], "Risk_Level"),
        "Systolic": (df_systolic, ["Systolic"], "Systolic_Risk"),
        "Diastolic": (df_diastolic, ["Diastolic"], "Diastolic_Risk"),
        "SpO2": (df_spo2, ["SpO2"], "Risk_Level")}

    scalers = {}
    models = {}

    for category, (df, features, label) in datasets.items():
        X = df[features]
        y = df[label]

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42)
        
        model.fit(X_train_scaled, y_train)

        scalers[category] = scaler
        models[category] = model
    return models, scalers

def initialize_models():
    global trained_models, trained_scalers
    if trained_models is None or trained_scalers is None:
        trained_models, trained_scalers = load_data_and_train_models()

def predict_risk_with_guidelines(glucose, systolic, diastolic, spo2):
    initialize_models()

    inputs = {
        "Glucose": glucose,
        "Systolic": systolic,
        "Diastolic": diastolic,
        "SpO2": spo2}

    results = {}
    for category, value in inputs.items():
        input_df = pd.DataFrame([[value]], columns=[category])

        scaled_value = trained_scalers[category].transform(input_df)
        prediction = trained_models[category].predict(scaled_value)[0]
        interpretation = who_guidelines[category].get(prediction, "Unknown")

        results[category] = {
            "predicted": int(prediction),
            "who_guidelines": interpretation}
    return results