import pickle  
import numpy as np
import pandas as pd

# Define dataset paths
files = {
    "Glucose": (r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/glucose_dataset_resampled.csv", "blood_glucose_level", "Risk_Glucose"),
    "Systolic": (r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/systolic_bp_dataset_resampled.csv", "Systolic", "Systolic_Risk"),
    "Diastolic": (r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/diastolic_bp_dataset_resampled.csv", "Diastolic", "Diastolic_Risk"),
    "SpO2": (r"C:/Users/Anand Raj/OneDrive/Desktop/project/dta/spo2_dataset_updated.csv", "SpO2", "Risk_SpO2")
}

# Load Data
df_glucose = pd.read_csv(files["Glucose"][0])
df_systolic = pd.read_csv(files["Systolic"][0])
df_diastolic = pd.read_csv(files["Diastolic"][0])
df_spo2 = pd.read_csv(files["SpO2"][0])

# Define file paths for trained models and scalers
model_files = {
    "Glucose": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\Glucose_model.pkl",
    "Systolic": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\Systolic_model.pkl",
    "Diastolic": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\Diastolic_model.pkl",
    "SpO2": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\SpO2_model.pkl"
}

scaler_files = {
    "Glucose": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\Glucose_scaler.pkl",
    "Systolic": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\Systolic_scaler.pkl",
    "Diastolic": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\Diastolic_scaler.pkl",
    "SpO2": r"C:\Users\Anand Raj\OneDrive\Desktop\project\ml_model\SpO2_scaler.pkl"
}

# Load models and scalers
models = {}
scalers = {}

for category in model_files.keys():
    try:
        with open(model_files[category], "rb") as model_file:
            models[category] = pickle.load(model_file)
        if category in scaler_files:
            with open(scaler_files[category], "rb") as scaler_file:
                scalers[category] = pickle.load(scaler_file)
    except FileNotFoundError:
        print(f"❌ Error: Missing model or scaler file for {category}.")
        exit()

# WHO Classification Mapping
who_guidelines = {
    "Glucose": {
        0: "Possibly Hypoglycemia",
        1: "Lower Normal Range",
        2: "Normal",
        3: "Pre-Diabetes",
        4: "Diabetes/Hyperglycemia"
    },
    "Systolic": {
        1: "Low BP/Hypotension",
        2: "Normal",
        3: "Elevated BP",
        4: "Hypertension Stage 1",
        5: "Hypertension Stage 2"
    },
    "Diastolic": {
        1: "Low BP/Hypotension",
        2: "Normal",
        3: "Elevated BP",
        4: "Hypertension"
    },
    "SpO2": {
        0: "Severe Hypoxia",
        1: "Moderate Hypoxia",
        2: "Mild Hypoxia",
        3: "Normal"
    }
}

def predict_classification():
    try:
        # Get user input
        glucose_level = float(input("Enter Blood Glucose Level: "))
        systolic = float(input("Enter Systolic Blood Pressure: "))
        diastolic = float(input("Enter Diastolic Blood Pressure: "))
        spo2 = float(input("Enter SpO2 Level: "))

        # Prepare input for models
        glucose_scaled = scalers["Glucose"].transform(np.array([[glucose_level]]))
        systolic_scaled = scalers["Systolic"].transform(np.array([[systolic]]))
        diastolic_scaled = scalers["Diastolic"].transform(np.array([[diastolic]]))
        spo2_scaled = scalers["SpO2"].transform(np.array([[spo2]]))

        # Make predictions
        glucose_prediction = models["Glucose"].predict(glucose_scaled)[0]
        systolic_prediction = models["Systolic"].predict(systolic_scaled)[0]
        diastolic_prediction = models["Diastolic"].predict(diastolic_scaled)[0]
        spo2_prediction = models["SpO2"].predict(spo2_scaled)[0]

        # Display results with WHO interpretation
        print("\n🔹 **Risk Classification Results:**")
        print(f"🩸 **Glucose Level:** {who_guidelines['Glucose'].get(glucose_prediction, 'Unknown')}")
        print(f"💓 **Systolic Pressure:** {who_guidelines['Systolic'].get(systolic_prediction, 'Unknown')}")
        print(f"💓 **Diastolic Pressure:** {who_guidelines['Diastolic'].get(diastolic_prediction, 'Unknown')}")
        print(f"🌬 **SpO2 Level:** {who_guidelines['SpO2'].get(spo2_prediction, 'Unknown')}")

    except ValueError:
        print("❌ Error: Invalid input. Please enter numeric values.")

if __name__ == "__main__":
    predict_classification()


