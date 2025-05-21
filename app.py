from flask import Flask, render_template, request, redirect, url_for, session, send_from_directory, flash
import os
import model

app = Flask(__name__)
app.secret_key = 'a9b1c2d3e4f567890123456789abcdef'
DATA_DIR = "data"

def check_credentials(username, password):
    try:
        with open(os.path.join(DATA_DIR, 'login.txt'), 'r') as file:
            for line in file:
                if line.strip() == f"{username}:{password}":
                    return True
    except FileNotFoundError:
        pass
    return False

def get_patient_data(username):
    details_path = os.path.join(DATA_DIR, username, 'details.txt')
    patient_data = {}
    if os.path.exists(details_path):
        with open(details_path, 'r') as file:
            for line in file:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    patient_data[key.strip()] = value.strip()
    return patient_data

def handle_prediction(glucose, systolic, diastolic, spo2):
    results = model.predict_risk_with_guidelines(glucose, systolic, diastolic, spo2)
    risky_terms = ["Hypoxia", "Hypertension", "Hypotension", "Diabetes", "Hyperglycemia", "Hypoglycemia"]
    statuses = [v["who_guidelines"] for v in results.values()]
    final_status = "Critical" if any(term in status for term in risky_terms for status in statuses) else "Healthy"

    return {
        "glucose": glucose,
        "glucose_status": results["Glucose"]["who_guidelines"],
        "systolic": systolic,
        "systolic_status": results["Systolic"]["who_guidelines"],
        "diastolic": diastolic,
        "diastolic_status": results["Diastolic"]["who_guidelines"],
        "oxygen_level": spo2,
        "oxygen_status": results["SpO2"]["who_guidelines"],
        "final_status": final_status}

@app.route('/')
def home():
    return render_template('login.html')

@app.route('/sign_up', methods=['GET', 'POST'])
def sign_up():
    if request.method == 'POST':
        data = request.form
        username = data['patient_name']
        user_folder = os.path.join(DATA_DIR, username)
        os.makedirs(user_folder, exist_ok=True)

        with open(os.path.join(user_folder, 'details.txt'), 'w') as file:
            for field in ['Patient Name', 'Caregiver Name', 'Contact Number', 'Address']:
                file.write(f"{field}:{data.get(field.lower().replace(' ', '_'))}\n")

        if 'profile_image' in request.files:
            dp = request.files['profile_image']
            if dp and dp.filename:
                dp.save(os.path.join(user_folder, 'dp.jpg'))

        with open(os.path.join(DATA_DIR, 'login.txt'), 'a') as f:
            f.write(f"\n{username}:{data['password']}")

        flash('Registered successfully! Redirecting to Login...', 'success')
        return redirect(url_for('home'))

    return render_template('sign_up.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    if check_credentials(username, password):
        session['username'] = username
        return redirect(url_for('profile'))
    flash('Invalid username or password', 'error')
    return redirect(url_for('home'))

@app.route('/forgot_password')
def forgot_password():
    return render_template('forgot_pass.html')

@app.route('/reset_password', methods=['POST'])
def reset_password():
    username = request.form['username']
    new_password = request.form['password']
    confirm_password = request.form['confirm_password']

    if new_password != confirm_password:
        flash("Passwords do not match", "error")
        return redirect(url_for('forgot_password'))

    login_file = os.path.join(DATA_DIR, 'login.txt')
    updated = False
    with open(login_file, 'r') as f:
        lines = f.readlines()
    with open(login_file, 'w') as f:
        for line in lines:
            if line.startswith(f"{username}:"):
                f.write(f"{username}:{new_password}\n")
                updated = True
            else:
                f.write(line)

    flash("Password updated successfully!" if updated else "Username not found", "success" if updated else "error")
    return redirect(url_for('home' if updated else 'forgot_password'))

@app.route('/profile_picture/<username>')
def profile_picture(username):
    image_path = f'data/{username}/dp.jpg'
    if os.path.exists(image_path):
        return send_from_directory(f'data/{username}/', 'dp.jpg')
    return send_from_directory('static', 'default_dp.jpg')

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    username = session.get('username')
    if not username:
        return redirect(url_for('home'))

    patient_data = get_patient_data(username)
    profile_image = url_for('profile_picture', username=username)

    if request.method == 'POST':
        return render_template('predict.html', **handle_prediction(
            float(request.form['glucose']),
            float(request.form['systolic']),
            float(request.form['diastolic']),
            float(request.form['oxygen_level'])))

    return render_template('profile.html',
                           username=username,
                           profile_image=profile_image,
                           name=patient_data.get('Patient Name', 'N/A'),
                           caregiver_name=patient_data.get('Caregiver Name', 'N/A'),
                           contact=patient_data.get('Contact Number', 'N/A'),
                           address=patient_data.get('Address', 'N/A'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get("username"):
        return redirect(url_for("home"))
    return render_template("predict.html", **handle_prediction(
        float(request.form['glucose']),
        float(request.form['systolic']),
        float(request.form['diastolic']),
        float(request.form['spo2'])))

if __name__ == '__main__':
    app.run(debug=True) 