from flask import Flask, render_template, request, redirect, url_for, session, flash
from database import SessionLocal, UserPreference, User
import tensorflow as tf
import numpy as np
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Use a secure key in production

# Load ML model and encoders
model = tf.keras.models.load_model('meal_recommender.h5')
label_encoder_pref = joblib.load('label_encoder_pref.pkl')
label_encoder_energy = joblib.load('label_encoder_energy.pkl')
label_encoder_meal = joblib.load('label_encoder_meal.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Unified login/register route
@app.route('/auth', methods=['GET', 'POST'])
def auth():
    session_db = SessionLocal()

    if request.method == 'POST':
        action_type = request.form.get('action')  # login or register
        username = request.form.get('username')
        password = request.form.get('password')

        if action_type == 'login':
            user = session_db.query(User).filter_by(username=username).first()
            if user and check_password_hash(user.password, password):
                session['user_id'] = user.id
                session['username'] = user.username
                flash("Login successful.")
                return redirect(url_for('index'))
            else:
                flash("❌ Invalid username or password.")

        elif action_type == 'register':
            existing_user = session_db.query(User).filter_by(username=username).first()
            if existing_user:
                flash("❌ Username already exists.")
            else:
                hashed_pw = generate_password_hash(password)
                new_user = User(username=username, password=hashed_pw)
                session_db.add(new_user)
                session_db.commit()
                flash("✅ Registration successful. Please log in.")

    return render_template('auth.html')

# Logout route
@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.")
    return redirect(url_for('auth'))

# Home and meal recommendation
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'user_id' not in session:
        return redirect(url_for('auth'))

    session_db = SessionLocal()
    recommendation = None

    preference_options = list(label_encoder_pref.classes_)
    energy_options = list(label_encoder_energy.classes_)

    if request.method == 'POST':
        dietary_pref = request.form.get('dietary')
        energy_pref = request.form.get('energy')

        try:
            dietary_encoded = label_encoder_pref.transform([dietary_pref])[0]
            energy_encoded = label_encoder_energy.transform([energy_pref])[0]

            avg_nutrient_score = 0.5
            avg_health_score = 0.5
            avg_diversity_score = 0.5

            input_features = np.array([[dietary_encoded, energy_encoded,
                                        avg_nutrient_score, avg_health_score,
                                        avg_diversity_score]])

            preds = model.predict(input_features)
            meal_index = np.argmax(preds)
            recommendation = label_encoder_meal.inverse_transform([meal_index])[0]

            user_pref = UserPreference(
                dietary_preference=dietary_pref,
                previous_choice=recommendation
            )
            session_db.add(user_pref)
            session_db.commit()

        except Exception as e:
            recommendation = f"❌ Error during prediction: {str(e)}"

    past_choices = session_db.query(UserPreference).all()

    return render_template(
        'index.html',
        recommendation=recommendation,
        preferences=preference_options,
        energies=energy_options,
        past_choices=past_choices
    )

if __name__ == '__main__':
    app.run(debug=True)
