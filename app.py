# app.py (UPDATED with past_choices)

from flask import Flask, render_template, request
from database import SessionLocal, UserPreference
import tensorflow as tf
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained TensorFlow model
model = tf.keras.models.load_model('meal_recommender.h5')

# Load LabelEncoders and Scaler
label_encoder_pref = joblib.load('label_encoder_pref.pkl')
label_encoder_energy = joblib.load('label_encoder_energy.pkl')
label_encoder_meal = joblib.load('label_encoder_meal.pkl')
scaler = joblib.load('feature_scaler.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    session = SessionLocal()
    recommendation = None

    # Dynamic dropdown options
    preference_options = list(label_encoder_pref.classes_)
    energy_options = list(label_encoder_energy.classes_)

    if request.method == 'POST':
        dietary_pref = request.form.get('dietary')
        energy_pref = request.form.get('energy')

        try:
            # Encode inputs
            dietary_encoded = label_encoder_pref.transform([dietary_pref])[0]
            energy_encoded = label_encoder_energy.transform([energy_pref])[0]

            # Use average values for numeric features
            avg_nutrient_score = 0.5
            avg_health_score = 0.5
            avg_diversity_score = 0.5

            input_features = np.array([[dietary_encoded, energy_encoded, avg_nutrient_score, avg_health_score, avg_diversity_score]])

            # Predict
            preds = model.predict(input_features)
            meal_index = np.argmax(preds)

            # Decode meal name
            recommendation = label_encoder_meal.inverse_transform([meal_index])[0]

            # 🆕 Save user selection + recommended meal
            user_pref = UserPreference(dietary_preference=dietary_pref, previous_choice=recommendation)
            session.add(user_pref)
            session.commit()

        except Exception as e:
            recommendation = f"❌ Error during prediction: {str(e)}"

    # 🆕 Fetch past choices after saving
    past_choices = session.query(UserPreference).all()

    return render_template(
        'index.html',
        recommendation=recommendation,
        preferences=preference_options,
        energies=energy_options,
        past_choices=past_choices  # 🆕 Pass to template
    )

if __name__ == '__main__':
    app.run(debug=True)
