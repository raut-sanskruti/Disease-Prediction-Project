import streamlit as st
import pickle
import numpy as np

# Set up the page
st.set_page_config(page_title=" Disease Predictor", layout="centered")

# Add a banner or logo (optional)
st.image("https://i.imgur.com/Z7AzH2c.png", width=300)  # Replace with your image URL or local path

# Load the model and encoders
with open("rf_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("symptom_encoder.pkl", "rb") as f:
    symptom_encoder = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Sidebar info
st.sidebar.title("About")
st.sidebar.info(
    "This app uses a trained Random Forest model to predict the most likely disease based on selected symptoms.\n\n"
    "**Tech Used:** Python, Scikit-learn, Streamlit\n\n"
    "**By:** Sanskruti Raut "
)

# App title and description
st.title(" Disease Prediction Based on Symptoms")
st.write(" Select symptoms from the list below and click **Predict Disease** to see the result.")

# Multiselect box for symptoms
all_symptoms = list(symptom_encoder.classes_)
selected_symptoms = st.multiselect("🔍 Choose your symptoms:", all_symptoms)

# Prediction button
if st.button(" Predict Disease"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom before predicting.")
    else: 
        # Encode input
        input_data = symptom_encoder.transform([selected_symptoms])

        # Predict using the trained model
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        confidence = np.max(prediction_proba) * 100
        disease_name = label_encoder.inverse_transform([prediction])[0]

        # Display results
        st.success(f" **Predicted Disease:** {disease_name}")
        st.info(f"**Confidence Score:** {confidence:.2f}%")

# Footer
st.markdown("---")

#streamlit run app.py

