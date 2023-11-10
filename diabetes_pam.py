import streamlit as st
import joblib
import numpy as np

# Load the model and the scaler
loaded_model = joblib.load('best_logreg_model_diabetespam.pkl')
loaded_scaler = joblib.load('scaler_diabetespam.pkl')

# Streamlit webpage title
st.title("Diabetes Classification App by Pamela")

# Creating input fields for user data
pregnancies = st.number_input("Number of Pregnancies", format='%f')
glucose = st.number_input("Glucose Level", format='%f')
blood_pressure = st.number_input("Blood Pressure", format='%f')
skin_thickness = st.number_input("Skin Thickness", format='%f')
insulin = st.number_input("Insulin Level", format='%f')
bmi = st.number_input("BMI", format='%f')
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", format='%f')
age = st.number_input("Age", format='%f')

# Prediction button
if st.button("Predict Diabetes"):
    # Prepare the data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Make a prediction
    scaled_data = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(scaled_data)

    # Display the prediction
    if prediction[0] == 1:
        st.write("The model predicts Diabetes.")
    else:
        st.write("The model predicts No Diabetes.")
