import streamlit as st
import joblib
import numpy as np

# Load the model and the scaler
loaded_model = joblib.load('best_logreg_model_diabetespam.pkl')
loaded_scaler = joblib.load('scaler_diabetespam.pkl')

# Streamlit webpage title
st.title("Diabetes Classification App by Pamela")

# Creating input fields for user data
pregnancies = st.number_input("Number of Pregnancies", min_value=0.0, value=1.0, step=1.0)
glucose = st.number_input("Glucose Level", min_value=0.0, value=100.0, step=1.0)
blood_pressure = st.number_input("Blood Pressure", min_value=0.0, value=70.0, step=1.0)
skin_thickness = st.number_input("Skin Thickness", min_value=0.0, value=20.0, step=1.0)
insulin = st.number_input("Insulin Level", min_value=0.0, value=80.0, step=1.0)
bmi = st.number_input("BMI", min_value=0.0, value=30.0, step=1.0)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, value=0.5, step=0.01)
age = st.number_input("Age", min_value=0.0, value=25.0, step=1.0)

# Prediction button
if st.button("Predict Diabetes"):
    # Prepare the data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Make a prediction
    scaled_data = loaded_scaler.transform(input_data)
    prediction = loaded_model.predict(scaled_data)

    # Display the prediction
    st.subheader("Prediction:")
    if prediction[0] == 1:
        st.write("The model predicts Diabetes.")
    else:
        st.write("The model predicts No Diabetes.")
