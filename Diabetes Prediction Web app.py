# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:19:42 2026

@author: anjith
"""
import numpy as np 
import pickle 
import streamlit as st 
import pandas as pd
import base64



loaded_model=pickle.load(open("DIABETES_PREDICTION.sav","rb"))
scaler = pickle.load(open("scaler.sav", "rb"))


def add_bg_from_local(image_file):

    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    st.markdown(
        f"""
    <style>

    .stApp {{
        background: linear-gradient(
            rgba(0,0,0,0.55),
            rgba(0,0,0,0.55)
        ),
        url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    h1, h2, h3 {{
        color: white;
        text-align: center;
    }}

    label {{
        color: white !important;
        font-weight: bold;
    }}

    /* Input boxes */
    div[data-baseweb="input"] > div {{
        background-color: rgba(255,255,255,0.90);
        border-radius: 8px;
    }}

    /* Table */
    table {{
        background-color: rgba(0,0,0,0.45);
        color: white;
    }}

    th {{
        background-color: rgba(0,70,120,0.8);
        color: white !important;
    }}

    td {{
        color: white !important;
    }}

    /* Button */
    .stButton > button {{
        width:100%;
        background:#0066cc;
        color:white;
        font-size:18px;
        border-radius:10px;
    }}
        </style>
        """,
        unsafe_allow_html=True,
    )
def diabetes_prediction(input_data):

    input_data_array = np.asarray(input_data, dtype=float)

    input_table = pd.DataFrame({
    "Feature": [
        "Pregnancies",
        "Glucose",
        "Blood Pressure",
        "Skin Thickness",
        "Insulin",
        "BMI",
        "Diabetes Pedigree Function",
        "Age"
    ],
    "Value": input_data_array
    })

    st.subheader("Patient Input Details")
    st.table(input_table)
    input_data_reshaped = input_data_array.reshape(1, -1)

    input_data_scaled = scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(input_data_scaled)

    

    if prediction[0] == 0:
        return "The person is Non Diabetic"
    else:
        return "The person is Diabetic"
    
  
    
def main():

    add_bg_from_local(
    r"diabetes background.png"
)
    st.title(" Diabetes Prediction App")

    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")

    submit = st.button("Diabetes Test Result")

    if submit:

        if "" in [
            Pregnancies,
            Glucose,
            BloodPressure,
            SkinThickness,
            Insulin,
            BMI,
            DiabetesPedigreeFunction,
            Age
        ]:
            st.error("Please fill all the fields.")

        else:

            input_data = [
                float(Pregnancies),
                float(Glucose),
                float(BloodPressure),
                float(SkinThickness),
                float(Insulin),
                float(BMI),
                float(DiabetesPedigreeFunction),
                float(Age)
            ]

            diagnosis = diabetes_prediction(input_data)
            st.success(diagnosis)
      
if __name__=="__main__":
    main()
