# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:13:56 2026

@author: anjith
"""

import pickle
import numpy as np
import pandas as pd

loaded_model=pickle.load(open("DIABETES_PREDICTION.sav","rb"))
scaler = pickle.load(open("scaler.sav", "rb"))


input_data = (5,166,72,19,175,25.8,0.587,51)

# 2. Change it to a numpy array and reshape
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# 3. Create a DataFrame with the exact feature names your scaler expects
# (Replace these with the exact names used during your model training)
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
input_data_df = pd.DataFrame(input_data_reshaped, columns=feature_names)

# 4. Standardize the data using the DataFrame instead of the raw array
std_data = scaler.transform(input_data_df)

# 5. Predict
prediction = loaded_model.predict(std_data)
print("prediction:",prediction[0])

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')
