import streamlit as st
import joblib
import pandas as pd

model = joblib.load('KNN_heart.pkl')  
scaler = joblib.load('scaler_heart.pkl')
expected_columns = joblib.load('columns.pkl')

st.title('Heart Disease Prediction')
st.markdown('Enter the following details to predict your heart disease risk')

age = st.slider('Age', min_value=18, max_value=100, value=40)
sex = st.selectbox('Sex', ['Male', 'Female'])
chest_pain = st.selectbox('Chest Pain Type', ['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.number_input('Resting Blood Pressure', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=600, value=200)
fasting_bs = st.selectbox('Fasting Blood Sugar', ['0', '1'])
resting_ecg = st.selectbox('Resting Electrocardiographic Results', ['Normal', 'ST', 'LVH'])
max_hr = st.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exercise_angina = st.selectbox('Exercise Induced Angina', ['N', 'Y'])
oldpeak = st.number_input('Oldpeak', min_value=0.0, max_value=6.0, value=1.0)
st_slope = st.selectbox('ST Slope', ['Up', 'Flat', 'Down'])

if st.button('Predict'):
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': int(fasting_bs),
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_M': 1 if sex == 'Male' else 0,
        'ChestPainType_ATA': 1 if chest_pain == 'ATA' else 0,
        'ChestPainType_NAP': 1 if chest_pain == 'NAP' else 0,
        'ChestPainType_TA': 1 if chest_pain == 'TA' else 0,
        'RestingECG_Normal': 1 if resting_ecg == 'Normal' else 0,
        'RestingECG_ST': 1 if resting_ecg == 'ST' else 0,
        'ExerciseAngina_Y': 1 if exercise_angina == 'Y' else 0,
        'ST_Slope_Flat': 1 if st_slope == 'Flat' else 0,
        'ST_Slope_Up': 1 if st_slope == 'Up' else 0
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0][prediction]
    probability = round(probability * 100, 2)
    st.write(f'Prediction: {prediction}')
    st.write(f'Probability: {probability}%')
    if prediction == 1:
        st.write('You are at risk of heart disease. Please consult a doctor.')
    else:
        st.write('You are not at risk of heart disease. You are healthy.')
        st.balloons()