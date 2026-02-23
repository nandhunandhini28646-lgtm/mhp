# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained pipeline
model = joblib.load('mental_health_model.pkl')

# Get the feature names from the model (for display)
# We'll define the input fields manually based on the dataset columns.

st.set_page_config(page_title="Mental Health Predictor", layout="wide")
st.title("🧠 Mental Health Issue Predictor")
st.markdown("Enter the details below to predict whether an individual may have a mental health issue.")

# Define all input fields (same order as in training)
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        country = st.selectbox("Country", ["USA", "India", "UK", "Germany", "Brazil", "Other"])
        education = st.selectbox("Education Level",
                                 ["High School", "Some College", "Bachelor", "Master", "PhD", "Other"])
        marital_status = st.selectbox("Marital Status",
                                      ["Single", "Married", "Divorced", "Widowed", "Separated", "Prefer not to say"])
        income_level = st.selectbox("Income Level", ["Low", "Middle", "High"])
        employment_status = st.selectbox("Employment Status",
                                         ["Full-time", "Part-time", "Self-employed", "Unemployed", "Student", "Retired"])
        work_hours = st.number_input("Work Hours Per Week", min_value=0, max_value=168, value=40)
        remote_work = st.selectbox("Remote Work", ["No", "Yes", "Hybrid"])

    with col2:
        job_satisfaction = st.slider("Job Satisfaction (1-10)", 1, 10, 5)
        work_stress = st.slider("Work Stress Level (1-10)", 1, 10, 5)
        work_life_balance = st.slider("Work Life Balance (1-10)", 1, 10, 5)
        ever_bullied = st.selectbox("Ever Bullied at Work?", ["0", "1"])
        company_support = st.selectbox("Company Mental Health Support",
                                       ["No", "Yes", "Not sure"])
        exercise = st.selectbox("Exercise Per Week",
                                ["Never", "1-2 times", "3-4 times", "5+ times"])
        sleep_hours = st.number_input("Sleep Hours Per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
        caffeine = st.number_input("Caffeine Drinks Per Day", min_value=0, max_value=20, value=2)
        alcohol = st.selectbox("Alcohol Frequency",
                               ["Never", "Rarely", "Weekly", "Daily"])
        smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])

    with col3:
        screen_time = st.number_input("Screen Time Hours/Day", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
        social_media = st.number_input("Social Media Hours/Day", min_value=0.0, max_value=24.0, value=2.0, step=0.1)
        hobby_time = st.number_input("Hobby Time Hours/Week", min_value=0, max_value=168, value=5)
        diet_quality = st.selectbox("Diet Quality", ["Poor", "Average", "Good", "Excellent"])
        financial_stress = st.slider("Financial Stress (1-10)", 1, 10, 5)

        # Symptoms (Likert 1-10)
        feeling_sad = st.slider("Feeling Sad/Down (1-10)", 1, 10, 5)
        loss_interest = st.slider("Loss of Interest (1-10)", 1, 10, 5)
        sleep_trouble = st.slider("Sleep Trouble (1-10)", 1, 10, 5)
        fatigue = st.slider("Fatigue (1-10)", 1, 10, 5)
        poor_appetite = st.slider("Poor Appetite or Overeating (1-10)", 1, 10, 5)
        worthless = st.slider("Feeling Worthless (1-10)", 1, 10, 5)
        concentration = st.slider("Concentration Difficulty (1-10)", 1, 10, 5)
        anxious = st.slider("Anxious/Nervous (1-10)", 1, 10, 5)
        panic_attacks = st.slider("Panic Attacks (1-10)", 1, 10, 5)
        mood_swings = st.slider("Mood Swings (1-10)", 1, 10, 5)
        irritability = st.slider("Irritability (1-10)", 1, 10, 5)
        obsessive = st.slider("Obsessive Thoughts (1-10)", 1, 10, 5)
        compulsive = st.slider("Compulsive Behavior (1-10)", 1, 10, 5)
        self_harm = st.slider("Self-Harm Thoughts (1-10)", 1, 10, 5)
        suicidal = st.slider("Suicidal Thoughts (1-10)", 1, 10, 5)

        family_history = st.selectbox("Family History of Mental Illness", ["0", "1"])
        prev_diagnosed = st.selectbox("Previously Diagnosed", ["0", "1"])
        ever_sought_treatment = st.selectbox("Ever Sought Treatment", ["0", "1"])
        on_therapy = st.selectbox("On Therapy Now", ["0", "1"])
        on_medication = st.selectbox("On Medication", ["0", "1"])
        trauma_history = st.selectbox("Trauma History", ["0", "1"])

        social_support = st.slider("Social Support (1-10)", 1, 10, 5)
        close_friends = st.number_input("Close Friends Count", min_value=0, max_value=20, value=3)
        feel_understood = st.slider("Feel Understood (1-10)", 1, 10, 5)
        loneliness = st.slider("Loneliness (1-10)", 1, 10, 5)
        discuss_mental = st.selectbox("Discuss Mental Health",
                                      ["Never", "Rarely", "Sometimes", "Yes easily"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create a dataframe from the inputs (order must match training columns)
    input_data = pd.DataFrame([[
        age, gender, country, education, marital_status, income_level,
        employment_status, work_hours, remote_work, job_satisfaction,
        work_stress, work_life_balance, ever_bullied, company_support,
        exercise, sleep_hours, caffeine, alcohol, smoking, screen_time,
        social_media, hobby_time, diet_quality, financial_stress,
        feeling_sad, loss_interest, sleep_trouble, fatigue, poor_appetite,
        worthless, concentration, anxious, panic_attacks, mood_swings,
        irritability, obsessive, compulsive, self_harm, suicidal,
        family_history, prev_diagnosed, ever_sought_treatment, on_therapy,
        on_medication, trauma_history, social_support, close_friends,
        feel_understood, loneliness, discuss_mental
    ]], columns=[
        'Age', 'Gender', 'Country', 'Education', 'Marital_Status', 'Income_Level',
        'Employment_Status', 'Work_Hours_Per_Week', 'Remote_Work', 'Job_Satisfaction',
        'Work_Stress_Level', 'Work_Life_Balance', 'Ever_Bullied_At_Work', 'Company_Mental_Health_Support',
        'Exercise_Per_Week', 'Sleep_Hours_Night', 'Caffeine_Drinks_Day', 'Alcohol_Frequency', 'Smoking',
        'Screen_Time_Hours_Day', 'Social_Media_Hours_Day', 'Hobby_Time_Hours_Week', 'Diet_Quality',
        'Financial_Stress', 'Feeling_Sad_Down', 'Loss_Of_Interest', 'Sleep_Trouble', 'Fatigue',
        'Poor_Appetite_Or_Overeating', 'Feeling_Worthless', 'Concentration_Difficulty', 'Anxious_Nervous',
        'Panic_Attacks', 'Mood_Swings', 'Irritability', 'Obsessive_Thoughts', 'Compulsive_Behavior',
        'Self_Harm_Thoughts', 'Suicidal_Thoughts', 'Family_History_Mental_Illness', 'Previously_Diagnosed',
        'Ever_Sought_Treatment', 'On_Therapy_Now', 'On_Medication', 'Trauma_History', 'Social_Support',
        'Close_Friends_Count', 'Feel_Understood', 'Loneliness', 'Discuss_Mental_Health'
    ])

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"⚠️ High risk of mental health issue (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low risk of mental health issue (Probability: {probability:.2f})")

    # Optional: show input data for verification
    with st.expander("View entered data"):
        st.dataframe(input_data)