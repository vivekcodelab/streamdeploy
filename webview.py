# Creating Web UI uysing streamlit

import streamlit as st
import pandas as pd
import joblib

st.title("HR Analytics")
st.write("Promotion Prediction App")

df = pd.read_csv("train_LZdllcl.csv")

#Categorical Columns
department = st.selectbox("Department", pd.unique(df['department']))
region = st.selectbox("Region", pd.unique(df['region']))
education = st.selectbox("Education", pd.unique(df['education']))
gender = st.selectbox("Gender", pd.unique(df['gender']))
recruitment_channel = st.selectbox("Recruitment_channel", pd.unique(df['recruitment_channel']))
# Numerical columns
no_of_trainings = st.number_input("No_of_trainings",)
age = st.number_input("Age")
previous_year_rating = st.selectbox("Previous_year_rating", pd.unique(df['previous_year_rating']))
length_of_service = st.selectbox("Length_of_service", pd.unique(df['length_of_service']))
KPIs_met_80_percent = st.number_input("KPIs_met >80%")
awards_won = st.number_input("Awards_won?") 
avg_training_score = st.number_input("Avg_training_score")


#Mapping the input to respective columns

inputs = {
"department" : department,
"region" : region,
"education" : education,
"gender" : gender,
"recruitment_channel" :  recruitment_channel,
"no_of_trainings" : no_of_trainings,
"age" : age,
"previous_year_rating" : previous_year_rating,
"length_of_service" : length_of_service,
"KPIs_met >80%" :  KPIs_met_80_percent,
"awards_won?" : awards_won,
"avg_training_score" : avg_training_score
}

#load the model from pickle file
model = joblib.load('promote_pipeline_model.pkl')

# Action for submit button
if st.button('Predict'):
    X_input = pd.DataFrame(inputs,index=[0])
    prediction = model.predict(X_input)
    st.write("The Predicted Value is : ")
    st.write(prediction)
