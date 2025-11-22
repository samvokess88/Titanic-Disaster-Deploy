import streamlit as st
import joblib  
import pandas as pd 

model = joblib.load("best_model.pkl")
LEC_Sex = joblib.load("LEC_Sex.pkl")
LEC_Embarked = joblib.load("LEC_Embarked.pkl")





st.title(' Titanic Survival Prediction App')

# user input field

pclass = st.selectbox("Passengers Class",[1,2,3])
sex = st.selectbox("Sex", ["male","female"])
age = st.number_input("Age",0,100,30)
sibsp = st.number_input("Siblings/Spouses Aboard",0,10,0)
parch = st.number_input("Parents/Children Aboard",0,10,0)
fare = st.number_input("Fare",0.0,600.0,32.0)
embarked = st.selectbox("embarked",["C","Q","S"])

# Encode user inputs

sex_encoded = LEC_Sex.transform([sex])[0]
embarked_encoded = LEC_Embarked.transform([embarked])[0]

# create input DataFrame
input_data = pd.DataFrame([{
    "Pclass":pclass,
    "Sex":sex_encoded,
    "Age":age,
    "SibSp":sibsp,
    "Parch":parch,
    "Fare":fare,
    "Embarked":embarked_encoded
}])

if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.success("The person survived")
    else:
        st.error("The person did not survive")


