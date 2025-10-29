import pandas as pd
import pickle
import streamlit as st

#Loading the saved model
loaded_bundle=pickle.load(open("C:/Users/muska/Desktop/Diabetes/trained_model_and_scaler.sav","rb"))
model = loaded_bundle["model"]
scaler = loaded_bundle["scaler"]

#creating a function for Prediction
def diabetes_prediction(input_data):
    columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
    ]

    input_df = pd.DataFrame([input_data], columns=columns)

    input_scaled=scaler.transform(input_df)
    prediction=model.predict(input_scaled)

    print(prediction)

    if(prediction[0]==0):
        return "The person is not diabetic"
    else:
        return "The person is diabetic"
    

def main():

    #Giving a Title
    st.title("Diabetes Prediction Web App")

    #Getting the Input Data from the User
    Pregnancies=st.text_input("Number of Pregnancies:")
    Glucose=st.text_input("Glucose Level:")
    BloodPressure=st.text_input("Blood Pressure Value:")
    SkinThickness=st.text_input("Skin Thickness Value:")
    Insulin=st.text_input("Insulin Level:")
    BMI=st.text_input("BMI value:")
    DiabetesPedigreeFunction=st.text_input("Diabetes Pedigree Function Value:")
    Age=st.text_input("Age of the person:")

    #code for prediction
    diagnosis=''

    #creating a button for prediction
    if(st.button("Diabetes Test Result")):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)

if __name__=='__main__':
    main()