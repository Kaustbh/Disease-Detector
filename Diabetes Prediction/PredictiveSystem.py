import pickle
import numpy as np
import streamlit as st

# loading the saved model

loaded_model=pickle.load(open('D:/Visual Studio Python/Project/Diabetes Prediction/trained_model.sav','rb'))

#creating the function 
def diabetes_prediction(input_data):
   
    # Changing the input data to numpy array 
    input_data_numpy_array=np.asarray(input_data)

    #reshaping the array as we are predicting only one instance

    input_data_reshaped=input_data_numpy_array.reshape(1,-1)

    #standarizing the data

    #std_data=scalar.transform(input_data_reshaped)

    prediction=loaded_model.predict(input_data_reshaped)

    if(prediction[0]==0):
        return 'The Person is not Diabetic'
    else:
        return 'The Person is Diabetic'
    

def main():

    #giving a title
    st.title('Diabetes Prediction WebApp')

    #getting the input data from user

    Pregnancies=st.text_input('Number of Pregnancies')
    
    Glucose=st.text_input('Glucose Level')

    BloodPressure=st.text_input('Blood Pressure')

    SkinThickness=st.text_input('Skinthickness value')

    Insulin=st.text_input('Insulin Level')

    BMI=st.text_input('BMI value')

    DiabetesPedigreeFunction=st.text_input('DPF value')

    Age=st.text_input('Age')

    diagonis=' '
    # creating a Button 

    if st.button('Diabetes Test Result'):
        diagonis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
                
    st.success(diagonis)


if __name__=='__main__':  # used for running a file from cmd only 
  main()
