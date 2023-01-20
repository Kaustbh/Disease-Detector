import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
from tensorflow import keras
import numpy as np

diabetes=pickle.load(open('D:/Visual Studio Python/Project/MultipleDisease Prediction/MutilpleDisease Prediction/diabetes_model.sav','rb'))
parkinson=pickle.load(open('D:/Visual Studio Python/Project/MultipleDisease Prediction/MutilpleDisease Prediction/Parkinson_model.sav','rb'))
Breast=tf.keras.models.load_model('D:/Visual Studio Python/Project/MultipleDisease Prediction/MutilpleDisease Prediction/BreastCancer_model.h5' )
#Breast=pickle.load(open('D:/Visual Studio Python/Project/MultipleDisease Prediction/model.pkl','rb'))

# sidebar for navigation

with st.sidebar: 
 selected=option_menu('MultipleDisease Prediction',
      ['BreastCancer Detection','Diabetes Prediction','Parkinson Prediction']
       ,default_index=0)


# BreastCancer Page

if(selected=='BreastCancer Detection'):

    st.title('BreastCancer Detection System')

    mean_radius=st.text_input('M_Radius')
    mean_texture=st.text_input('M_Textur')
    mean_perimeter=st.text_input('M_Perimeter')
    mean_area=st.text_input('M_Area')
    mean_smoothness=st.text_input('M_Smoothness')
    mean_compactness=st.text_input('M_Compactness')
    mean_concavity=st.text_input('M_Concavity')
    mean_concave_points=st.text_input('M_Concave_Points')
    mean_symmetry=st.text_input('M_Symmetry')
    mean_fractal_dimension=st.text_input('M_Fractal_Dimension')
    radius_error=st.text_input('Radius_E_Er')
    texture_error=st.text_input('Texture')
    perimeter_error=st.text_input('Perimeter_E')
    area_error=st.text_input('Area_E')
    smoothness_error=st.text_input('Smoothness_E')
    compactness_error=st.text_input('Compactness_E')
    concavity_error=st.text_input('Concavity_E')
    concave_points_error=st.text_input('Comcave_Points_E')
    symmetry_error=st.text_input('Symmetry_E')
    fractal_dimension_error=st.text_input('Fractal_Dimension_E')
    worst_radius=st.text_input('Radius_W')
    worst_texture=st.text_input('Textur_W')
    worst_perimeter=st.text_input('Perimeter_W')
    worst_area=st.text_input('Area_W')
    worst_smoothness=st.text_input('Smoothness_W')
    worst_compactness=st.text_input('Compactness_W')
    worst_concavity=st.text_input('Concavity_W')
    worst_concave_points=st.text_input('Concave_Points_W')
    worst_symmetry=st.text_input('Symmetry_W')
    worst_fractal_dimension=st.text_input('Fractal_Dimension_W')
    
    Bdiagnosis=''

    if st.button('Breast Cancer Result'):

        Bdiagnosis_pred=Breast.predict([[mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness,mean_compactness,
                        mean_concavity,mean_concave_points,mean_symmetry,mean_fractal_dimension,radius_error,texture_error,perimeter_error,
                        area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,
                        worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]])

        Bdiagnosis_pred==[np.argmax(Bdiagnosis_pred)]

        if(Bdiagnosis_pred[0]==0):
           Bdiagnosis='The tumor is Malignant'
        
        else:
            Bdiagnosis='The tumor is Benign'
    
    st.success(Bdiagnosis)
    

# Diabetes Page

if(selected=='Diabetes Prediction'):

    st.title('Diabetes Prediction System')

    col1,col2,col3=st.columns(3)
    #getting the input data from user
    with col1:

        Pregnancies=st.text_input('Number of Pregnancies')

        SkinThickness=st.text_input('Skinthickness value')

        DiabetesPedigreeFunction=st.text_input('DPF value')

    
    with col2:

        Glucose=st.text_input('Glucose Level')

        Insulin=st.text_input('Insulin Level')

        Age=st.text_input('Age')
    
    with col3:

        BloodPressure=st.text_input('Blood Pressure')

        BMI=st.text_input('BMI value')

    
    
    diagonis=' '

    input_data=[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]
    # creating a Button 

    if st.button('Diabetes Test Result'):
        
        input_data_numpy_array=np.asarray(input_data)

    #reshaping the array as we are predicting only one instance

        input_data_reshaped=input_data_numpy_array.reshape(1,-1)

        diagonis_pred1=diabetes[0].transform(input_data_reshaped)
        diagonis_pred2=diabetes[1].predict(diagonis_pred1)

        if(diagonis_pred2[0]==0):

            diagonis='The Person is not Diabetic'

        else:
            diagonis='The Person is Diabetic'


    st.success(diagonis)


# Parkinson Prediction Page

if(selected=='Parkinson Prediction'):

    st.title('Parkinson Prediction System')

    col1,col2,col3,col4,col5=st.columns(5)
    with col1:
         Fo=st.text_input('Fo')
         RAP=st.text_input('MDVP:RAP')
         APQ3=st.text_input('Shimmer_APQ3')
         HNR=st.text_input('HNR')
         D2=st.text_input('D2')

    with col2:
        Fhi=st.text_input('Fhi')
        PPQ=st.text_input('MDVP_PPQ')
        APQ5=st.text_input('Shimmer_APQ5')
        RPDE=st.text_input('RPDE')
        PPE=st.text_input('PPE')

    with col3:
        Flo=st.text_input('Flo')
        DDP=st.text_input('Jitter_DDP')
        APQ=st.text_input('MDVP_APQ')
        DFA=st.text_input('DFA')
    
    with col4:
        Jitter_precent= st.text_input('MDVP_Jitter(%)')
        Shimmer=st.text_input('MDVP_Shimmer')
        DDA=st.text_input('Shimmer_DDA')
        spread1=st.text_input('spread1')
    
    with col5:
        Jitter_Abs=st.text_input('MDVP_Jitter(Abs)')
        Shimmer_dB=st.text_input('MDVP_Shimmer(dB)')
        NHR=st.text_input('NHR')
        spread2=st.text_input('spread2')


    parkinson_diagnosis=''

    if st.button('Parkinsons Test Result'):
        Pdiagonis_pred=parkinson[1].predict([[Fo,Fhi,Flo,Jitter_precent,Jitter_Abs,RAP,PPQ,DDP,Shimmer,Shimmer_dB,
                                          APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])
        
        if(Pdiagonis_pred[0]==1):
            parkinson_diagnosis='The Person has Parkinson Disease'
        
        else:
            parkinson_diagnosis='The Person has not Parkinson Disease'
        
    
    st.success(parkinson_diagnosis)


