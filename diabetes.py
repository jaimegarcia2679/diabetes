# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:07:37 2022

@author: JGarciaP
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib

model_filename = 'rf_model.pkl'
loaded_model = joblib.load(model_filename)
print("Modelo Cargado")


st.title('Status Diabetes')
st.header("Consulte su glucosa")
st.subheader("Ingrese los datos para su verificación")


with st.form(key='diabetes-pred-form'):

    col1, col2, col3, col4= st.columns(4)
    embarazos = col1.slider(label='Cuantos embarazos ha tenido:', min_value=0, max_value=20)
    glucosa = col2.slider(label='Ingrese valor de glucosa:', min_value=50, max_value=400)
    diastolica = col3.text_input(label='Ingrese su presión:')
    grosor_piel = col4.text_input(label='Ingrese su grosor de piel:')


    col5, col6, col7, col8= st.columns(4)
    insulina= col5.slider(label='Valor de Insulina:', min_value=20, max_value=400)
    BMI = col6.slider(label='Ingrese valor IMC:', min_value=50, max_value=400)
    diabetes_hereditaria= col7.text_input(label='Ingrese su ascendencia:')
    age= col8.text_input(label='Ingrese su edad:')

    submit = st.form_submit_button(label='Analizar')
    
    calculo = pd.DataFrame({'Pregnancies':embarazos, 'PlasmaGlucose':glucosa, 'DiastolicBloodPressure':diastolica, 
                            'TricepsThickness':grosor_piel, 'SerumInsulin':insulina, 'BMI':BMI,
                            'DiabetesPedigree':diabetes_hereditaria, 
                            'Age':age},index=[0])
    predicted_diabetes = loaded_model.predict(calculo)[0]
    print(predicted_diabetes)
    

