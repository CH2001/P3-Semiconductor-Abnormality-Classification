import pandas as pd
import streamlit as st
from scipy.io import arff
import joblib

st.header("Semiconductor abnormality classification")

option = st.sidebar.selectbox(
    'Select page',
     ['Visualization', 'Prediction model']
)

if option=='Visualization':
    st.title('Data visualization')


else: 
    st.title('Data visualization')