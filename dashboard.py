import pandas as pd
import streamlit as st
from scipy.io import arff
from joblib import load

st.header("Semiconductor abnormality classification")

option = st.sidebar.selectbox(
    'Select a plot',
     ['Visualization', 'Prediction model']
)

if option=='Visualization':
    st.title('Data visualization')


else: 
    st.title('Data visualization')