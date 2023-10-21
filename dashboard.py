import pandas as pd
from scipy.io import arff

st.header("Semiconductor abnormality classification")

option = st.sidebar.selectbox(
    'Select a plot',
     ['Visualization', 'Prediction model']
)

if option=='Visualization':
    st.title('Data visualization')


else: 
    st.title('Data visualization')