import pandas as pd
import streamlit as st
from scipy.io import arff
import pickle
import numpy as np
import sklearn
import time

st.write(f"sklearn version: {sklearn.__version__}")
st.header("Semiconductor abnormality classification")

def import_dataset(dataset): 
    with open(dataset, 'r', encoding='utf-8') as file:
        raw_data, meta = arff.loadarff(file)

    df = pd.DataFrame(raw_data)
    return df

# Import data 
df_train = import_dataset('Wafer_TRAIN.arff')
df_test = import_dataset('Wafer_TEST.arff')

# Replace character target to binary 
df_train['target'] = df_train['target'].replace({b'1': 1, b'-1': 0})
df_test['target'] = df_test['target'].replace({b'1': 1, b'-1': 0})

option = st.sidebar.selectbox(
    'Select page',
     ['Visualization', 'Prediction model']
)

if option=='Visualization':
    st.title('Data visualization')

    def pair_wise(dataframe):
        corr = dataframe.corr().abs()
        corr_df = corr[(corr > 0.7) & (corr < 1)]
        
        corr_df.loc["No. of pairs"] = corr_df.count()
        corr_df.loc["Sum of pairs"] = corr_df.sum()
        return corr_df

    corr_df = pair_wise(X_train).reset_index()

    attribute_names = [row[0] for row in corr_df.values if isinstance(row[0], str) and row[0].startswith('att')]
    first_array = corr_df.values[0]

    prop = st.selectbox("Select attributes", options=attribute_names)

else: 
    st.text('Machine Learning Model App')

    # Load the model
    try:
        with open('rf_model.pkl', 'rb') as model_file:
            rf_model = pickle.load(model_file)

    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")

    # Input data
    input_data = {}
    input_data['att1'] = st.slider('Att1', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att3'] = st.slider('Att3', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att4'] = st.slider('Att4', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att5'] = st.slider('Att5', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att6'] = st.slider('Att6', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att7'] = st.slider('Att7', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att8'] = st.slider('Att8', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att9'] = st.slider('Att9', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att10'] = st.slider('Att10', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att30'] = st.slider('Att30', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att35'] = st.slider('Att35', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att37'] = st.slider('Att37', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att38'] = st.slider('Att38', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att39'] = st.slider('Att39', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att41'] = st.slider('Att41', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att46'] = st.slider('Att46', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att47'] = st.slider('Att47', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att49'] = st.slider('Att49', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att111'] = st.slider('Att111', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att112'] = st.slider('Att112', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att113'] = st.slider('Att113', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att114'] = st.slider('Att114', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att115'] = st.slider('Att115', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att117'] = st.slider('Att117', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att118'] = st.slider('Att118', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att121'] = st.slider('Att121', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att123'] = st.slider('Att123', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att136'] = st.slider('Att136', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att138'] = st.slider('Att138', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att139'] = st.slider('Att139', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att140'] = st.slider('Att140', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att141'] = st.slider('Att141', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att142'] = st.slider('Att142', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att143'] = st.slider('Att143', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att144'] = st.slider('Att144', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att145'] = st.slider('Att145', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att146'] = st.slider('Att146', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att147'] = st.slider('Att147', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att148'] = st.slider('Att148', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att149'] = st.slider('Att149', min_value=-4.0, max_value=4.0, value=0.5)
    input_data['att151'] = st.slider('Att151', min_value=-4.0, max_value=4.0, value=0.5) 

    input_data_df = pd.DataFrame([input_data])

    st.text(" ")
    if st.button('Predict'):
        # Predict output 
        prediction = rf_model.predict(input_data_df)[0]

        if prediction == 0: 
            result = "abnormal"
        else: 
            result = "normal"

        with st.spinner('Sending input features to model...'):
            time.sleep(2)

        st.write(f"This set of sensor data is: {result}")