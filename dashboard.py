import pandas as pd
from scipy.io import arff
from joblib import load

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
    'Select a plot',
     ['Visualization', 'Prediction model']
)

if option=='Visualization':


else: 
    # Load model
    loaded_model = load('rf_model.joblib')

    st.title('Machine Learning Model App')

    att1 = st.slider('Att1', min_value=-4.0, max_value=4.0, value=0.5)
    att3 = st.slider('Att3', min_value=-4.0, max_value=4.0, value=0.5)
    att4 = st.slider('Att4', min_value=-4.0, max_value=4.0, value=0.5)
    att5 = st.slider('Att5', min_value=-4.0, max_value=4.0, value=0.5)
    att6 = st.slider('Att6', min_value=-4.0, max_value=4.0, value=0.5)
    att7 = st.slider('Att7', min_value=-4.0, max_value=4.0, value=0.5)
    att8 = st.slider('Att8', min_value=-4.0, max_value=4.0, value=0.5)
    att9 = st.slider('Att9', min_value=-4.0, max_value=4.0, value=0.5)
    att10 = st.slider('Att10', min_value=-4.0, max_value=4.0, value=0.5)
    att30 = st.slider('Att30', min_value=-4.0, max_value=4.0, value=0.5)
    att35 = st.slider('Att35', min_value=-4.0, max_value=4.0, value=0.5)
    att37 = st.slider('Att37', min_value=-4.0, max_value=4.0, value=0.5)
    att38 = st.slider('Att38', min_value=-4.0, max_value=4.0, value=0.5)
    att39 = st.slider('Att39', min_value=-4.0, max_value=4.0, value=0.5)
    att41 = st.slider('Att41', min_value=-4.0, max_value=4.0, value=0.5)
    att46 = st.slider('Att46', min_value=-4.0, max_value=4.0, value=0.5)
    att47 = st.slider('Att47', min_value=-4.0, max_value=4.0, value=0.5)
    att49 = st.slider('Att49', min_value=-4.0, max_value=4.0, value=0.5)
    att111 = st.slider('Att111', min_value=-4.0, max_value=4.0, value=0.5)
    att112 = st.slider('Att112', min_value=-4.0, max_value=4.0, value=0.5)
    att113 = st.slider('Att113', min_value=-4.0, max_value=4.0, value=0.5)
    att114 = st.slider('Att114', min_value=-4.0, max_value=4.0, value=0.5)
    att115 = st.slider('Att115', min_value=-4.0, max_value=4.0, value=0.5)
    att117 = st.slider('Att117', min_value=-4.0, max_value=4.0, value=0.5)
    att118 = st.slider('Att118', min_value=-4.0, max_value=4.0, value=0.5)
    att121 = st.slider('Att121', min_value=-4.0, max_value=4.0, value=0.5)
    att123 = st.slider('Att123', min_value=-4.0, max_value=4.0, value=0.5)
    att136 = st.slider('Att136', min_value=-4.0, max_value=4.0, value=0.5)
    att138 = st.slider('Att138', min_value=-4.0, max_value=4.0, value=0.5)
    att139 = st.slider('Att139', min_value=-4.0, max_value=4.0, value=0.5)
    att140 = st.slider('Att140', min_value=-4.0, max_value=4.0, value=0.5)
    att141 = st.slider('Att141', min_value=-4.0, max_value=4.0, value=0.5)
    att142 = st.slider('Att142', min_value=-4.0, max_value=4.0, value=0.5)
    att143 = st.slider('Att143', min_value=-4.0, max_value=4.0, value=0.5)
    att144 = st.slider('Att144', min_value=-4.0, max_value=4.0, value=0.5)
    att145 = st.slider('Att145', min_value=-4.0, max_value=4.0, value=0.5)
    att146 = st.slider('Att146', min_value=-4.0, max_value=4.0, value=0.5)
    att147 = st.slider('Att147', min_value=-4.0, max_value=4.0, value=0.5)
    att148 = st.slider('Att148', min_value=-4.0, max_value=4.0, value=0.5)
    att149 = st.slider('Att149', min_value=-4.0, max_value=4.0, value=0.5)
    att151 = st.slider('Att151', min_value=-4.0, max_value=4.0, value=0.5) 

    
