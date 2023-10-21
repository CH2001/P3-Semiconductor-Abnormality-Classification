import pandas as pd
from scipy.io import arff

st.header("Semiconductor abnormality classification")

def import_dataset(dataset): 
    with open(dataset, 'r', encoding='utf-8') as file:
        raw_data, meta = arff.loadarff(file)

    df = pd.DataFrame(raw_data)
    return df
