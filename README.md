# Semiconductor-Abnormality-Classification

## About 
This project aims to classify abnormality using sensor data collected during semiconductor fabrication stage. The best tuned RandomForestClassifier (F1-score of 99.6%) is deployed into a Streamlit app where users could make predictions interactively and download the output results. 

## Summary 
Data visualization | Model building | Model deployment 

## How to run? 
**Run locally**
1. Create and launch new Anaconda Python environment. 
```
conda env list
conda create --name streamlit python=3.10.5
conda activate <environment-name>
```
2. Install required packages. `pip install -r requirements.txt`
3. Run Streamlit dashboard. `streamlit run dashboard.py`

**Deploy (Streamlit)**
1. Sign up for a Streamlit account. 
2. Run dashboard. 

## Sample results/ output
![](https://github.com/CH2001/Employee-churn-classification/blob/main/Image/Demo.gif)

## Links 
Dataset: [Semiconductor abnormally dataset dataset](https://www.timeseriesclassification.com/description.php?Dataset=Wafer) <br> 
Dashboard: [Streamlit dashboard](https://semiconductor-abnormality-classificationc.streamlit.app/)
