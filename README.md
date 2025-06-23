# loan_eligibility_predictor
#  Loan Eligibility Predictor (Fintech ML App)

This project is a user-friendly machine learning web app that predicts whether a loan applicant is eligible for loan approval based on their financial and personal details.

It uses **Logistic Regression** and **Random Forest** models trained on a cleaned and preprocessed dataset, with a simple **Gradio interface** for real-time predictions.

---

## Problem Statement

Banks want to predict whether an applicant should be granted a loan.  
This tool helps automate that process using machine learning.

---

##  Objective

Build a classification model that predicts **loan approval** based on input features such as:
- Applicant Income
- Coapplicant Income
- Loan Amount
- Loan Term
- Credit History
- Marital status, education, employment, property area, etc.

---

## 
Features Used

| Feature                  | Description                             |
|--------------------------|-----------------------------------------|
| Dependents               | Number of dependents                    |
| ApplicantIncome          | Monthly income of the applicant         |
| CoapplicantIncome        | Monthly income of the coapplicant       |
| LoanAmount               | Requested loan amount                   |
| Loan_Amount_Term         | Duration of loan in months              |
| Credit_History           | Credit history: 1 = Yes, 0 = No         |
| Total_Income             | Combined income                         |
| Loan_Income_Ratio        | LoanAmount / Total_Income               |
| Gender_Male              | 1 = Male, 0 = Female                    |
| Married_Yes              | 1 = Married, 0 = Not married            |
| Education_Not Graduate   | 1 = Not graduate, 0 = Graduate          |
| Self_Employed_Yes        | 1 = Self-employed, 0 = No               |
| Property_Area_Semiurban | Location                                |
| Property_Area_Urban      | Location                                |

---

## 🛠 How to Use

###  Clone the repository

cd loan-eligibility-predictor
pip install -r requirements.txt
python loan_app.py




or 


---

##  Run in Google Colab

You can also run this project entirely in **Google Colab** with no local setup required.

### Google Colab Script (copy and paste into a notebook cell)


# Step 1: Install required libraries
!pip install pandas scikit-learn matplotlib gradio joblib

# Step 2: Upload your dataset and required files
from google.colab import files
uploaded = files.upload()

# Step 3: Load the data
import pandas as pd
df = pd.read_csv("loan_data.csv")  # or replace with your uploaded filename

# Step 4: Preprocess and Train Model
# (Copy the entire model training code here: missing handling, encoding, feature engineering,
# scaling, model training, saving scaler and model with joblib)

# Step 5: Evaluate Model (ROC Curve + Confusion Matrix)
# (Paste the ROC + Confusion Matrix code block here)

# Optional Step: Run a Gradio app inside Colab (if UI is needed)
import gradio as gr
# (Paste the Gradio UI code here from loan_app.py)

