import gradio as gr
import numpy as np
import joblib

# Load model, scaler, and column order
model = joblib.load("loan_rf_model.pkl")  # Or switch to "loan_lr_model.pkl"
scaler = joblib.load("scaler.pkl")

with open("column_order.txt", "r") as f:
    column_order = f.read().splitlines()

# Predict function
def predict_loan_eligibility(age, applicant_income, coapplicant_income, loan_amount, credit_score,
                             dependents, loan_term, gender, married, education, self_employed, property_area):

    # Manual feature engineering
    total_income = applicant_income + coapplicant_income
    loan_income_ratio = loan_amount / (total_income + 1)

    # One-hot encoding simulation
    input_dict = {
        "Dependents": dependents,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_score": credit_score,
        "Total_Income": total_income,
        "Loan_Income_Ratio": loan_income_ratio,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Married_Yes": 1 if married == "Yes" else 0,
        "Education_Not Graduate": 1 if education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if self_employed == "Yes" else 0,
        "Property_Area_Semiurban": 1 if property_area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if property_area == "Urban" else 0
    }

    # Arrange in correct order
    input_data = np.array([input_dict[col] for col in column_order]).reshape(1, -1)
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    return "✅ Approved" if prediction == 1 else "❌ Not Approved"

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("🏦 **Loan Eligibility Predictor**")

    with gr.Row():
        age = gr.Number(label="Age", value=30)
        applicant_income = gr.Number(label="Applicant Income")
        coapplicant_income = gr.Number(label="Coapplicant Income")
        loan_amount = gr.Number(label="Loan Amount")
        credit_score = gr.Number(label="Credit Score")

    with gr.Row():
        dependents = gr.Number(label="Dependents", value=0)
        loan_term = gr.Number(label="Loan Amount Term", value=360)
        gender = gr.Dropdown(["Male", "Female"], label="Gender")
        married = gr.Dropdown(["Yes", "No"], label="Married")
        education = gr.Dropdown(["Graduate", "Not Graduate"], label="Education")
        self_employed = gr.Dropdown(["Yes", "No"], label="Self Employed")
        property_area = gr.Dropdown(["Urban", "Rural", "Semiurban"], label="Property Area")

    predict_btn = gr.Button("Predict")
    result = gr.Textbox(label="Loan Status")

    predict_btn.click(
        predict_loan_eligibility,
        inputs=[age, applicant_income, coapplicant_income, loan_amount, credit_score,
                dependents, loan_term, gender, married, education, self_employed, property_area],
        outputs=result
    )

# Launch the app
app.launch()
