import pandas as pd
import joblib

saved_package = joblib.load("models/credit_default_model.joblib")
model = saved_package["model"]
threshold = saved_package["threshold"]

new_customer = pd.DataFrame([{
    "LIMIT_BAL": 50000,
    "SEX": 2,
    "EDUCATION": 2,
    "MARRIAGE": 1,
    "AGE": 30,
    "PAY_0": 2,
    "PAY_2": 1,
    "PAY_3": 0,
    "PAY_4": 0,
    "PAY_5": 0,
    "PAY_6": 0,
    "BILL_AMT1": 30000,
    "BILL_AMT2": 28000,
    "BILL_AMT3": 25000,
    "BILL_AMT4": 20000,
    "BILL_AMT5": 15000,
    "BILL_AMT6": 10000,
    "PAY_AMT1": 1000,
    "PAY_AMT2": 1000,
    "PAY_AMT3": 1000,
    "PAY_AMT4": 1000,
    "PAY_AMT5": 1000,
    "PAY_AMT6": 1000
}])

risk_probability = model.predict_proba(new_customer)[0, 1]
risk_flag = risk_probability >= threshold

print("Threshold used:", threshold)
print("Default-risk probability:", round(risk_probability, 3))
print("Flag for review:", risk_flag)
