import joblib
import pandas as pd
import preprocess_util
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI()

class Prediction_Data(BaseModel):
    person_age: int = 24
    person_income: int = 37500
    person_home_ownership: str = "RENT"
    person_emp_length: float = 2.0
    loan_intent: str = "DEBTCONSOLIDATION"
    loan_grade: str = "c"
    loan_amnt: int = 1600
    loan_int_rate: float = 11.03
    loan_percent_income: float = 0.04
    cb_person_default_on_file: str = "Y"
    cb_person_cred_hist_length: int = 3

num_imputer = joblib.load("../models/preprocess_num_imputer_20221231.pkl")
cat_imputer = joblib.load("../models/preprocess_cat_imputer_20221231.pkl")
ohe_encoder = joblib.load("../models/preprocess_ohe_20221231.pkl")
scaler = joblib.load("../models/preprocess_scaler_20221231.pkl")
model = joblib.load("../models/best_model.pkl")

@app.post("/predict/")
def predict_data(data: Prediction_Data):
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)

    data = preprocess_util.transform_preprocess_data(
        X = data,
        num_imputer = num_imputer,
        cat_imputer = cat_imputer,
        ohe_encoder = ohe_encoder,
        scaler = scaler
    )

    pred = model.predict(data)

    if(pred[0] == 1):
        return {"result": "Default", "error_msg": ""}
    
    elif(pred[0] == 0):
        return {"result": "Non Default", "error_msg": ""}
    
    else:
        return HTTPException(status_code = 400, detail = "Price must be greater than zero.")

@app.get("/health")
def health_check():
    return {"result": "OK", "error_msg": ""}
