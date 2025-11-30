from fastapi import FastAPI, HTTPException
from pydantic import ValidationError
import joblib
import pandas as pd
import numpy as np
from models import PredictRequest

app = FastAPI(title="Income XGB Predict API")

# تحميل المكونات مرة واحدة عند بدء التطبيق
preprocessor = joblib.load("preprocessor.pkl")
scaler = joblib.load("scaler.pkl")          # إن كان موجوداً ومستخدمًا
xgb_model = joblib.load("xgb_model.pkl")

# الأعمدة المتوقعة بالترتيب — عدّلها لتطابق بالضبط ما استعملته أثناء التدريب
EXPECTED_COLS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country"
]

def prepare_input(data: dict) -> pd.DataFrame:
    """
    يحول dict إلى DataFrame ثم يطبّق preprocessor وscaler كما في التدريب.
    نفترض أن preprocessor يتعامل مع القيم الفارغة أو أن لديك Imputer فيه.
    """
    # تأكد من وجود كل الأعمدة — إذا غاب عمود ضع قيمة None
    row = {col: data.get(col) for col in EXPECTED_COLS}
    df = pd.DataFrame([row], columns=EXPECTED_COLS)

    # إذا استخدمت scaler منفصل لتعداد الأعمدة الرقمية:
    # أ) إذا كان preprocessor يتوقع raw cols ثم يقوم بتحويل عددي، طبّق preprocessor أولاً.
    # ب) أو إذا لديك pipeline موحّد (preprocessor+scaler) فاستعمله مباشرة.
    try:
        X_pre = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {e}")

    # لو كان scaler منفصل بعد preprocessor (تأكد من سير التدريب)
    try:
        X_scaled = scaler.transform(X_pre)
    except Exception:
        # لو لم يكن هناك scaler منفصل أو preprocessor عاد مصفوفة جاهزة، استخدم X_pre
        X_scaled = X_pre

    return X_scaled

@app.post("/predict")
def predict(payload: PredictRequest):
    # تحويل النموذج Pydantic إلى dict ثم تجهيز للدخول للموديل
    input_dict = payload.dict()
    X = prepare_input(input_dict)

    # التنبؤ
    try:
        proba = xgb_model.predict_proba(X)[0]  # [prob_for_class0, prob_for_class1]
        pred_class = xgb_model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    response = {
        "prediction": str(pred_class),
        "probability": {
            "class_0": float(proba[0]),
            "class_1": float(proba[1])
        }
    }
    return response

@app.get("/health")
def health():
    return {"status": "ok"}
