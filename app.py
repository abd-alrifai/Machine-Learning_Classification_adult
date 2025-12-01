from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from models import PredictRequest

app = FastAPI(title="Income XGB Predict API")

# تحميل المكونات مرة واحدة عند بدء التطبيق
preprocessor = joblib.load("preprocessor.pkl")
scaler = joblib.load("scaler.pkl")
xgb_model = joblib.load("xgb_model.pkl")


def prepare_input(data: dict):
    """
    تحويل أسماء الأعمدة من JSON إلى الأسماء الأصلية
    ثم تمريرها إلى preprocessor و scaler
    """

    COLUMN_RENAME_MAP = {
        "education_num": "education-num",
        "marital_status": "marital-status",
        "capital_gain": "capital-gain",
        "capital_loss": "capital-loss",
        "hours_per_week": "hours-per-week",
        "native_country": "native-country"
    }

    # تحويل أسماء الحقول
    fixed_data = {}
    for k, v in data.items():
        fixed_key = COLUMN_RENAME_MAP.get(k, k)
        fixed_data[fixed_key] = v

    # تحويل إلى DataFrame
    df = pd.DataFrame([fixed_data])

    # تطبيق الـ preprocessor
    try:
        X_pre = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {e}")

    # تطبيق الـ scaler إن وُجد
    try:
        X_scaled = scaler.transform(X_pre)
    except Exception:
        X_scaled = X_pre

    return X_scaled


@app.post("/predict")
def predict(payload: PredictRequest):
    input_dict = payload.dict()
    X = prepare_input(input_dict)

    try:
        proba = xgb_model.predict_proba(X)[0]
        pred_class = xgb_model.predict(X)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    label_map = {
    0: "<=50K",
    1: ">50K"
       }

    response = {
    "prediction": label_map.get(int(pred_class), str(pred_class)),
    "probability": {
        "<=50K": float(proba[0]),
        ">50K": float(proba[1])
    }
}

    return response


@app.get("/health")
def health():
    return {"status": "ok"}
