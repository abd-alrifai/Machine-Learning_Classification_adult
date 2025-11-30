# Adult Income Classification with Machine Learning

This project offers a binary rating system to predict whether an individual's income exceeds

50K سنوياً باستخدام بيانات تعداد البالغين (Adult Census Income Dataset).

The system relies on comparing several machine learning algorithms, then deploying the best model using FastAPI to provide instant predictions.

## Table of Contents

-   Overview\
-   Dataset\
-   Project Structure\
-   Installation\
-   Model Training\
-   API Deployment\
-   Model Performance\
-   Usage Examples\
-   Technologies Used\
-   Notes\
-   Contributing\
-   License\
-   Author\
-   Acknowledgments

## Overview

The project aims to build a classification model that predicts income level based on demographic characteristics. Four different models were trained:

1.  Logistic Regression\
2.  Random Forest Classifier\
3.  Support Vector Machine (SVM)\
4.  XGBoost Classifier

The XGBoost deployment model was chosen because it performed best.

## Dataset
The project relies on the Adult Census Income Dataset, which contains demographic data from the 1994 census.

### Features

-   age\
-   workclass\
-   fnlwgt\
-   education\
-   education_num\
-   marital_status\
-   occupation\
-   relationship\
-   race\
-   sex\
-   capital_gain\
-   capital_loss\
-   hours_per_week\
-   native_country

### Target Variable

- Income: Takes one of the two values ​​(<=50K or >50K)

## Project Structure

    ├── Machine_Learning_Classification_adult.ipynb
    ├── app.py
    ├── models.py
    ├── requirements.txt
    ├── adult_data.csv
    ├── preprocessor.pkl
    ├── scaler.pkl
    └── xgb_model.pkl

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/abd-alrifai/Machine-Learning_Classification_adult.git
cd Machine-Learning_Classification_adult
```

### 2. Create a virtual environment

``` bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

## Model Training

### Data Preprocessing

#### Data Cleaning

- Replace missing values ​​"?" with NaN\
- Delete rows containing missing values\
- Remove duplicate rowsة

#### Feature Engineering

Separating digital features from textual ones\

-   One-Hot Encoding\
-   StandardScaler

### Train-Test Split

- 90% training
- 10% testing
  
- random_state = 42

### Model Configuration

#### XGBoost

``` python
XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_lambda=1,
    random_state=42
)
```

#### Random Forest

``` python
RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)
```

## API Deployment

### Start server

``` bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

#### GET /health

``` json
{"status": "ok"}
```

#### POST /predict

Request:

``` json
{
  "age": 39,
  "workclass": "State-gov",
  "fnlwgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States"
}
```

Response:

``` json
{
  "prediction": "0",
  "probability": {
    "class_0": 0.85,
    "class_1": 0.15
  }
}
```

## Model Performance

  ---------------------------------------------------------------------------
  Model                   Accuracy   Precision   Recall   F1-Score   AUC
  ----------------------- ---------- ----------- -------- ---------- --------
  Logistic Regression     \~0.85     \~0.75      \~0.60   \~0.67     \~0.90

  Random Forest           \~0.87     \~0.78      \~0.65   \~0.71     \~0.92

  SVM (RBF)               \~0.85     \~0.76      \~0.61   \~0.68     \~0.91

  XGBoost                 \~0.87     \~0.79      \~0.66   \~0.72     \~0.93
  ---------------------------------------------------------------------------

## Usage Examples

### Python

``` python
import requests

url = "http://localhost:8000/predict"
payload = {
    "age": 45,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": "Masters",
    "education_num": 14,
    "marital_status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 15024,
    "capital_loss": 0,
    "hours_per_week": 50,
    "native_country": "United-States"
}
print(requests.post(url, json=payload).json())
```

### cURL

``` bash
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{...}'
```

## Technologies Used

-   Python 3.8+\
-   scikit-learn\
-   XGBoost\
-   pandas\
-   numpy\
-   matplotlib\
-   seaborn\
-   FastAPI\
-   joblib\
-   uvicorn

## Notes
Dealing with missing data

-   One-Hot Encoding\
-   StandardScaler\
  
Choosing XGBoost for publishing

## Contributing

Contributions can be made via Pull Request.

## License

Available for educational and research purposes.

## Author

Abd Alrifai\
GitHub: @abd-alrifai
Email: bodi12381@gmail.com

## Acknowledgments

Adult Census Income Dataset -- UCI Machine Learning Repository
