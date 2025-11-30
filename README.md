Adult Income Classification with Machine Learning
A complete machine learning project that predicts whether an individual's income exceeds $50K/year based on census data. The project includes data preprocessing, model training with multiple algorithms, and a FastAPI deployment for real-time predictions.
 Table of Contents

Overview
Dataset
Project Structure
Installation
Model Training
API Deployment
Model Performance
Usage Examples
Technologies Used

Overview
This project implements a binary classification system to predict income levels using the Adult Census Income dataset. Four different machine learning algorithms are trained and compared:

Logistic Regression
Random Forest Classifier
Support Vector Machine (SVM)
XGBoost Classifier

The best performing model (XGBoost) is then deployed via a FastAPI REST API for production use.
 Dataset
The Adult Census Income dataset contains demographic information from the 1994 Census database.
Features:

age: Age of the individual
workclass: Type of employment
fnlwgt: Final weight (census sampling weight)
education: Highest level of education
education_num: Numerical encoding of education
marital_status: Marital status
occupation: Type of occupation
relationship: Relationship status
race: Race
sex: Gender
capital_gain: Capital gains
capital_loss: Capital losses
hours_per_week: Hours worked per week
native_country: Country of origin

Target Variable:

income: Binary classification (<=50K or >50K)

Project Structure
├── Machine_Learning_Classification_adult.ipynb  # Main training notebook
├── app.py                                       # FastAPI application
├── models.py                                    # Pydantic models for API
├── requirements.txt                             # Python dependencies
├── adult_data.csv                              # Dataset
├── preprocessor.pkl                            # Saved preprocessing pipeline
├── scaler.pkl                                  # Saved feature scaler
└── xgb_model.pkl                               # Saved XGBoost model


Installation

Clone the repository:

bashgit clone https://github.com/abd-alrifai/Machine-Learning_Classification_adult.git
cd Machine-Learning_Classification_adult

Create a virtual environment (recommended):

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashpip install -r requirements.txt
Required libraries:
numpy
pandas
matplotlib
seaborn
scikit-learn
xgboost
joblib
fastapi
uvicorn
pydantic

🔧 Model Training
Data Preprocessing Steps:

Data Cleaning:

Replace missing values (marked as "?") with NaN
Drop rows with missing values
Remove duplicate entries


Feature Engineering:

Separate categorical and numerical features
One-Hot Encoding for categorical variables
Standard scaling for numerical features


Train-Test Split:

90% training data
10% testing data
Random state: 42 for reproducibility



Model Configuration:
XGBoost (Best Performer):
pythonXGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_lambda=1,
    random_state=42
)
Random Forest:
pythonRandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

  API Deployment
Starting the API Server:
bashuvicorn app:app --reload --host 0.0.0.0 --port 8000
API Endpoints:
1. Health Check
httpGET /health
Response:
json{
  "status": "ok"
}
2. Income Prediction
httpPOST /predict
Content-Type: application/json
Request body:
json{
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
Response:
json{
  "prediction": "0",
  "probability": {
    "class_0": 0.85,
    "class_1": 0.15
  }
}
 Model Performance
Evaluation Metrics:
ModelAccuracyPrecisionRecallF1-ScoreAUCLogistic Regression~0.85~0.75~0.60~0.67~0.90Random Forest~0.87~0.78~0.65~0.71~0.92SVM (RBF)~0.85~0.76~0.61~0.68~0.91XGBoost~0.87~0.79~0.66~0.72~0.93
Note: Exact metrics may vary based on the specific data split and preprocessing
ROC Curve Analysis:
The project includes ROC curve visualization comparing all four models, with XGBoost achieving the highest Area Under Curve (AUC), indicating superior classification performance.
 Usage Examples
Python API Request:
pythonimport requests

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

response = requests.post(url, json=payload)
print(response.json())
cURL Request:
bashcurl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
🛠️ Technologies Used

Python 3.8+
Machine Learning: scikit-learn, XGBoost
Data Processing: pandas, numpy
Visualization: matplotlib, seaborn
API Framework: FastAPI
Model Persistence: joblib
API Server: uvicorn

 Notes

The preprocessing pipeline handles missing values and unknown categories
All categorical features are one-hot encoded
Numerical features are standardized using StandardScaler
The XGBoost model is selected for deployment based on superior performance
Class imbalance is handled through model hyperparameter tuning

 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
 License
This project is available for educational and research purposes.
 Author
Abdulrhman Essam Alrifai

GitHub: @abd-alrifai

🙏 Acknowledgments

Dataset source: UCI Machine Learning Repository
Adult Census Income Dataset (1994 Census database)


For questions or issues, please open an issue on the GitHub repository.
