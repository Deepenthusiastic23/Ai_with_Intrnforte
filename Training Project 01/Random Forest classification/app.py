from flask import Flask, render_template, request
import joblib
import pandas as pd


# ============================================================
# 1. CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

MODEL_PATH = "models/random_forest_churn.pkl"

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# 3. HOME ROUTE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# 4. PREDICTION ROUTE
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # Get values from HTML form

    gender = request.form["gender"]

    senior_citizen = int(
        request.form["SeniorCitizen"]
    )

    partner = request.form["Partner"]

    dependents = request.form["Dependents"]

    tenure = int(
        request.form["tenure"]
    )

    phone_service = request.form["PhoneService"]

    multiple_lines = request.form["MultipleLines"]

    internet_service = request.form["InternetService"]

    online_security = request.form["OnlineSecurity"]

    online_backup = request.form["OnlineBackup"]

    device_protection = request.form["DeviceProtection"]

    tech_support = request.form["TechSupport"]

    streaming_tv = request.form["StreamingTV"]

    streaming_movies = request.form["StreamingMovies"]

    contract = request.form["Contract"]

    paperless_billing = request.form["PaperlessBilling"]

    payment_method = request.form["PaymentMethod"]

    monthly_charges = float(
        request.form["MonthlyCharges"]
    )

    total_charges = float(
        request.form["TotalCharges"]
    )


    # ========================================================
    # CREATE INPUT DATAFRAME
    # ========================================================

    input_data = pd.DataFrame(
        {
            "gender": [gender],

            "SeniorCitizen": [
                senior_citizen
            ],

            "Partner": [
                partner
            ],

            "Dependents": [
                dependents
            ],

            "tenure": [
                tenure
            ],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless_billing
            ],

            "PaymentMethod": [
                payment_method
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ]
        }
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = model.predict(
        input_data
    )[0]


    # ========================================================
    # PROBABILITY
    # ========================================================

    probability = model.predict_proba(
        input_data
    )[0][1]


    churn_probability = round(
        probability * 100,
        2
    )


    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        result = "Customer is likely to CHURN."

    else:

        result = "Customer is likely to STAY."


    return render_template(
        "result.html",
        prediction=result,
        probability=churn_probability
    )


# ============================================================
# 5. RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )