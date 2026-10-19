
from flask import Flask, render_template, request
import os
import pickle
import pandas as pd

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "churn_model.pkl")

# Load the saved preprocessor
with open(PREPROCESSOR_PATH, "rb") as file:
    preprocessor = pickle.load(file)

# Load the trained model
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# Features used by the model, in the original training order
FEATURES = [
    "gender",
    "SeniorCitizen",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "TechSupport",
    "StreamingTV",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
]

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "Contract": [
        "Month-to-month",
        "One year",
        "Two year",
    ],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            row = {}

            # Read categorical inputs
            for feature, options in CATEGORICAL_OPTIONS.items():
                value = request.form.get(feature, "").strip()

                if value not in options:
                    raise ValueError(
                        f"Please select a valid value for {feature}."
                    )

                row[feature] = value

            # Read senior citizen status
            senior_citizen = request.form.get(
                "SeniorCitizen", ""
            ).strip()

            if senior_citizen not in ["0", "1"]:
                raise ValueError(
                    "Please select senior citizen status."
                )

            row["SeniorCitizen"] = int(senior_citizen)

            # Read numeric inputs
            for feature in ["tenure", "MonthlyCharges"]:
                raw_value = request.form.get(feature, "").strip()

                if not raw_value:
                    raise ValueError(
                        f"Please enter {feature.replace('_', ' ')}."
                    )

                value = float(raw_value)

                if value < 0:
                    raise ValueError(
                        f"{feature.replace('_', ' ').capitalize()} "
                        "cannot be negative."
                    )

                row[feature] = value

            if row["tenure"] > 72:
                raise ValueError(
                    "Tenure should be between 0 and 72 months."
                )

            # Create input DataFrame in training feature order
            input_df = pd.DataFrame(
                [[row[feature] for feature in FEATURES]],
                columns=FEATURES,
            )

            # Apply the saved preprocessing
            transformed = preprocessor.transform(input_df)

            # Preserve feature names if the model was trained
            # using a DataFrame with these names
            try:
                transformed_columns = (
                    preprocessor.get_feature_names_out()
                )
                transformed_df = pd.DataFrame(
                    transformed,
                    columns=transformed_columns,
                )
            except (AttributeError, ValueError):
                transformed_df = transformed

            # Make prediction
            result = int(model.predict(transformed_df)[0])

            prediction = (
                "Likely to churn"
                if result == 1
                else "Likely to stay"
            )

            # Calculate churn probability when supported
            if hasattr(model, "predict_proba"):
                probability = round(
                    float(
                        model.predict_proba(transformed_df)[0][1]
                    ) * 100,
                    2,
                )

        except ValueError as exc:
            error = str(exc)

        except Exception as exc:
            app.logger.exception("Prediction failed")
            error = (
                "Prediction failed. Check that your saved "
                "model and preprocessor match your notebook."
            )

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error,
        categorical_options=CATEGORICAL_OPTIONS,
        senior_citizen_options=[0, 1],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
