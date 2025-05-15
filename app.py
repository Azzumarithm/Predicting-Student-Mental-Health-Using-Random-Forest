from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load Random Forest model and preprocessing artifacts
model = joblib.load("random_forest_model.pkl")  # Load the Random Forest model
label_encoders = joblib.load("label_encoders.pkl")
imputer = joblib.load("imputer.pkl")

# Define expected columns (in order)
expected_columns = [
    "Choose your gender",
    "Age",
    "What is your course?",
    "Your current year of Study",
    "What is your CGPA?",
    "Marital status",
    "Do you have Depression?",
    "Do you have Anxiety?",
    "Do you have Panic attack?",
    "Did you seek any specialist for a treatment?",
]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Collect input in the correct order
        input_data = [request.form[col] for col in expected_columns]
        input_df = pd.DataFrame([input_data], columns=expected_columns)

        # Debug: Print the raw input data
        print("Raw Input Data:", input_df)

        # Label encode categorical columns using saved encoders
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                input_df[col] = le.transform(input_df[col].astype(str))
                # Debug: Print the encoded values for each column
                print(f"Encoded {col}: {input_df[col].values}")
            else:
                # Convert to numeric where no encoder exists (e.g., Age, CGPA)
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
                # Debug: Print the numeric conversion for non-categorical columns
                print(f"Numeric {col}: {input_df[col].values}")

        # Debug: Print the DataFrame after encoding and numeric conversion
        print("DataFrame After Encoding and Conversion:", input_df)

        # Impute missing values (if any)
        input_data_imputed = imputer.transform(input_df)

        # Debug: Print the imputed data
        print("Imputed Data:", input_data_imputed)

        # Predict using the trained model
        prediction = model.predict(input_data_imputed)[0]

        # Debug: Print the final input to the model and the prediction
        print("Final Input to Model:", input_data_imputed)
        print("Model Prediction:", prediction)

        result = (
            "Might Have Mental Health Issue"
            if prediction == 1
            else "Might Not Have Mental Health Issue"
        )

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # frontend sends JSON
        print("Received Data:", data)

        # Create a DataFrame from the received data
        input_df = pd.DataFrame([data], columns=expected_columns)

        # Debug: Print the raw input data
        print("Raw Input Data:", input_df)

        # Exclude target-related columns to match training features
        columns_to_exclude = [
            "Do you have Depression?",
            "Do you have Anxiety?",
            "Do you have Panic attack?",
        ]
        input_df = input_df.drop(columns=columns_to_exclude)

        # Debug: Print the DataFrame after dropping excluded columns
        print("DataFrame After Dropping Excluded Columns:", input_df)

        # Label encode categorical fields
        for col in input_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                try:
                    # Attempt to transform the column
                    input_df[col] = le.transform(input_df[col].astype(str))
                except ValueError as e:
                    # Handle unseen or invalid categories
                    print(f"ValueError for column {col}: {e}")
                    input_df[col] = np.nan  # Assign NaN for invalid categories
                # Debug: Print the encoded values for each column
                print(f"Encoded {col}: {input_df[col].values}")
            else:
                # Convert numerics to float (skip encoding for numeric columns)
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
                # Debug: Print the numeric conversion for non-categorical columns
                print(f"Numeric {col}: {input_df[col].values}")

        # Debug: Print the DataFrame after encoding and numeric conversion
        print("DataFrame After Encoding and Conversion:", input_df)

        # Impute missing values (if any)
        input_data_imputed = imputer.transform(input_df)

        # Debug: Print the imputed data
        print("Imputed Data:", input_data_imputed)

        # Predict using the trained Random Forest model
        proba = model.predict_proba(input_data_imputed)[0][1]  # Probability of class 1
        prediction = 1 if proba >= 0.95 else 0  # Apply 95% confidence threshold

        result = {
            "label": (
                "Might have Mental Health Issue"
                if prediction == 1
                else "Might Not have Mental Health Issue"
            ),
            "probability": round(proba * 100, 2),
        }
        print("Model Prediction:", prediction)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
