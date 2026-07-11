
import pandas as pd
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and LabelEncoder
model = joblib.load('model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define the expected columns in the order the model expects them
# This list should match the feature columns (x) used during training
expected_columns = [' no_of_dependents', ' education', ' self_employed', ' income_annum', ' loan_amount', ' loan_term', ' cibil_score', ' residential_assets_value', ' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])

        # Apply the same LabelEncoder transformations
        # Assuming ' education' and ' self_employed' were the columns encoded
        if ' education' in input_df.columns:
            input_df[' education'] = label_encoder.transform(input_df[' education'])
        if ' self_employed' in input_df.columns:
            input_df[' self_employed'] = label_encoder.transform(input_df[' self_employed'])

        # Ensure the input DataFrame has the same columns and order as the training data
        input_df = input_df[expected_columns]

        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        response = {
            'loan_status_prediction': int(prediction[0]), # 0 for rejected, 1 for approved
            'probability_rejected': prediction_proba[0][0],
            'probability_approved': prediction_proba[0][1]
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
