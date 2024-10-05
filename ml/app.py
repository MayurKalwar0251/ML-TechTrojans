from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoders
model = joblib.load('best_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Load the dataset and preprocess
df = pd.read_csv('healthcare_dataset.csv')
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Days of Admission'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# List of features used for predictions
features = ['Age', 'Gender', 'Blood Type', 'Medical Condition', 'Admission Type', 'Insurance Provider', 'Hospital']

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert the data to a DataFrame
        input_data = pd.DataFrame({
            'Age': [data['Age']],
            'Gender': label_encoders['Gender'].transform([data['Gender']]),
            'Blood Type': label_encoders['Blood Type'].transform([data['Blood Type']]),
            'Medical Condition': label_encoders['Medical Condition'].transform([data['Medical Condition']]),
            'Admission Type': label_encoders['Admission Type'].transform([data['Admission Type']]),
            'Insurance Provider': label_encoders['Insurance Provider'].transform([data['Insurance Provider']]),
            'Hospital': label_encoders['Hospital'].transform([data['Hospital']])
        })

        # Make prediction
        prediction = model.predict(input_data)

        # Return the result as JSON
        return jsonify({'Predicted Billing Amount': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})
    

@app.route('/least_billing_hospitals', methods=['GET'])
def least_billing_hospitals():
    try:
        # Get the first 100 unique hospital names
        unique_hospitals = df['Hospital'].unique()[:100]
        
        # Filter the DataFrame to only include these hospitals
        filtered_df = df[df['Hospital'].isin(unique_hospitals)]
        
        # List to store hospital billing predictions
        hospital_billing_predictions = []

        for hospital in filtered_df['Hospital'].unique():
            hospital_df = filtered_df[filtered_df['Hospital'] == hospital]
            
            # Check if the hospital name is in the label encoder's classes
            if hospital in label_encoders['Hospital'].classes_:
                # Prepare the input data for prediction
                input_data = pd.DataFrame({
                    'Age': hospital_df['Age'],
                    'Gender': label_encoders['Gender'].transform(hospital_df['Gender']),
                    'Blood Type': label_encoders['Blood Type'].transform(hospital_df['Blood Type']),
                    'Medical Condition': label_encoders['Medical Condition'].transform(hospital_df['Medical Condition']),
                    'Admission Type': label_encoders['Admission Type'].transform(hospital_df['Admission Type']),
                    'Insurance Provider': label_encoders['Insurance Provider'].transform(hospital_df['Insurance Provider']),
                    'Hospital': label_encoders['Hospital'].transform([hospital])[0]  # Safely transform the hospital name
                })

                # Predict billing amount
                predicted_billing = model.predict(input_data)
                avg_billing = np.mean(predicted_billing)

                # Append the result to the predictions list
                hospital_billing_predictions.append({'Hospital': hospital, 'Average Predicted Billing Amount': avg_billing})

        # Convert to DataFrame and get the 5 hospitals with the least average billing amount
        hospital_billing_predictions_df = pd.DataFrame(hospital_billing_predictions)
        least_billing_hospitals = hospital_billing_predictions_df.nsmallest(5, 'Average Predicted Billing Amount')

        # Decode hospital names back to original
        least_billing_hospitals['Hospital'] = least_billing_hospitals['Hospital'].apply(
            lambda x: label_encoders['Hospital'].inverse_transform([x])[0] if x in label_encoders['Hospital'].classes_ else x
        )

        return least_billing_hospitals.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': str(e)})



@app.route('/least_days_hospitals', methods=['POST'])
def least_days_hospitals():
    try:
        unique_hospitals = df['Hospital'].unique()[1:100]
        
        # Filter the DataFrame to only include these hospitals
        filtered_df = df[df['Hospital'].isin(unique_hospitals)]
        
        # Calculate the average days of admission for each hospital
        hospital_days_of_admission = filtered_df.groupby('Hospital', as_index=False)['Days of Admission'].mean()
        hospital_days_of_admission.columns = ['Hospital', 'Average Days of Admission']

        # Handle unseen hospitals by transforming with label encoder
        hospital_days_of_admission['Hospital'] = hospital_days_of_admission['Hospital'].apply(
            lambda x: label_encoders['Hospital'].transform([x])[0] if x in label_encoders['Hospital'].classes_ else np.nan
        )

        # Get the 5 hospitals with the least average days of admission, excluding NaN values
        least_days_hospitals = hospital_days_of_admission.dropna().nsmallest(5, 'Average Days of Admission')

        # Decode hospital names back to original
        least_days_hospitals['Hospital'] = label_encoders['Hospital'].inverse_transform(least_days_hospitals['Hospital'].astype(int))

        return least_days_hospitals.to_json(orient='records')

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
