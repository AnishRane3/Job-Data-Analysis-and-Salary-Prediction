# -*- coding: utf-8 -*-
"""
Created on Sun May 19 23:09:58 2024

@author: ranea
"""

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

# Define the home route to render the HTML form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        input_data = request.form.to_dict()

        columns = [
            'Size', 'Rating', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 
            'Per_Hour', 'Employer_Provided', 'Location_States', 'Python_Req', 'Sql_Req', 
            'Spark_Req', 'Aws_Req', 'Excel_Req', 'Hadoop_Req', 'Job_Roles', 'Job_levels', 
            'Desc_Len', 'Num_Competitors'
        ]

        numerical_features = [
            'Rating', 'Per_Hour', 'Employer_Provided', 'Python_Req', 'Sql_Req', 
            'Spark_Req', 'Aws_Req', 'Excel_Req', 'Hadoop_Req', 'Desc_Len', 'Num_Competitors'
        ]

        for key, value in input_data.items():
            if key in numerical_features:
                if '.' in value:
                    input_data[key] = float(value)
                else:
                    input_data[key] = int(value)
            else:
                input_data[key] = value

        for column in columns:
            if column not in input_data:
                input_data[column] = 0 if column in numerical_features else 'Unknown'

        input_df = pd.DataFrame([input_data], columns=columns)

        prediction = model.predict(input_df)
        prediction = np.round(prediction, 2)

        return render_template('index.html', prediction=prediction[0])

    return render_template('index.html', prediction=None)

@app.route('/explore.html')
def explore():
    return render_template('explore.html')

if __name__ == '__main__':
    app.run(debug=True)
