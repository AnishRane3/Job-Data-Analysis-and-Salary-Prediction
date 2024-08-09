# -*- coding: utf-8 -*-
"""
Created on Tue May 14 22:15:34 2024

@author: ranea
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
df = pd.read_csv('eda.csv')

# Select relevant features
df_model = df[['Average_Salary', 'Size', 'Rating', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Per_Hour', 'Employer_Provided', 'Location_States', 'Python_Req', 'Sql_Req', 'Spark_Req', 'Aws_Req', 'Excel_Req', 'Hadoop_Req', 'Job_Roles', 'Job_levels', 'Desc_Len', 'Num_Competitors']]

# Define features and target variable
X = df_model.drop('Average_Salary', axis=1)
y = df_model['Average_Salary']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define categorical and numerical features
categorical_features = ['Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Location_States', 'Job_Roles', 'Job_levels']
numerical_features = ['Rating', 'Per_Hour', 'Employer_Provided', 'Python_Req', 'Sql_Req', 'Spark_Req', 'Aws_Req', 'Excel_Req', 'Hadoop_Req', 'Desc_Len', 'Num_Competitors']

# Preprocessing pipeline
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append regressor to preprocessing pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', XGBRegressor(objective='reg:squarederror'))])

# Define hyperparameters for grid search
param_grid = {
    'regressor__n_estimators': [100, 200, 300],  
    'regressor__max_depth': [3, 6, 9],   
    'regressor__learning_rate': [0.01, 0.1, 0.2],  
    'regressor__subsample': [0.6, 0.8, 1.0] 
}

# Grid search
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error')
grid_search.fit(X_train, y_train)

# Get the best model
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
print(best_model)

r2 = r2_score(y_test, y_pred)
print("R-squared (coefficient of determination):", r2)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red')  # Diagonal line
plt.xlabel('Actual Average Salary')
plt.ylabel('Predicted Average Salary')
plt.title('Actual vs Predicted Average Salary')
plt.grid(True)
plt.show()

joblib.dump(best_model, 'best_model.pkl')
    
