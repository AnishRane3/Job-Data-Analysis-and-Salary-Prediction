# -*- coding: utf-8 -*-
"""
Created on Fri May 10 22:22:47 2024

@author: ranea
"""

import pandas as pd

df = pd.read_csv('data_scraped.csv')

#Salary :
df['Per_Hour'] = df['Salary Estimate'].apply(lambda x: 1 if 'Per Hour' in x else 0)     
df['Employer_Provided'] = df['Salary Estimate'].apply(lambda x: 1 if 'Employer Provided Salary' in x else 0) 
df = df[df['Salary Estimate'] != '-1']
remove_Glassdoor_est = df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'\s*\(Glassdoor est.\)', '', regex=True)
remove_Employer_est = df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'\s*\(Employer est.\)', '', regex=True)
remove_dollar_k = df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'[$K,]', '', regex=True)
remove_per_hour = df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'\sPer\s(Hour)?', '', regex=True)
remove_employer_provided_salary = df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'^Employer Provided Salary:\s*', '', regex=True)
df['Min_Salary'] = remove_employer_provided_salary.apply(lambda x: int(x.split('-')[0]))
df['Max_Salary'] = remove_employer_provided_salary.apply(lambda x: int(x.split('-')[1]))
df['Average_Salary'] = (df.Min_Salary + df.Max_Salary)/2

#Company :
df['Company_Name_Txt'] = df.apply( lambda x : x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)
df['Years_of_Company'] = df['Founded'].apply(lambda x : x if x < 1 else 2024 - x)

#State :
df = df[df['Headquarters'] != '-1']
df['Location_States'] = df['Location'].apply(lambda x : x.split(',')[1])
#print(df.Location_States.value_counts())
df['Same_State'] = df.apply(lambda x : 1 if x['Location'] == x['Headquarters'] else 0, axis=1)

# req from Description
df['Python_Req'] = df['Job Description'].apply(lambda x : 1 if 'python' in x.lower() else 0)
print(df.Python_Req.value_counts())
df['Matlab_Req'] = df['Job Description'].apply(lambda x : 1 if 'matlab' in x.lower() else 0)
print(df.Matlab_Req.value_counts())
df['R_Req'] = df['Job Description'].apply(lambda x : 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() or 'r programming' in x.lower() else 0)
print(df.R_Req.value_counts())
df['Sas_Req'] = df['Job Description'].apply(lambda x : 1 if 'sas' in x.lower() else 0)
print(df.Sas_Req.value_counts())
df['Sql_Req'] = df['Job Description'].apply(lambda x : 1 if 'sql' in x.lower() else 0)
print(df.Sql_Req.value_counts())
df['Spark_Req'] = df['Job Description'].apply(lambda x : 1 if 'spark' in x.lower() else 0)
print(df.Spark_Req.value_counts())
df['Aws_Req'] = df['Job Description'].apply(lambda x : 1 if 'aws' in x.lower() else 0)
print(df.Aws_Req.value_counts())
df['Excel_Req'] = df['Job Description'].apply(lambda x : 1 if 'excel' in x.lower()  else 0)
print(df.Excel_Req.value_counts())
df['Hadoop_Req'] = df['Job Description'].apply(lambda x : 1 if 'hadoop' in x.lower()  else 0)
print(df.Hadoop_Req.value_counts())
print(df.columns)
df_out = df.drop(['Unnamed: 0'], axis=1)




df_out.to_csv('job_cleaned_data.csv', index= False)


