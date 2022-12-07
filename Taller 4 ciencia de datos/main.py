# Dependencies
from flask import Flask, render_template, render_template_string
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from flask import request
import random
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
import numpy as np
from sklearn import pipeline
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
import joblib
import requests

# Your API definition
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def get_plot():
 if request.method == 'POST': 
  gender= str(request.form['gender'])
  SeniorCitizen=int(request.form['SeniorCitizen'])
  Partner=str(request.form['Partner'])
  Dependents=str(request.form['Dependents'])
  tenure=int(request.form['tenure'])
  PhoneService=str(request.form['PhoneService'])
  MultipleLines=str(request.form['MultipleLines'])
  InternetService=str(request.form['InternetService'])
  OnlineSecurity=str(request.form['OnlineSecurity'])
  OnlineBackup=str(request.form['OnlineBackup'])
  DeviceProtection=str(request.form['DeviceProtection'])
  TechSupport=str(request.form['TechSupport'])
  StreamingTV=str(request.form['StreamingTV'])
  StreamingMovies=str(request.form['StreamingMovies'])
  Contract=str(request.form['Contract'])
  PaperlessBilling=str(request.form['PaperlessBilling'])
  PaymentMethod=str(request.form['PaymentMethod'])
  MonthlyCharges=int(request.form['MonthlyCharges'])
  TotalCharges=float(request.form['TotalCharges'])
  list=[['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
  df=pd.DataFrame(list, columns=['gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges'])
  #Ajustar valores irreglulares y definir tipos de datos para cada una de las features
  df.loc[df['TotalCharges'] == '', 'TotalCharges'] = 'NaN'
  #df["TotalCharges"] = df["TotalCharges"].astype("float64")
  df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
  column_names={'gender':0, 'SeniorCitizen':1, 'Partner':2, 'Dependents':3, 'tenure':4,
       'PhoneService':5, 'MultipleLines':6, 'InternetService':7, 'OnlineSecurity':8,
       'OnlineBackup':9, 'DeviceProtection':10, 'TechSupport':11, 'StreamingTV':12,
       'StreamingMovies':13, 'Contract':14, 'PaperlessBilling':15, 'PaymentMethod':16,
       'MonthlyCharges':17, 'TotalCharges':18}
  df=df.rename(columns=column_names)
  numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', StandardScaler())])
  categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant'))
      ,('encoder', OrdinalEncoder())])
  numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
  categorical_features = df.select_dtypes(include=['object','int32']).columns
  preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_features),('categorical', categorical_transformer, categorical_features)])
  X_test=pd.DataFrame(preprocessor.fit_transform(df))
  model=joblib.load('best_grid_search_pipeline.pkl')
  y_pred=model.predict(X_test)
  y_pred_probs=model.predict_proba(X_test)
  final_results = pd.DataFrame()
  y_pred_probs=y_pred_probs[:, 1]
  final_results['predictions'] = y_pred
  final_results["propensity_to_churn(%)"] = y_pred_probs
  final_results["propensity_to_churn(%)"] = final_results["propensity_to_churn(%)"]*100
  final_results["propensity_to_churn(%)"]=final_results["propensity_to_churn(%)"].round(2)

  return render_template('INDIVIDUAL.html', tables=[final_results.to_html(classes='table table-stripped')])
 else:
  return render_template('INDIVIDUAL.html')

@app.route('/DOCUMENTO', methods=['GET', 'POST'])
def get_plot2():
 if request.method == 'POST': 
     df= pd.read_json(request.form['archivosubido'])
     model=joblib.load('best_grid_search_pipeline.pkl')
     #Ajustar valores irreglulares y definir tipos de datos para cada una de las features
     df.loc[df['TotalCharges'] == '', 'TotalCharges'] = 'NaN'
     df["TotalCharges"] = df["TotalCharges"].astype("float64")
     df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
     ID=df['customerID'].to_frame()
     df=df.drop('customerID', axis=1)
     column_names={'gender':0, 'SeniorCitizen':1, 'Partner':2, 'Dependents':3, 'tenure':4,
       'PhoneService':5, 'MultipleLines':6, 'InternetService':7, 'OnlineSecurity':8,
       'OnlineBackup':9, 'DeviceProtection':10, 'TechSupport':11, 'StreamingTV':12,
       'StreamingMovies':13, 'Contract':14, 'PaperlessBilling':15, 'PaymentMethod':16,
       'MonthlyCharges':17, 'TotalCharges':18}
     df=df.rename(columns=column_names)
     numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', StandardScaler())])
     categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant'))
      ,('encoder', OrdinalEncoder())])
     numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
     categorical_features = df.select_dtypes(include=['object','int32']).columns
     preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_features),('categorical', categorical_transformer, categorical_features)])
     X_test=pd.DataFrame(preprocessor.fit_transform(df))
     model=joblib.load('best_grid_search_pipeline.pkl')
     y_pred=model.predict(X_test)
     y_pred_probs=model.predict_proba(X_test)
     final_results = pd.concat([ID], axis = 1)
     #final_results = pd.DataFrame()
     y_pred_probs=y_pred_probs[:, 1]
     final_results['predictions'] = y_pred
     final_results["propensity_to_churn(%)"] = y_pred_probs
     final_results["propensity_to_churn(%)"] = final_results["propensity_to_churn(%)"]*100
     final_results["propensity_to_churn(%)"]=final_results["propensity_to_churn(%)"].round(2)
     return render_template('DOCUMENTO.html', tables=[final_results.to_html(classes='table table-stripped')])
 else:
  return render_template('DOCUMENTO.html')


@app.route('/ENTRENAMIENTO', methods=['GET', 'POST'])
def get_plot3():
 if request.method == 'POST': 
     df= pd.read_json(request.form['archivosubido'])
     model=joblib.load('best_grid_search_pipeline.pkl')
     #Ajustar valores irreglulares y definir tipos de datos para cada una de las features
     df.loc[df['TotalCharges'] == '', 'TotalCharges'] = 'NaN'
     df["TotalCharges"] = df["TotalCharges"].astype("float64")
     df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
     ID=df['customerID'].to_frame()
     df=df.drop('customerID', axis=1)
     column_names={'gender':0, 'SeniorCitizen':1, 'Partner':2, 'Dependents':3, 'tenure':4,
       'PhoneService':5, 'MultipleLines':6, 'InternetService':7, 'OnlineSecurity':8,
       'OnlineBackup':9, 'DeviceProtection':10, 'TechSupport':11, 'StreamingTV':12,
       'StreamingMovies':13, 'Contract':14, 'PaperlessBilling':15, 'PaymentMethod':16,
       'MonthlyCharges':17, 'TotalCharges':18, 'Churn':19}
     df=df.rename(columns=column_names)
     numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', StandardScaler())])
     categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant'))
      ,('encoder', OrdinalEncoder())])
     numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
     categorical_features = df.select_dtypes(include=['object','int32']).columns
     preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_features),('categorical', categorical_transformer, categorical_features)])
     X_test=pd.DataFrame(preprocessor.fit_transform(df))
     model=joblib.load('best_grid_search_pipeline.pkl')
     df2=pd.DataFrame(preprocessor.fit_transform(df))
     X_train, X_test, y_train, y_test = train_test_split(df2.iloc[:,:-1].values,df2[19],test_size = 0.4,random_state = 10)
     pipe_lr = Pipeline([('clf', LogisticRegression(random_state=42))])
     pipe_dt = Pipeline([('model', DecisionTreeClassifier(random_state=42))])
     pipe_rf = Pipeline([('clf', RandomForestClassifier(random_state=42))])
     pipe_svm = Pipeline([('clf', svm.SVC(random_state=42))])
     jobs = -1
     param_range = [9, 10]
     param_range_fl = [1.0, 0.5]
     grid_params_lr = [{'clf__penalty': ['l1', 'l2'],
        'clf__C': param_range_fl,
        'clf__solver': ['liblinear']}] 
     grid_params_rf = [{'clf__criterion': ['gini', 'entropy'],
        'clf__max_depth': param_range,
        'clf__min_samples_split': param_range[1:]}]
     grid_params_svm = [{'clf__kernel': ['linear', 'rbf'], 
        'clf__C': param_range}]
     LR = GridSearchCV(estimator=pipe_lr,
            param_grid=grid_params_lr,
            scoring='roc_auc',
            cv=10)
     RF = GridSearchCV(estimator=pipe_rf,
            param_grid=grid_params_rf,
            scoring='roc_auc',
            cv=10, 
            n_jobs=jobs)
     SVM = GridSearchCV(estimator=pipe_svm,
            param_grid=grid_params_svm,
            scoring='roc_auc',
            cv=10,
            n_jobs=jobs)
      # List of pipelines for iterating through each of 
     grids = [LR,RF,SVM]

     grid_dict = {0: 'Logistic Regression', 
        1: 'Random Forest',
        2: 'Support Vector Machine'}
     #Fit the grid search objects
     best_auc = 0.0
     best_clf = 0
     best_gs = ''
     for idx, gs in enumerate(grids):
          gs.fit(X_train, y_train)
          y_pred = gs.predict(X_test)
          if roc_auc_score(y_test, y_pred) > best_auc:
               best_auc = roc_auc_score(y_test, y_pred)
               best_gs = gs
               best_clf = idx
     dump_file = 'best_grid_search_pipeline2.pkl'
     best_gs.fit(X_train,y_train)
     joblib.dump(best_gs, dump_file, compress=1)
     return render_template('ENTRENAMIENTO.html', get_plot = True, plot_url = 'static/img/img4.png')
 else:
  return render_template('ENTRENAMIENTO.html')
@app.route('/MODELO', methods=['GET', 'POST'])
def get_plot4():
 if request.method == 'POST': 
     df= pd.read_json(request.form['archivosubido'])
     model_select=str(request.form['modelo'])
     model=joblib.load(model_select)
     #Ajustar valores irreglulares y definir tipos de datos para cada una de las features
     df.loc[df['TotalCharges'] == '', 'TotalCharges'] = 'NaN'
     df["TotalCharges"] = df["TotalCharges"].astype("float64")
     df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")
     ID=df['customerID'].to_frame()
     df=df.drop('customerID', axis=1)
     column_names={'gender':0, 'SeniorCitizen':1, 'Partner':2, 'Dependents':3, 'tenure':4,
       'PhoneService':5, 'MultipleLines':6, 'InternetService':7, 'OnlineSecurity':8,
       'OnlineBackup':9, 'DeviceProtection':10, 'TechSupport':11, 'StreamingTV':12,
       'StreamingMovies':13, 'Contract':14, 'PaperlessBilling':15, 'PaymentMethod':16,
       'MonthlyCharges':17, 'TotalCharges':18}
     df=df.rename(columns=column_names)
     numeric_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='mean'))
      ,('scaler', StandardScaler())])
     categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant'))
      ,('encoder', OrdinalEncoder())])
     numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
     categorical_features = df.select_dtypes(include=['object','int32']).columns
     preprocessor = ColumnTransformer(transformers=[('numeric', numeric_transformer, numeric_features),('categorical', categorical_transformer, categorical_features)])
     X_test=pd.DataFrame(preprocessor.fit_transform(df))
     model=joblib.load('best_grid_search_pipeline.pkl')
     y_pred=model.predict(X_test)
     y_pred_probs=model.predict_proba(X_test)
     final_results = pd.concat([ID], axis = 1)
     #final_results = pd.DataFrame()
     y_pred_probs=y_pred_probs[:, 1]
     final_results['predictions'] = y_pred
     final_results["propensity_to_churn(%)"] = y_pred_probs
     final_results["propensity_to_churn(%)"] = final_results["propensity_to_churn(%)"]*100
     final_results["propensity_to_churn(%)"]=final_results["propensity_to_churn(%)"].round(2)
     return render_template('MODELO.html', tables=[final_results.to_html(classes='table table-stripped')])
 else:
  return render_template('MODELO.html')
if __name__ == '__main__':
    app.run(debug=True)