import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import joblib


with open('model_svm.pkl', 'rb') as file_1:
  SVM = joblib.load(file_1)

with open('model_transformer.pkl', 'rb') as file_2:
  transformer = joblib.load(file_2)

with st.form(key='form_credit_default'):
  limbal = st.slider('Limit Balance', 10000, 800000, 50000)
  sex = st.slider('Sex', 1, 2, 1)
  education = st.slider('Education Level', 0, 6, 4)
  marital = st.slider('Marital Status', 0, 3, 2)
  age = st.slider('age', 21, 69, 33)
  pay0 = st.slider('Pay 0', -2, 8, 1)
  pay2 = st.slider('Pay 2', -2, 7, 1)
  pay3 = st.slider('Pay 3', -2, 7, 1)
  pay4 = st.slider('Pay 4', -2, 8, 1)
  pay5 = st.slider('Pay 5', -2, 7, 1)
  pay6 = st.slider('Pay 6', -2, 7, 1)

  bill1 = st.slider('Bill 1', -11545, 613860, 100)
  bill2 = st.slider('Bill 2', -67526, 512650, 100)
  bill3 = st.slider('Bill 3', -25443, 578971, 100)
  bill4 = st.slider('Bill 4', -46627, 488808, 100)
  bill5 = st.slider('Bill 5', -46627, 441981, 100)
  bill6 = st.slider('Bill 6', -73895, 436172, 100)

  payamt1 = st.slider('Pay Amount 1', 0, 493358, 100)
  payamt2 = st.slider('Pay Amount 1', 0, 1100000, 100)
  payamt3 = st.slider('Pay Amount 1', 0, 199209, 100)
  payamt4 = st.slider('Pay Amount 1', 0, 202076, 100)
  payamt5 = st.slider('Pay Amount 1', 0, 388071, 100)
  payamt6 = st.slider('Pay Amount 6', 0, 403500, 100)

  submitted = st.form_submit_button('Predict')

test_data = pd.DataFrame({
  'limit_balance': [limbal],
  'sex': [sex],
  'education_level': [education],
  'marital_status': [marital],
  'age': [age],
  'pay_0': [pay0],
  'pay_2': [pay2],
  'pay_3': [pay3],
  'pay_4': [pay4],
  'pay_5': [pay5],
  'pay_6': [pay6],
  'bill_amt_1': [bill1],
  'bill_amt_2': [bill2],
  'bill_amt_3': [bill3],
  'bill_amt_4': [bill4],
  'bill_amt_5': [bill5],
  'bill_amt_6': [bill6],
  'pay_amt_1': [payamt1],
  'pay_amt_2': [payamt2],
  'pay_amt_3': [payamt3],
  'pay_amt_4': [pay4],
  'pay_amt_5': [payamt5],
  'pay_amt_6': [payamt6],
  'default_payment_next_month': [0]
  }
)

if submitted:
  y_pred_inf = SVM.predict(test_data)
  st.write('# Likelihood of Defaulting : ', str(int(y_pred_inf)))

