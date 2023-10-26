#!/usr/bin/python3.10

from flask import Flask, jsonify, request
import pickle
import xgboost as xgb 

with open('emloyeeleaveclassifier.pkl', 'rb') as f_in: 
    model = pickle.load(f_in)

with open('dictvictorizer.pkl', 'rb') as f_n: 
    dv = pickle.load(f_n)

app = Flask('ping') 

def transform_employee(employee):
    employee["yearsinthecompany"] = 2024 - employee["joiningyear"]
    del employee['joiningyear']
    return employee 


def predict_single(employee, dv, model):

  X = dv.transform([employee])  ## apply the one-hot encoding feature to the employee data 
  feature_names = list(dv.get_feature_names_out())
  d_val_xgb = xgb.DMatrix(X, 
                        feature_names=feature_names)
  y_predict = model.predict(d_val_xgb)
  return y_predict[0]


@app.route('/predict', methods=['POST'])  ## in order to send the employee information we need to post its data.
def predict():
    employee = transform_employee(request.get_json())  
    leave_proba = predict_single(employee, dv, model)
    if leave_proba > 0.5:
       leave = "yes"
    else:
       leave= "No"   
    result = {
        'leave_probability': round(float(leave_proba),3),
        'leave': leave,  
    }

    return jsonify(result)

if __name__ == '__main__':
   app.run(debug=True, host='0.0.0.0', port=9696) 