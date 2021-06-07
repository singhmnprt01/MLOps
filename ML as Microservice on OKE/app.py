from flask import Flask, request, render_template, redirect, url_for, jsonify
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)

@app.route('/')
def myhome():
    return "My ML App is Live. Hit data to /pred to get predictions"


@app.route('/pred', methods=['POST','GET'])
def make_pred():
    json_data = request.json
    print(json_data)
    
    ### Data Prep Steps ###
    data = pd.DataFrame(json_data)
    data = data.reindex(columns=my_model_columns, fill_value=0) 
    data['Age'].fillna(data[data['Age'].notnull()]['Age'].median(),inplace=True)

    pred_data = (my_model_rf.predict_proba(data)[:,1]).tolist()
    return json.dumps(pred_data)

if __name__ == '__main__':
    model='rf_survival.pkl'
    columns='model_survival_columns.pkl'

    my_model_rf = pickle.load(open(model,'rb'))
    my_model_columns = pickle.load(open(columns,'rb'))

    app.run(debug=True,host='0.0.0.0')

