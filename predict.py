# #### Load the model

import pickle
import numpy as np 
from flask import Flask
from flask import request
from flask import jsonify
import xgboost as xgb


version = 1
model_file = f'model_v{version}.bin'

#load saved dv,model and store it to runtime variable
with open(model_file,'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')


@app.route('/predict', methods=['POST'])
def predict():
    player_data = request.get_json()
    
    X = dv.transform(player_data)
    dplayer = xgb.DMatrix(X, feature_names=dv.feature_names_)
    y_pred = model.predict(dplayer)[0]
    price_prediction = np.expm1(y_pred)


    result = {
        'price_prediction_in_euro': float(price_prediction),
    }

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port =9696)