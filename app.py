import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():

    db_diabetes = pd.json_normalize(
        data=request.json, record_path=['data'])

    lb_gender = LabelEncoder()
    lb_smoking_history = LabelEncoder()

    X_diabetes = db_diabetes.iloc[:, 0:8].values

    X_diabetes[:, 0] = lb_gender.fit_transform(X_diabetes[:, 0])
    X_diabetes[:, 4] = lb_smoking_history.fit_transform(X_diabetes[:, 4])

    scaler = StandardScaler()
    # X_diabetes = scaler.fit_transform(X_diabetes)

    # print(pd.DataFrame(X_diabetes))

    neural_network = pickle.load(open('data/rn_final_diabetes.sav', 'rb'))
    print(X_diabetes[0])

    result = neural_network.predict(X_diabetes[0].reshape(1, -1))
    # print(result)

    return request.json


if __name__ == '__main__':
    app.run(debug=True)
