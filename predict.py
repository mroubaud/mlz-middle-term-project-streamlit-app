import pandas as pd
import numpy as np
import pickle
import sklearn
from flask import Flask
from flask import request
from flask import jsonify

dv_input_file = "dv.bin"
model_input_file = "final_model.bin"

with open(dv_input_file, "rb") as dv_input_file:
  dv=pickle.load(dv_input_file)
app = Flask("score")

with open(model_input_file, "rb") as model_input_file:
  model=pickle.load(model_input_file)
app = Flask("score")

@app.route("/predict", methods=["POST"])
def predict():
  input_data = request.get_json()
  #transform categorical columns
  X = dv.transform(input_data)
  y_pred = model.predict(X)
  result = {
    "predicted salary": float(np.round(y_pred,2)),
  }
  return jsonify(result)

if __name__ == "__main__":
  app.run(debug=True, host="0.0.0.0", port=9696)