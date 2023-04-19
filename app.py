import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import machine_learning_model

# Create flask app
flask_app = Flask(__name__)


@flask_app.route("/",methods=["GET","POST"])
def Home():
    if request.method == "POST":
        sent=request.form['user_input']
        prediction=machine_learning_model.predict_hatespeech(sent)
        result=prediction[0]
    return render_template("index.html",prediction_result=result)

# @flask_app.route("/predict", methods = ["POST"])
# def submit():
#     if request.method == "POST":
#         name = request.form["sentence"]
#     return render_template("predict.html",n = name)    

if __name__ == "__main__":
    flask_app.run(debug=True,port=8000)
    