import numpy as np
from flask import Flask,request, url_for, redirect, render_template
import pickle
import os



app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))
feature_extraction = pickle.load(open("feature_extractor.pkl", "rb"))

@app.route('/')
def home():
    return render_template("spam.html")


@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        mail_text = request.form["mail"]
        inp_mail = [mail_text]
        input_data_features = feature_extraction.transform(inp_mail)
        prediction = model.predict(input_data_features)
        if prediction[0] == 1:
            result="Ham"
            return render_template("spam.html", ham_text="No worry, its a Ham mail")
        else:
            result="Spam"
            return render_template("spam.html", spam_text="Oops ! Its a Spam mail")



if __name__ == '__main__':
    app.run(debug=True)