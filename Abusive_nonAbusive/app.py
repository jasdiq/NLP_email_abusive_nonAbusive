# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 12:26:39 2021

@author: yasir Arafath
"""


from flask import Flask, render_template, request
import pickle
import joblib

filename='final_model.pkl'
clf=joblib.load(open(filename, 'rb'))
cv=joblib.load(open('vector.pkl', 'rb'))

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('home.htm')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        message= request.form['message']
        data= [message]
        vect= cv.transform(data).toarray()
        my_prediction=clf.predict(vect)
    return render_template('result.htm', prediction =my_prediction)


if __name__=='__main__':
    app.run(debug=True)
    