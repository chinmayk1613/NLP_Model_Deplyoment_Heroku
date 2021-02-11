import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

nlpModel=pickle.load(open('nlpModel.pkl','rb'))
CVtransform=pickle.load(open('transform.pkl','rb'))

app=Flask(__name__)

@app.route('/', methods=['POST','GET'])
def homePage():
    return render_template('homePage.html')
@app.route('/predictReview',methods=['POST','GET'])
def predictReview():
    if request.method == 'POST':
        review = request.form['message']
        data = [review]
        vector = CVtransform.transform(data).toarray()
        prediction= nlpModel.predict(vector)
    return render_template('result.html',prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)