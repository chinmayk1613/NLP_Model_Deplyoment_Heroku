import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

nlpModel=pickle.load(open('nlpModel.pkl','rb'))
CVtransform=pickle.load(open('transform.pkl','rb'))

appNLP=Flask(__name__)

@appNLP.route(('/'))
def homePage():
    return render_template('homePage.html')
@appNLP.route('/predictReview',methods=['POST'])
def predictReview():
    if request.method == 'POST':
        review = request.form['message']
        data = [review]
        vector = CVtransform.transform(data).toarray()
        prediction= nlpModel.predict(vector)
    return render_template('result.html',prediction=prediction)


if __name__ == '__main__':
    appNLP.run(debug=True)