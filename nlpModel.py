import re
import nltk
import pandas as pd
import  pickle
from nltk.corpus import stopwords #for stopwords
from nltk.stem.porter import PorterStemmer #for stamming to each of the word....to find the root of word
from sklearn.feature_extraction.text import CountVectorizer # to create sparse matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def restaurant_review_sentiment():
    #Importing the Dataset
    dataset=pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
    #Cleaning the texts
    nltk.download('stopwords')

    corpus_list = []
    for i in range(0,1000):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        review = [ps.stem(word) for word in review if not  word in set(stopwords.words('english'))]
        #Stamming...taking the root of word eg. loved--->love
        review = ' '.join(review)
        corpus_list.append(review)


    #creating the bag of words model

    cv = CountVectorizer(max_features = 1500) # for cleaning the text which we did above
    X = cv.fit_transform(corpus_list).toarray()
    pickle.dump(cv,open('transform.pkl','wb'))
    y = dataset.iloc[:, 1].values

    #Splitting dataset into train and test data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

    #Fitting the Naive Bayes classifier to the training set

    classifier=GaussianNB()
    classifier.fit(X_train,y_train)
    pickle.dump(classifier,open('nlpModel.pkl','wb'))
    print("Model Trained SUccessfully")

if __name__ == '__main__':
    restaurant_review_sentiment()