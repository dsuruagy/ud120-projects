#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
def predict_author():
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()

    t0 = time()
    clf.fit(features_train, labels_train)
    # training time is around 1.414 seconds
    print "training time:", round(time() - t0, 3), "s"

    t1 = time()
    pred = clf.predict(features_test)
    # predicting time is around 0.212 s
    print "predicting time:", round(time() - t1, 3), "s"

    return pred

def verify_accuracy(labels, predict):
    from sklearn.metrics import accuracy_score

    return accuracy_score(labels, predict)

accuracy = verify_accuracy(predict_author(), labels_test)
print 'accuracy:', accuracy
#########################################################


