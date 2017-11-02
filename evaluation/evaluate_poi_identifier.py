#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""
import numpy as np
import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 
features_train, features_test, labels_train, labels_test = \
	train_test_split(features, labels, test_size = 0.3, random_state = 42)

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print 'accuracy metrics:', accuracy_score(labels_test, pred)
print '\nPOIs identified on our test set prediction:', len(pred[pred == 1])
print 'number of people on test set:', len(features_test)

pred_zeros = np.zeros_like(pred)
print 'accuracy metrics if all prediction were zeros:', accuracy_score(labels_test, pred_zeros)

print 'number of true positives in prediction:', np.count_nonzero(pred[pred == labels_test])

print 'precision metrics:', precision_score(labels_test, pred)
print 'recall metrics:', recall_score(labels_test, pred)


#predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
#true labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

#true positives: 6
#true negatives: 9
#false positives: 3
#false negatives: 2
#precision = true_positives / (true_positives + false_positives) = 6 / (6+3) = 6 / 9 = 0.667
#recall    = true_positives / (true_positives + false_negatives) = 6 / (6+2) = 6 / 8 = 0.75