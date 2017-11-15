#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from poi_id_utils import poi_correlation, select_feature_from_model
from tester import dump_classifier_and_data, main

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                   'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                   'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                   'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                   'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# print features' correlation with poi
poi_correlation(data_dict, features_list)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data)


# Feature selecting
from sklearn.linear_model import LinearRegression
#features_list = select_feature_from_model(LinearRegression(), labels, features, features_list)

from sklearn.svm import LinearSVC
features_list = select_feature_from_model(LinearSVC(C=0.01, penalty="l1", dual=False), labels, features, features_list)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Accuracy: 0.37620	Precision: 0.15599	Recall: 0.83400	F1: 0.26282	F2: 0.44616
#Total predictions: 15000	True positives: 1668	False positives: 9025	False negatives:  332	True negatives: 3975

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

#Accuracy: 0.78860	Precision: 0.19897	Recall: 0.19350	F1: 0.19620	F2: 0.19457
#Total predictions: 15000	True positives:  387	False positives: 1558	False negatives: 1613	True negatives: 11442

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()


#Accuracy: 0.85427	Precision: 0.24451	Recall: 0.04450	F1: 0.07530	F2: 0.05320
#Total predictions: 15000	True positives:   89	False positives:  275	False negatives: 1911	True negatives: 12725
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators' : [20, 40],
	'max_depth' : [1, 5],
	'criterion' : ('gini', 'entropy'),
	'random_state' : [5, 7]}

clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)
'''

#Accuracy: 0.83633	Precision: 0.20945	Recall: 0.08200	F1: 0.11786	F2: 0.09336
#Total predictions: 15000	True positives:  164	False positives:  619	False negatives: 1836	True negatives: 12381
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
clf = AdaBoostClassifier(
	base_estimator = DecisionTreeClassifier(max_depth = 2),
	n_estimators = 50,
	learning_rate = 0.03,
	algorithm = 'SAMME.R',
	random_state = 6)
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

#### TODO Remover esta linha no final
main()