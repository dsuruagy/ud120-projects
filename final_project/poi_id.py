#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# You will need to use more features
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
                   'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',\
                   'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',\
                   'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# calculating the financial features' correlation with poi
import pandas as pd
df = pd.DataFrame(data_dict).transpose()

financial_feats = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', \
                   'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',\
                   'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',\
                   'director_fees', 'poi']
df[financial_feats] = df[financial_feats].apply(pd.to_numeric, errors ='coerce')
#print df.corr()['poi']

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=False)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators' : [20, 40],
	'max_depth' : [1, 5],
	'criterion' : ('gini', 'entropy'),
	'random_state' : [5, 7]}

clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)


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