#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import poi_id_utils as pu
from feature_format import featureFormat, targetFeatureSplit
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

### Task 2: Remove outliers
pu.describe_data(data_dict)

# removing TOTAL data_point, because it's an outlier:
data_dict.pop('TOTAL')

# converting data dict to dataframe, to use statistics functions
df = pu.datadict_to_dataframe(data_dict, features_list)

# with the describe function, it's possible to see if there are other max values for outliers
# like for the TOTAL observation
#print df.describe()


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data)


# print features' correlation with poi
#pu.poi_correlation(data_dict, features_list)

# Feature selecting
from sklearn.preprocessing import scale
feat_scale = scale(features)

from sklearn.svm import LinearSVC
features_list = pu.select_features(LinearSVC(C=0.1, penalty="l1", dual=False), labels, feat_scale, features_list)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html



# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


#Accuracy: 0.82736	Precision: 0.38711	Recall: 0.35750	F1: 0.37172	F2: 0.36305
#Total predictions: 14000	True positives:  715	False positives: 1132	False negatives: 1285	True negatives: 10868
#classif = DecisionTreeClassifier()
from sklearn.ensemble import AdaBoostClassifier
classif = AdaBoostClassifier(
	base_estimator = DecisionTreeClassifier())
	#n_estimators = 250,
	#learning_rate = 0.03,
	#algorithm = 'SAMME.R',
	#random_state = 6)
estimators = [('reduce_dim', PCA()), ('classifier', classif)]

#Accuracy: 0.85450	Precision: 0.47323	Recall: 0.16350	F1: 0.24303	F2: 0.18813
#Total predictions: 14000	True positives:  327	False positives:  364	False negatives: 1673	True negatives: 11636
#estimators = [('reduce_dim', PCA()), ('classifier', RandomForestClassifier())]

pipe = Pipeline(estimators)
'''
'reduce_dim__n_components' :[None, 2, 3],
'reduce_dim__whiten' : (True, False),

, scoring='recall'
'''

param_grid = {
              'classifier__base_estimator__criterion' : ('gini', 'entropy'),
              'classifier__base_estimator__splitter' : ('best', 'random'),
              'classifier__base_estimator__max_depth' : [None, 1, 2],
              'classifier__algorithm' : ('SAMME', 'SAMME.R'),
              'classifier__random_state' : range(1,7)}
#param_grid = {'classifier__max_depth' : range(2,30)}
clf = GridSearchCV(pipe, param_grid=param_grid, n_jobs=4)
#clf =GridSearchCV(classif, param_grid=param_grid, n_jobs=4)
#clf = pipe


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(feat_scale, labels, test_size=0.3, random_state=42)

if True:
    clf.fit(features_train, labels_train)
    print '\nBest score: %0.3f' % clf.best_score_
    print 'Best parameters set:'
    best_parameters = clf.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print '\t%s: %r' % (param_name, best_parameters[param_name])

    from sklearn.metrics import accuracy_score, precision_score, recall_score
    print 'accuracy:', accuracy_score(labels_test, clf.predict(features_test))
    print 'precision:', precision_score(labels_test, clf.predict(features_test))
    print 'recall:', recall_score(labels_test, clf.predict(features_test))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

#### TODO Remover esta linha no final
#main()