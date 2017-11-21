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

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
anova_filter = SelectKBest(f_classif, k=5)
#features_list = anova_filter.fit_transform(feat_scale, labels)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

if True:
    from sklearn.tree import DecisionTreeClassifier
    classif = DecisionTreeClassifier()

    param_grid = {
              'classifier__criterion' : ('gini', 'entropy'),
              'classifier__splitter' : ('best', 'random'),
              'classifier__max_depth' : [None, 1, 2, 3],
              'classifier__random_state' : range(20,42)}


if False:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    classif = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())

    param_grid = {'reduce_dim__n_components' : [None, 1, 19],
              'classifier__base_estimator__criterion' : ('gini', 'entropy'),
              'classifier__base_estimator__splitter' : ('best', 'random'),
              'classifier__base_estimator__max_depth' : [None, 1, 2],
              'classifier__algorithm' : ('SAMME', 'SAMME.R'),
              'classifier__random_state' : range(30,42)}


if False:
    from sklearn.ensemble import RandomForestClassifier
    classif = RandomForestClassifier()

    param_grid = {'reduce_dim__n_components' : [None, 1, 19],
            'classifier__n_estimators': [1, 10, 40],
            'classifier__max_depth': [None, 1, 2],
            'classifier__criterion': ('gini', 'entropy'),
            'classifier__random_state': range(30,42)}



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=42)

estimators = [('reduce_dim', PCA()), ('classifier', classif)]
pipe = Pipeline(estimators)
clf = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=sss, scoring="f1")

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(feat_scale, labels, test_size=0.3, random_state=42)

clf.fit(features_train, labels_train)
clf = clf.best_estimator_
print clf

from sklearn.metrics import accuracy_score, precision_score, recall_score

predict = clf.predict(features_test)
print 'accuracy:', accuracy_score(labels_test, predict)
print 'precision:', precision_score(labels_test, predict)
print 'recall:', recall_score(labels_test, predict)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
