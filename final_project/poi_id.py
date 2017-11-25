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
features_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                   'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                   'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                   'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages',
                   'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
print 'Total number of data points:', len(data_dict)

print_feature = True
total_poi = 0
for data in data_dict:
    if print_feature:
        feature = data_dict[data]
        print '\nEach person\'s features:', len(feature)
        print_feature = False
    # print feature
    if data_dict[data]['poi']:
        total_poi += 1
print '\nnumber of persons of interest:', total_poi

# removing TOTAL data_point, because it's an outlier:
data_dict.pop('TOTAL')

# converting data dict to dataframe, to use statistics functions
import pandas as pd
df = pd.DataFrame(data_dict).transpose()
df.drop('email_address', inplace=True, axis=1)

# converting values to numeric format
df[features_list] = df[features_list].apply(pd.to_numeric, errors ='coerce')

# with the describe function, it's possible to see if there are other max values for outliers
# like for the TOTAL observation
##print df.describe()


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.


sent_to_poi_rate = 'sent_to_poi_rate'
shared_poi_rate = 'shared_poi_rate'
received_poi_rate = 'received_poi_rate'

df[sent_to_poi_rate] = df['from_this_person_to_poi'] / df['from_messages']
df[shared_poi_rate] = df['shared_receipt_with_poi']/df['to_messages']
df[received_poi_rate] = df['from_poi_to_this_person'] / df['to_messages']
df.fillna(0, inplace=True)
my_dataset = df.to_dict(orient='index')
features_list.extend([sent_to_poi_rate, shared_poi_rate, received_poi_rate])


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True, remove_NaN=True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

if False:
    from sklearn.tree import DecisionTreeClassifier
    classif = DecisionTreeClassifier()

    param_grid = {'skb__k' : range(4, 8),
              'classifier__criterion' : ('gini', 'entropy'),
              'classifier__splitter' : ('best', 'random'),
              'classifier__max_depth' : [None, 1, 2, 3],
              'classifier__random_state' : [31,42]}


if True:
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import AdaBoostClassifier
    classif = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())

    param_grid = {'skb__k' : [5, 7],
              'reduce_dim__n_components' : [None, 1],
              'classifier__base_estimator__criterion' : ('gini', 'entropy'),
              'classifier__base_estimator__splitter' : ('best', 'random'),
              'classifier__base_estimator__max_depth' : [None, 1, 2],
              'classifier__algorithm' : ('SAMME', 'SAMME.R'),
              'classifier__n_estimators': [1, 10],
              'classifier__random_state' : [31, 42]}


if False:
    from sklearn.ensemble import RandomForestClassifier
    classif = RandomForestClassifier()

    param_grid = {'skb__k' : [5, 7],
            'reduce_dim__n_components' : [None, 1],
            'classifier__n_estimators': [1, 10],
            'classifier__max_depth': [None, 1, 2],
            'classifier__criterion': ('gini', 'entropy'),
            'classifier__random_state': [31, 42]}



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
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

sss = StratifiedShuffleSplit(n_splits=100, test_size=0.3, random_state=42)
estimators = [('scaler', MinMaxScaler()), ('skb', SelectKBest(chi2)), ('reduce_dim', PCA()), ('classifier', classif)]
pipe = Pipeline(estimators)
gs = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=sss, scoring="f1")
gs.fit(features, labels)
clf = gs.best_estimator_
print clf


## discovering the selected features from SelectKBest:
skb = clf.get_params()['skb']
feature_idx = skb.get_support()

### removing the first feature (poi), to match selected indexes
import numpy as np
poi = features_list.pop(0)
features_np = np.array(features_list)[feature_idx]
print '\nselected features:', features_np
print 'feature scores:', np.array(skb.scores_)[feature_idx]

### reinserting poi feature, needed to execute the test
features_list = np.insert(features_np, 0, poi)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
