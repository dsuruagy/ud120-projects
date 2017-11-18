import pandas as pd

def describe_data(data_dict):
    '''
    Answer some of important characteristics of data_dict, like:
        - total number of data points
        - allocation across classes (POI/non-POI)
        - number of features used
        - are there features with many missing values? etc.

    :param data_dict:
    :return:
    '''
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

def datadict_to_dataframe(ddict, feats_names):
    df = pd.DataFrame(ddict).transpose()
    df.drop('email_address', inplace=True, axis=1)

    # converting values to numeric format
    df[feats_names] = df[feats_names].apply(pd.to_numeric, errors ='coerce')

    # filling NaN as zero
    df.fillna(0, inplace=True)

    return df

# calculate and print the financial features' correlation with poi
def poi_correlation(ddict, feats_names):
    df = datadict_to_dataframe(ddict, feats_names)
    #corr = df.corr()['poi']
    corr = df.corr()

    # Forcing pandas to display without line breaking
    #pd.set_option('display.expand_frame_repr', False)
    print corr#.abs().sort_values(ascending=False)


### Doing Feature Selection
def select_features(classifier, lbls, feats, feats_names, print_shape=False, threshold='0.008*mean'):

    from sklearn.feature_selection import SelectFromModel
    import numpy as np

    if print_shape:
        print 'features.shape:', feats[0]

    clazzf = classifier.fit(feats, lbls)
    model = SelectFromModel(clazzf, prefit=True, threshold=threshold)
    feats = model.transform(feats)

    if print_shape:
        print 'new features.shape:', feats[0]

    feature_idx = model.get_support()
    # creating a copy of the original features list names
    feats_names = list(feats_names)

    # removing the first feature (poi), to match selected indexes
    poi = feats_names.pop(0)
    features_np = np.array(feats_names)[feature_idx]
    print '\nselected features:', features_np

    # reinserting poi feature, needed to execute the test
    features_np = np.insert(features_np, 0, poi)

    return features_np


### TODO REMOVE - START
'''
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data, main

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
data_dict.pop('TOTAL')

### Extract features and labels from dataset for local testing
financial_feat_list = ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                   'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                   'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                   'director_fees']
data = featureFormat(data_dict, financial_feat_list, sort_keys = True, remove_all_zeroes=False, remove_NaN=True)
labels, features = targetFeatureSplit(data)

from sklearn.decomposition import PCA
pca = PCA()
financial_feature = pca.fit_transform(features)

print 'financial_feature'
print financial_feature
print len(financial_feature[0])

print '\npca.components_'
print pca.components_
print len(pca.components_[0])

idx = 0
for people in data_dict:
    data_dict[people]['financial_component'] = financial_feature[idx][0]
    idx = idx + 1

#poi_correlation(data_dict, features_list)

'''
### TODO REMOVE - END
