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
