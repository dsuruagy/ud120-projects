# calculate and print the financial features' correlation with poi
def poi_correlation(ddict, feats_names):
    import pandas as pd
    df = pd.DataFrame(ddict).transpose()

    # converting values to numeric format
    df[feats_names] = df[feats_names].apply(pd.to_numeric, errors ='coerce')
    corr = df.corr()['poi']
    print corr.abs().sort_values(ascending=False)


### Doing Feature Selection
def select_feature_from_model(classifier, lbls, feats, feats_names, print_shape=False, threshold='0.008*mean'):

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