# calculate and print the financial features' correlation with poi
def poi_correlation(ddict, feats_names):
    import pandas as pd
    df = pd.DataFrame(ddict).transpose()

    # converting values to numeric format
    df[feats_names] = df[feats_names].apply(pd.to_numeric, errors ='coerce')
    corr = df.corr()['poi']
    print corr.abs().sort_values(ascending=False)


### Doing Feature Selection
def select_feature_from_model(classifier, lbls, feats, feats_names, print_shape=False):

    from sklearn.feature_selection import SelectFromModel
    import numpy as np

    if print_shape:
        print 'features.shape:', feats[0]

    clazzf = classifier.fit(feats, lbls)
    model = SelectFromModel(clazzf, prefit=True, threshold='0.008*mean')
    feats = model.transform(feats)

    if print_shape:
        print 'new features.shape:', feats[0]

    feature_idx = model.get_support()
    # creating a copy of the original features list names
    feats_names = list(feats_names)
    feats_names.pop(0)
    features_np = np.array(feats_names)
    print '\nselected features:', features_np[feature_idx]

    return feats