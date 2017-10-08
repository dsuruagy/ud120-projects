import sys
sys.path.append('../2-naiveBayes')
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData

import numpy as np
import pylab as pl

features_train, labels_train, features_test, labels_test = makeTerrainData()



#################################################################################


########################## DECISION TREE #################################



#### your code goes here
from sklearn.metrics import accuracy_score
from sklearn import tree

clf2 = tree.DecisionTreeClassifier(min_samples_split = 2)
clf2.fit(features_train, labels_train)
pred2 = clf2.predict(features_test)
acc_min_samples_split_2 = accuracy_score(pred2, labels_test)

print 'accuracy 2:', acc_min_samples_split_2


clf50 = tree.DecisionTreeClassifier(min_samples_split = 50)
clf50.fit(features_train, labels_train)
pred50 = clf50.predict(features_test)
acc_min_samples_split_50 = accuracy_score(pred50, labels_test)

print 'accuracy 50:', acc_min_samples_split_50
### be sure to compute the accuracy on the test set


    
#def submitAccuracies():
#  return {"acc":round(acc,3)}
def submitAccuracies():
  return {"acc_min_samples_split_2":round(acc_min_samples_split_2,3),
          "acc_min_samples_split_50":round(acc_min_samples_split_50,3)}