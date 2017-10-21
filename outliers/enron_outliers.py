#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

# removing TOTAL data_point, because it's an outlier:
data_dict.pop('TOTAL')

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)


### your code below
#print "data_dict keys:", sorted(data_dict.keys())
for person in data_dict:
	p = data_dict[person]
	if p['salary'] != 'NaN' and p['salary'] > 1000000 and \
		p['bonus'] != 'NaN' and p['bonus'] > 5000000:
		print person

for point in data:
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()