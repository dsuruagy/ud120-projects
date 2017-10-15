#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'total people on dataset:', len(enron_data)

print_feature = True
total_poi = 0

for data in enron_data:
	if print_feature:
		feature  = enron_data[data]
		print 'each person\'s features:', len(feature)
		print_feature = False
	if enron_data[data]['poi']:
		print 'POI:', data
		total_poi += 1
print 'number of persons of interest:', total_poi
