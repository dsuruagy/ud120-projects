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
import sys
import numpy as np
sys.path.append('../tools')
from feature_format import featureFormat
from feature_format import targetFeatureSplitNP

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print 'total people on dataset:', len(enron_data)

print_feature = True
total_poi = 0

for data in enron_data:
	if print_feature:
		feature  = enron_data[data]
		print '\neach person\'s features:', len(feature)
		print_feature = False
		#print feature
	if enron_data[data]['poi']:
		print 'POI:', data
		total_poi += 1
print '\nnumber of persons of interest:', total_poi

print '\nJames Prentice\'s value of stock:', enron_data['PRENTICE JAMES']['total_stock_value']

print '\nWesley Colwell\'s emails to POI:', enron_data['COLWELL WESLEY']['from_this_person_to_poi']

print '\nJeffrey Skilling\'s exercised_stock_options:', \
    enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print '\nLAY KENNETH L total_payments:', enron_data['LAY KENNETH L']['total_payments']
print 'FASTOW ANDREW S total_payments:', enron_data['FASTOW ANDREW S']['total_payments']
print 'SKILLING JEFFREY K total_payments:', enron_data['SKILLING JEFFREY K']['total_payments']

with_salary = {}
with_email = {}
wo_total_pay = {}

for value in enron_data:
	if enron_data[value]['salary'] != 'NaN':
		with_salary[value] = enron_data[value]
	if enron_data[value]['email_address'] != 'NaN':
		with_email[value] = enron_data[value]
	if enron_data[value]['total_payments'] == 'NaN':
		wo_total_pay[value] = enron_data[value]

print '\ntotal with quantified salary:', len(with_salary)
print 'total with email address:', len(with_email)
print '\npercent with total_salary equal to NaN:', (len(wo_total_pay) / float(len(enron_data)))

#feature_list = ['salary', 'to_messages', 'deferral_payments', 'total_payments',
# 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi',
# 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
# 'other', 'from_this_person_to_poi', 'poi', 'director_fees', 'deferred_income', 'long_term_incentive',
# 'email_address', 'from_poi_to_this_person']

feature_list = ['poi', 'total_payments']
data_array = featureFormat(enron_data, feature_list, remove_all_zeroes = False)
poi, features = targetFeatureSplitNP(data_array)

total_payments_poi = features[poi == True]
count_NaN = 0.0
for pay in total_payments_poi:
	if pay == 0.0:
		count_NaN = count_NaN + 1.0

print '\npercent of POI with NaN in total_payments:', count_NaN / len(total_payments_poi) 
#print enron_data
print '\nnumber of people with NaN in total_payments + 10:', len(features[features == 0.0]) + 10