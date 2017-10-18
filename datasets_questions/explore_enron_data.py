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
		print '\neach person\'s features:', len(feature)
		print_feature = False
		print feature
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


#print 'total with quantified salary', len(enron_data[enron_data.keys()]['salary' != 'NaN'])
#values = enron_data.values()
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
print '\npercent with email address:', (len(wo_total_pay) / float(len(enron_data)))


