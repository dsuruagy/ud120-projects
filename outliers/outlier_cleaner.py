#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    for i in range(len(predictions)):
        cleaned_data.append((ages[i][0], net_worths[i][0], (predictions[i][0] - net_worths[i][0]) ** 2.0))
    
    # sorting list by the residual error:
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])

    # returning only 90% of original data
    return cleaned_data[:int(len(cleaned_data) * 0.9)]

