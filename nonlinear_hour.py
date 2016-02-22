# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 22:30:13 2015

@author: yunxiao
"""
import numpy as np
import pandas
import scipy
import statsmodels.api as sm
import datetime
import pandasql

"""
In this optional exercise, you should complete the function called 
predictions(turnstile_weather). This function takes in our pandas 
turnstile weather dataframe, and returns a set of predicted ridership values,
based on the other information in the dataframe.  

In exercise 3.5 we used Gradient Descent in order to compute the coefficients
theta used for the ridership prediction. Here you should attempt to implement 
another way of computing the coeffcients theta. You may also try using a reference implementation such as: 
http://statsmodels.sourceforge.net/devel/generated/statsmodels.regression.linear_model.OLS.html

One of the advantages of the statsmodels implementation is that it gives you
easy access to the values of the coefficients theta. This can help you infer relationships 
between variables in the dataset.

You may also experiment with polynomial terms as part of the input variables.  

The following links might be useful: 
http://en.wikipedia.org/wiki/Ordinary_least_squares
http://en.wikipedia.org/w/index.php?title=Linear_least_squares_(mathematics)
http://en.wikipedia.org/wiki/Polynomial_regression

This is your playground. Go wild!

How does your choice of linear regression compare to linear regression
with gradient descent computed in Exercise 3.5?

You can look at the information contained in the turnstile_weather dataframe below:
https://www.dropbox.com/s/meyki2wl9xfa7yk/turnstile_data_master_with_weather.csv

Note: due to the memory and CPU limitation of our amazon EC2 instance, we will
give you a random subset (~10%) of the data contained in turnstile_data_master_with_weather.csv

If you receive a "server has encountered an error" message, that means you are hitting 
the 30 second limit that's placed on running your program. See if you can optimize your code so it
runs faster.
"""
def normalize_features(df):
    """
    Normalize the features in the data set.
    """
    mu = df.mean()
    sigma = df.std()
    
    if (sigma == 0).any():
        raise Exception("One or more features had the same value for all samples, and thus could " + \
                         "not be normalized. Please do not include features with only a single value " + \
                         "in your model.")
    df_normalized = (df - df.mean()) / df.std()

    return df_normalized, mu, sigma

def predict(weather_turnstile):
    #
    # Your implementation goes here. Feel free to write additional
    # helper functions
    #
    dataframe = weather_turnstile
    busy_hours = [0, 9, 12, 16, 20]
    
    def process_hour(hour):
        hour_list = [hour for i in range(len(busy_hours))]
        diff = np.subtract(busy_hours, hour_list)
        return np.exp(-min(np.square(diff)))
    
    processed_hour = dataframe['Hour'].map(process_hour)
    # print processed_hour[0:5]
    # Select Features (try different features!)
    features = dataframe[['rain', 'fog', 'precipi', 'meantempi']]
    features = features.join(processed_hour)
    # features = dataframe[['rain', 'precipi', 'Hour', 'meantempi']]
    
    # Add a weekend column to the features, 1 if it's weekend, 0 otherwise.
    weekend = dataframe['DATEn'].map(lambda x: int(datetime.datetime.strptime(x, '%Y-%m-%d').isoweekday() > 5))
    features = features.join(weekend)
    
    # Add UNIT to features using dummy variables
    dummy_units = pandas.get_dummies(dataframe['UNIT'], prefix='unit')
    features = features.join(dummy_units) #This will add a lot of features!!!
    
    # Values
    values = dataframe['ENTRIESn_hourly']
    m = len(values)

    features, mu, sigma = normalize_features(features)
    features['ones'] = np.ones(m) # Add a column of 1s (y intercept)
    
    # Convert features and values to numpy arrays
    features_array = np.array(features)
    values_array = np.array(values)
    model = sm.OLS(values_array, features_array)
    results = model.fit()
    # print results.params
    parameters = results.params
    predictions = results.predict()
    return predictions, parameters,

df = pandas.read_csv('turnstile_data_master_with_weather.csv')
predictions, parameters = predict(df)
data = df['ENTRIESn_hourly']
r_squared = 1 - np.sum((data - predictions)**2)/np.sum((data-np.mean(data))**2)
print r_squared