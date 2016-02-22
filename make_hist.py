# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 18:57:50 2015

@author: yunxiao
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandasql
import scipy
import scipy.stats

def make_hist(filename):
    df = pd.read_csv(filename)
    df_rain = df[df['rain']==1]
    f1 = plt.figure()
    a1 = f1.add_subplot(111)
    df['meantempi'][df['rain'] == 0].hist(bins=25, range=(50, 80), color='blue', label='No Rain') # your code here to plot a historgram for hourly entries when it is not raining    
    df['meantempi'][df['rain'] == 1].hist(bins=25, range=(50, 80), color='green', label='Rain') # your code here to plot a historgram for hourly entries when it is raining
    plt.xlabel('meantempi')
    plt.ylabel('Frequency')
    plt.title('Histogram of meantempi')
    plt.legend()
    f2 = plt.figure()
    a2 = f2.add_subplot(111)
    df['Hour'][df['rain'] == 1].hist(bins=24, range=(-5, 25), color = 'blue', label='Rain')
    plt.xlabel('Hour')
    plt.ylabel('Frequency')
    plt.title('Histogram of Hour')
    plt.legend(loc='upper center')
    
    rain_temp = df['meantempi'][df['rain'] == 1]
    norain_temp = df['meantempi'][df['rain'] == 0]
    U, p = scipy.stats.mannwhitneyu(rain_temp, norain_temp)
    print "U=%.2e, p=%.2e" % (U, p) 

    
make_hist('turnstile_data_master_with_weather.csv')
