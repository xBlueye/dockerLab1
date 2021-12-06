# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:39:47 2021

@author: gebruiker
"""

import Helper
import os
from csv import reader
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import scipy.optimize as opt
import changefinder
import  matplotlib.pyplot as plt
import pandas as pd
import os.path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import pickle
#Read data 
input_data=Helper.read_input_data()
test_data=Helper.read_test_data()


# Read the features , extraction static data (This has increased the performance of my predictions)

ram_test = pd.read_csv(r'./Data/ram_test.csv')
ram_test = ram_test.drop(ram_test.columns[[12,20]],1)
ram_test = ram_test.iloc[:,9:]

inj_test = pd.read_csv(r'./Data/inj_test.csv')
inj_test = inj_test.drop(inj_test.columns[[5,6,7,8]],1)

sen_test = pd.read_csv(r'./Data/sen_test.csv')

X_test = np.hstack([ram_test,inj_test,sen_test])

loaded_model = pickle.load(open('./shared/model_ramprediction.sav', 'rb'))
ram_pred = loaded_model.predict(ram_test)
loaded_model = pickle.load(open('./shared/model_injection.sav', 'rb'))
inj_pred = loaded_model.predict(inj_test)
loaded_model = pickle.load(open('./shared/model_sensor.sav', 'rb'))
sen_pred = loaded_model.predict(sen_test)
loaded_model = pickle.load(open('./shared/model_combined.sav', 'rb'))
X_pred = loaded_model.predict(X_test)

print("The predicted labels for ramposition dataset are :" , ram_pred)

print("The predicted labels for injection dataset are :" , inj_pred)

print("The predicted labels for sensor dataset are :" , sen_pred)

print("The predicted labels for combined dataset are :" , X_pred)


