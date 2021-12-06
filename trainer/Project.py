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

# Due to the time neccesary to process the timeseries to features , I've decided
# to run these functions once and save the results in seperate csv's .
if not (os.path.isfile(r'./Data/ram_train.csv')):
    ram_train = Helper.time_series2features(input_data["ramposition"], input_data["ramposition_time"])
    ram_test = Helper.time_series2features(test_data["ramposition"], test_data["ramposition_time"])
    inj_train = Helper.time_series2features(input_data["injection_pressure"], input_data["injection_pressure_time"])
    inj_test = Helper.time_series2features(test_data["injection_pressure"], test_data["injection_pressure_time"])
    sen_train = Helper.time_series2features(input_data["sensor_pressure"], input_data["sensor_pressure_time"])
    sen_test = Helper.time_series2features(test_data["sensor_pressure"], test_data["sensor_pressure_time"])
    ram_train = pd.DataFrame(ram_train)
    ram_test = pd.DataFrame(ram_test)
    inj_train = pd.DataFrame(inj_train)
    inj_test = pd.DataFrame(inj_test) 
    sen_train = pd.DataFrame(sen_train)
    sen_test = pd.DataFrame(sen_test)   
    ram_train.to_csv(r'./Data/ram_train.csv', index=None)
    ram_test.to_csv(r'./Data/ram_test.csv', index=None)
    inj_train.to_csv(r'./Data/inj_train.csv', index=None)
    inj_test.to_csv(r'./Data/inj_test.csv', index=None)
    sen_train.to_csv(r'./Data/sen_train.csv', index=None)
    sen_test.to_csv(r'./Data/sen_test.csv', index=None)
# Read the features , extraction static data (This has increased the performance of my predictions)
ram_train = pd.read_csv(r'./Data/ram_train.csv')
ram_train = ram_train.drop(ram_train.columns[[12,20]],1)
ram_train = ram_train.iloc[:,9:]

ram_test = pd.read_csv(r'./Data/ram_test.csv')
ram_test = ram_test.drop(ram_test.columns[[12,20]],1)
ram_test = ram_test.iloc[:,9:]

inj_train = pd.read_csv(r'./Data/inj_train.csv')
inj_train = inj_train.drop(inj_train.columns[[5,6,7,8]],1)


inj_test = pd.read_csv(r'./Data/inj_test.csv')
inj_test = inj_test.drop(inj_test.columns[[5,6,7,8]],1)


sen_train = pd.read_csv(r'./Data/sen_train.csv')
sen_test = pd.read_csv(r'./Data/sen_test.csv')

X_train = np.hstack([ram_train,inj_train,sen_train])
X_test = np.hstack([ram_test,inj_test,sen_test])


# Read the label
Y_train= np.array(Helper.read_Y(os.getcwd()+"/Data/Y4.csv"))

# A decision tree prediction
d_tree = DecisionTreeClassifier(max_depth=10,random_state=101,
                                max_features=None,min_samples_leaf=10).fit(ram_train,Y_train)
pickle.dump(d_tree, open('./shared/model_ramprediction.sav', 'wb'))

ram_pred = d_tree.predict(ram_test)
cv = 4
scores = cross_val_score(d_tree,ram_train,Y_train,cv=cv)
# Print te accuracy on the training set to console & graph a comparison between
# the original training data and the predictions on the test set
print("Crossvalidation : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

print("The accuracy on the ramposition training set was = %0.2f " % (d_tree.score(ram_train,Y_train)*100), '%')
plt.subplot(2, 2, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.5)

plt.hist(Y_train, label = "Training data")
plt.hist(ram_pred, label = "Predicted test data")
plt.legend()
plt.ylabel('Probability')
plt.xlabel('label')
plt.title("Ramposition")


# A decision tree prediction
d_tree = DecisionTreeClassifier(max_depth=10,random_state=101,
                                max_features=None,min_samples_leaf=10).fit(inj_train,Y_train)
pickle.dump(d_tree, open('./shared/model_injection.sav', 'wb'))

inj_pred = d_tree.predict(inj_test)
scores = cross_val_score(d_tree,inj_train,Y_train,cv=cv)
print("Crossvalidation : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# Print te accuracy on the training set to console & graph a comparison between
# the original training data and the predictions on the test set
print("The accuracy on the injection training set was = %0.2f " % (d_tree.score(inj_train,Y_train)*100), '%' )

plt.subplot(2, 2, 2)

plt.hist(Y_train)
plt.hist(inj_pred)
plt.ylabel('Probability')
plt.xlabel('label')
plt.title("Injection")


# A decision tree prediction
d_tree = DecisionTreeClassifier(max_depth=10,random_state=101,
                                max_features=None,min_samples_leaf=10).fit(sen_train,Y_train)
pickle.dump(d_tree, open('./shared/model_sensor.sav', 'wb'))

sen_pred = d_tree.predict(sen_test)
scores = cross_val_score(d_tree,sen_train,Y_train,cv=cv)

# Print te accuracy on the training set to console & graph a comparison between
# the original training data and the predictions on the test set
plt.subplot(2, 2, 3)
print("Crossvalidation : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
print("The accuracy on the sensor training set was = %0.2f "% (d_tree.score(sen_train,Y_train)*100), '%')
plt.hist(Y_train)
plt.hist(sen_pred)
plt.ylabel('Probability')
plt.xlabel('label')
plt.title("Sensor")

# A decision tree prediction
d_tree = DecisionTreeClassifier(max_depth=10,random_state=101,
                                max_features=None,min_samples_leaf=10).fit(X_train,Y_train)
pickle.dump(d_tree, open('./shared/model_combined.sav', 'wb'))

X_pred = d_tree.predict(X_test)
scores = cross_val_score(d_tree,X_train,Y_train,cv=cv)

# Print te accuracy on the training set to console & graph a comparison between
# the original training data and the predictions on the test set
plt.subplot(2, 2, 4)
print("Crossvalidation : %0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

print("The accuracy on the combined training set was = %0.2f "% (d_tree.score(X_train,Y_train)*100), '%')
plt.hist(Y_train)
plt.hist(X_pred)
plt.ylabel('Probability')
plt.xlabel('label')
plt.title("Combined")

plt.show()

# Export test set predictions to csv
pd.Series(ram_pred).to_csv(r'./Data/Ramposition_prediction.csv', index = False, header = False)
pd.Series(inj_pred).to_csv(r'./Data/Injection_prediction.csv', index = False, header = False)
pd.Series(sen_pred).to_csv(r'./Data/Sensor_prediction.csv', index = False, header = False)
pd.Series(X_pred).to_csv(r'./Data/Combined_prediction.csv', index = False, header = False)
