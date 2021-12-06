# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 11:02:18 2021

This is a helper file for the 2021-2022 Project of AI Applications 
& Python for AI

It contains the following functions:
    read_input_data(dir)
        - input: directory of the data files (csv). 
        default = current directory + /Data
        - output: dictionary of data lists. E.g. "ramposition" is a list with 
        N elements where each element is a list containing the values for the 
        ram position over time. "ramposition_time" is a list with N elements
        where each element is a list of the corresponding time instances.
    
    read_test_data(dir)
        - input: directory of the data files (csv). 
        default = current directory + /Data
        - output: dictionary of data lists of the independent test set
        
    read_Y(fid)
        - input: file name (+directory) of your Yx.csv (where x=1,2,3,4 or 5)
        - output: list of the N labels
        
    time_series2features(values_list,time_values_list)
        -input: time series, with values_list= a list of values and 
        time_values_list the corresponding time instances.
        E.g. if you have loaded the input data using read_input_data you can
        do for example 
        features1=time_series2features(input_data["ramposition"], input_data["ramposition_time"])
        -output: a matrix with N rows and 22 columns, containing 22 features 
        for each time series (see function for more details on the features).

@author: Lynn Houthuys
"""

import os
from csv import reader
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
import changefinder


def read_input_data(dir=os.getcwd()+"/Data"):
    input_data=dict()
    
    ramposition=list()
    with open(dir+'/ramposition.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            ramposition.append(row)
    input_data["ramposition"]=ramposition
            
    ramposition_time=list()
    with open(dir+'/ramposition_time.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            ramposition_time.append(row)
    input_data["ramposition_time"]=ramposition_time
    
    injection_pressure=list()
    with open(dir+'/injection_pressure.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            injection_pressure.append(row)
    input_data["injection_pressure"]=injection_pressure
            
    injection_pressure_time=list()
    with open(dir+'/injection_pressure_time.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            injection_pressure_time.append(row)
    input_data["injection_pressure_time"]=injection_pressure_time
    
    sensor_pressure=list()
    with open(dir+'/sensor_pressure.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            sensor_pressure.append(row)
    input_data["sensor_pressure"]=sensor_pressure
            
    sensor_pressure_time=list()
    with open(dir+'/sensor_pressure_time.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            sensor_pressure_time.append(row)
    input_data["sensor_pressure_time"]=sensor_pressure_time
    
            
    return input_data

def read_test_data(dir=os.getcwd()+"/Data"):
    test_data=dict()
    
    ramposition=list()
    with open(dir+'/test_ramposition.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            ramposition.append(row)
    test_data["ramposition"]=ramposition
            
    ramposition_time=list()
    with open(dir+'/test_ramposition_time.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            ramposition_time.append(row)
    test_data["ramposition_time"]=ramposition_time
    
    injection_pressure=list()
    with open(dir+'/test_injection_pressure.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            injection_pressure.append(row)
    test_data["injection_pressure"]=injection_pressure
            
    injection_pressure_time=list()
    with open(dir+'/test_injection_pressure_time.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            injection_pressure_time.append(row)
    test_data["injection_pressure_time"]=injection_pressure_time
    
    sensor_pressure=list()
    with open(dir+'/test_sensor_pressure.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            sensor_pressure.append(row)
    test_data["sensor_pressure"]=sensor_pressure
            
    sensor_pressure_time=list()
    with open(dir+'/test_sensor_pressure_time.csv','r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=[float(i) for i in row]
            sensor_pressure_time.append(row)
    test_data["sensor_pressure_time"]=sensor_pressure_time
    
            
    return test_data

def read_Y(fid):
    Y=list()
    with open(fid,'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            row=float(row[0])
            Y.append(row)
    return Y


def time_series2features(values_list,time_values_list):
    input_matrix=np.zeros(shape=(len(values_list),22))   
    
    for k in range(len(values_list)):
        values=values_list[k]
        time_values=time_values_list[k]
        
        #trim approx 0
        trim_value=0.00001
        values_trimmed=[v for v in values if v>trim_value]
        times_trimmed=[time_values[i] for i in range(len(time_values)) if values[i]>trim_value]
        
        #cycletime
        ct=times_trimmed[-1]
        input_matrix[k,0]=ct
        
        # peaks - values, positions, widths at half prominence, prominences (height peak)
        peaks_info=signal.find_peaks(values_trimmed, prominence=0.01, width=0.01,rel_height=0.5)
        if len(peaks_info[0])>0:
            peak_value=values_trimmed[peaks_info[0][0]]
            peak_time=times_trimmed[peaks_info[0][0]]
            peak_width=peaks_info[1]["widths"][0]
            peak_prominence=peaks_info[1]["prominences"][0]
            if len(peaks_info[0])>1: #second peak if there is one
                peak2_value=values_trimmed[peaks_info[0][1]]
                peak2_time=times_trimmed[peaks_info[0][1]] 
                peak2_width=peaks_info[1]["widths"][1]
                peak2_prominence=peaks_info[1]["prominences"][1]
            else:
                peak2_value=0
                peak2_time=0
                peak2_width=0
                peak2_prominence=0
        else:
            peak_value=0
            peak_time=0
            peak_width=0
            peak_prominence=0
            peak2_value=0
            peak2_time=0
            peak2_width=0
            peak2_prominence=0
        input_matrix[k,1:9]=np.array([peak_value,peak_time,peak_width,peak_prominence,peak2_value,peak2_time,peak2_width,peak2_prominence])
        
        #mean, median, min, max, std, q75, q90
        m=np.mean(values_trimmed)
        md=np.median(values_trimmed)
        mi=np.min(values_trimmed)
        ma=np.max(values_trimmed)
        st=np.std(values_trimmed)
        q75=np.quantile(values_trimmed,0.75)
        q90=np.quantile(values_trimmed,0.9)
        input_matrix[k,9:16]=np.array([m,md,mi,ma,st,q75,q90])
        
        #Root-mean-square level
        rms=np.sqrt(np.dot(values_trimmed,values_trimmed)/len(values_trimmed))
        input_matrix[k,16]=rms
        
        #skewness, kurtosis
        sk=skew(values_trimmed)
        kur=kurtosis(values_trimmed)
        input_matrix[k,17:19]=np.array([sk,kur])
        
        #time of most abrupt change in series
        cf = changefinder.ChangeFinder(r=0.01, order=3, smooth=5)
        ts_score = [cf.update(p) for p in values_trimmed]
        ch_time=times_trimmed[ts_score.index(max(ts_score))]
        input_matrix[k,19]=ch_time
        
        #mid-level= 50%-reference level= halfway the highest point
        midlev=0.5*ma
        #time when graph crosses the mid-level for the first time
        just_before=np.where(values_trimmed>midlev)[0][0]-1
        just_after=np.where(values_trimmed>midlev)[0][0]
        midlev_time=(times_trimmed[just_after]-times_trimmed[just_before])/2 + times_trimmed[just_before]
        input_matrix[k,20:22]=np.array([midlev,midlev_time])
        
    return input_matrix


