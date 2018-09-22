# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:51:23 2018

@author: z3525552
"""
import requests
import os
import pandas as pd

#Dowload data

def download_data(base_url, filename, path='data'):
    
   
    
    
    if not os.path.exists(path):
        os.mkdir(path)
    
    url = "{}{}".format(base_url, filename)
    print("Downloading: {} ...".format(url))
    
    response = requests.get(url)
    
    with open(os.path.join(path, filename), 'wb') as f:
        f.write(response.content)

    print("'{}' is ready!".format(filename))



#Read the data from database, type csv, excel or other system

def read_data(filename):
    
    
    filename = filename.lower()
    filetype = filename.split('.')[-1]
    
    if filetype == 'csv':
        df = pd.read_csv(filename)
    
    else:
        df = pd.read_excel(filename, enconding='utf-8')
    
    return df

#Data quality check

def data_quality(dataframe):
        
    for column in dataframe.columns:
        print("Column {} has {} unique values datatype {}".format(
            dataframe[column].name, dataframe[column].nunique(), dataframe[column].dtype
        ))
        print()

    print(dataframe.isnull().sum()) 





#Data transformation








#Data Aggregation

