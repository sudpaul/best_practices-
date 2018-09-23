# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 13:51:23 2018

@author: z3525552
"""
import os
import requests
import pandas as pd

#Dowload data

def download_data(base_url, filename, path='data'):
    
    """This function download a file from given url and
       write to a file in a local data directory.
    
    Parameters
    ----------
    base_url : str 
               URL for the download file
    filename : str
               Filename
    path  : str
            Local directory to save in the disk. Default is data
    Returns
    ----------
     None 
    """
    
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
    
    """This function read data from local disk
       and return pandas dataframe object.
    
    Parameters
    ----------
    filename : str 
               file name or filepath with filename 

    Returns
    ----------
    dataframe : obj
               pandas dataframe object
     """
    
    filename = filename.lower()
    filetype = filename.split('.')[-1]
    
    if filetype == 'csv':
        df = pd.read_csv(filename)
    
    else:
        df = pd.read_excel(filename, enconding='utf-8')
    
    return df

#Data quality check

def data_quality(dataframe):
        
    """Basice data quality check routine. Input is a pandas dataframe
    print out the number of columns, name, unique values in and datatypes of the
    column
    
    Parameters
    ----------
    dataframe : obj
               pandas dataframe object
    Returns: None
    
    """
     
    print('Number of Columns the dataset has {}'. format(dataframe.shape[1]))
    
    for column in dataframe.columns:
        print("Column {} has {} unique values datatype {}".format(
            dataframe[column].name, dataframe[column].nunique(), dataframe[column].dtype
        ))
        print()

    print(dataframe.isnull().sum()) 

#Data transformation

def data_transform(df, group_by, numeric_column):
    
    total = df.groupby(group_by)[numeric_column].transform('sum')
    percentage = 100*(df[numeric_column]/total)
    
    return percentage
    

def data_aggregate(df, group_by, numeric_column):
    
    df.groupby(group_by)[numeric_column].mean()









#Data Aggregation

