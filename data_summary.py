# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:31:00 2018

@author: z3525552
"""
import missingno as msno
import pandas as pd
import sidetable

def frequency_table(dataframe, data_col1, data_col2):
    
             
    return pd.crosstab(dataframe.data_col1, dataframe.data_col2, margins=True, 
                  margins_name="Total",rownames=data_col1, colnames= data_col2)

def side_table(dataframe, row_index, col_index,value)
    return dataframe.stb.freq([row_index, col_index], value=value)
def missing_data(dataframe):
    
    #Matrix
    #The nullity matrix is a data-dense display which lets 
    #you quickly visually pick out patterns in data completion.
   
    
    msno.matrix(dataframe.sample(500))
    
    
    #Heatmap
    # The missingno correlation heatmap measures nullity correlation: 
    #how strongly the presence or absence of one variable affects
    #the presence of another:
    msno.heatmap(dataframe)