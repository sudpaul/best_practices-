# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 12:31:00 2018

@author: z3525552
"""
import missingno as msno


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