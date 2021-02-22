# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:36:49 2020

@author: z3525552
"""
import os
import glob
import pandas as pd

directory_name = ''
file_type = 'csv'
output_file = ''
# Changing the file directory
os.chdir(directory_name)
# Joining all same file to a filenames list
filenames = [f for f in glob.glob('*.{}'.format(file_type))]
# Combaine all csv data to a pandas dataframe
data = pd.concat([pd.read_csv(name) for name in filenames ])
# Writing the data file to a excel file
writer = pd.ExcelWriter(directory_name+'/'+ output_file)

data.to_excel(writer, index=False, encoding='utf-8')

writer.save()
