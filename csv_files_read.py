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
os.chdir(directory_name)
filenames = [f for f in glob.glob('*.{}'.format(file_type))]

data = pd.concat([pd.read_csv(name) for name in filenames ])

writer = pd.ExcelWriter(directory_name+'/'+ output_file)
data.to_excel(writer,sheet_name=sheetname, index=False, encoding='utf-8')
writer.save()
