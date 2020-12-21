# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 11:36:49 2020

@author: z3525552
"""
import os
import glob
import pandas as pd
os.chdir("C:/Users/z3525552/BORIS_import")
file_type = 'csv'


def load_data(file):
    df = pd.read_csv(file,skiprows=1)
    return df

filenames = [f for f in glob.glob('*.{}'.format(file_type))]

data = pd.concat([load_data(name) for name in filenames ])


writer = pd.ExcelWriter("C:/Users/z3525552/BORIS_import/Medicine_Clinical_school.xlsx")
data.to_excel(writer,'schools', index=False, encoding='utf-8')
writer.save()
