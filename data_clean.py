# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 11:02:07 2018

@author: z3525552
"""

# Data cleaning for analysis and visualisation


#Columns = list the important column from the dataset


#df.columns = Columns

#Drop empty columns, Rename the useful columns

## Rename several DataFrame columns
#df = df.rename(columns = {
   # 'col1 old name':'col1 new name',
   # 'col2 old name':'col2 new name',
   # 'col3 old name':'col3 new name',
#})

# Lower-case all DataFrame column names
#df.columns = map(str.lower, df.columns)

# Even more fancy DataFrame column re-naming

#df.rename(columns=lambda x: x.split('.')[-1], inplace=True)



# Get a report of all duplicate records in a dataframe, based on specific columns
#dupes = df[df.duplicated(['col1', 'col2', 'col3'], keep=False)]

# Sort dataframe by multiple columns
#df = df.sort_values(['col1','col2','col3'],ascending=[1,1,0])

# Note when droping rows and columns from the dataset, specify axis=1 for columns and axis=0 for rows

#df = df.drop([col1, col2], axis=1)


# Split delimited values in a DataFrame column into two new columns
#df['new_col1'], df['new_col2'] = zip(*df['original_col'].apply(lambda x: x.split(': ', 1)))

#Crate an pandas series object with a new column name
#df.assign(newcolumn=value)

#Create a custom column from a python function which takes pandas series as input argument
# df['new_column'] = df.apply(my_function, axis=1) 



# Clean up missing values in multiple DataFrame columns
#df = df.fillna({
    #'col1': 'missing',
    #'col2': 'Unknown',
    #'col3': '999',
    #'col4': '0',
    #'col5': 'missing',
   # 'col6': '99'
#})

