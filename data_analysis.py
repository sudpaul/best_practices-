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
        
    """Basic data quality check routine. Input is a pandas dataframe
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

# To do data casting on given columns    
def data_cast(df, key_column, value_column, join_how= 'outer'):
    """Cast the input data frame into a new cast data frame,
    where given the key column containing new variable names and 
    a value column containing the corresponding cells.
    """
    assert type (df) is pd.DataFrame
    assert key_column in df.columns and value_column in df.columns
    assert join_how in ['outer', 'inner']
    
    fixed_columns = df.columns.difference([key_column, value_column])
    df_cast = pd.DataFrame(columns=fixed_columns)
    
    cast_columns = df[key_column].unique()
    
    for column in cast_columns:
        temp_df = df[df[key_column]==column]
        del temp_df[key_column]
        temp_df = temp_df.rename(columns={value_column:column})
        df_cast = df_cast.merge(temp_df, on=list(fixed_columns), how=join_how)
    
    return df_cast     
   

def data_melt(df, col_vals, key, value):
    """Melt the input data frame where given the values often
    appear as columns, the function takes columns into rows, 
    making a "fat" table more tall and skinny and return melted data frame"""
    
    assert type(df) is pd.DataFrame
    keep_vars = df.columns.difference(col_vals)
    melted_sections = []
    
    for c in col_vals:
        melted_c = df[keep_vars].copy()
        melted_c[key] = c
        melted_c[value] = df[c]
        melted_sections.append(melted_c)
    melted = pd.concat(melted_sections)
    
    return melted  
    

def data_transform(df, group_by, numeric_column):
    
    """Data is tranformed by categories and applied numeric sum of the numeric_column
    Inputs are dataframe, group_by column and quantative column and 
    return percentage by the categories. 
       
    Parameters
    ----------
    dataframe : obj
               pandas dataframe object
    group_by : str
               input categorical variable name
    numeric_column: str
                input numeric variable name
    Returns: float
             pandas series object
    """
    total = df.groupby(group_by)[numeric_column].transform('sum')
    percentage = 100*(df[numeric_column]/total)
    
    return percentage
    
# To do data aggregating 

def data_aggregate(df, group_by, numeric_column):
    
    df.groupby(group_by)[numeric_column].mean()

def custom_fuc(df, index, column):
    
    if df[column].loc[index]> 0:
        return 'group_1'
    else:
        return 'group_2'
# Clustering numeric column to categorical variable on aggregate data
df.groupby(lambda x: custom_fuc(df, x, numeric_column))
# Dictionary lookups (Lookup value from a reference source)
 source_dict = {   }
%%timeit
df['newcolumn_lookup'] = np.where(
    df['Col'].values > condition, 
    'val', # default return value
    df['Col_category'].map(source_dict)
)

# Transform the data to a new dimension based on reference condition of other column
conditions = [
    df['Source_col'].str.contains(r'', case=False, na=False),
    df['Source_col'].str.contains(r'', case=False, na=False),
]

outcome = [
    # result1
    #result 2
]

df['Target_col'] = np.select(conditions, choices, default='unknown') # default is the fallback of catch-all


def data_write(df, filename, sheetname):
    '''This helper function takes input dataframe, name of the excel file,
    sheetname and save the file as specified in the directory 
    print out the status''' 
    writer = pd.ExcelWriter(filename)
    df.to_excel(writer,sheetname, index=False, encoding='utf-8')
    writer.save()


