# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:13:58 2024

@author: Lenovo
"""

"""
Objective: To compute and analyze basic statistical measures for numerical columns in the dataset.
Steps:
Load the dataset into a data analysis tool or programming environment (e.g., Python with pandas library).
Identify numerical columns in the dataset.
Calculate the mean, median, mode, and standard deviation for these columns.
Provide a brief interpretation of these statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt


data  = pd.read_csv(r"D:\EXCELR\Assignments\Basic stats - 1\sales_data_with_discounts.csv")
# print(data)

print(data.dtypes)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

df = data.select_dtypes(include=numerics)
print(df)

df.describe()

mode_vol = df['Volume'].mode
mode_ap = df['Avg Price'].mode
mode_tsv = df['Total Sales Value'].mode
mode_dr = df['Discount Rate (%)'].mode
mode_da = df['Discount Amount'].mode
mode_nsv = df['Net Sales Value'].mode


# data1  = pd.read_csv(r"D:\EXCELR\Assignments\Basic stats - 1\sales_data_with_discounts.csv")
# start_date = pd.to_datetime('2020-4-1')
# end_date = pd.to_datetime('2020-9-30')                         
# df['Date'] = pd.to_datetime(df['Date']) 
# new_df = (df['Date']>= start_date) & (df['Date']<= end_date)
# df1 = df.loc[new_df]
df2 = df[['Volume']]
#df3 = df2.set_index('Date')
plt.figure(figsize=(25,25))
df2.plot.hist(alpha=0.5)

plt.suptitle('Acha', fontsize=12, color='blue')
plt.show()    