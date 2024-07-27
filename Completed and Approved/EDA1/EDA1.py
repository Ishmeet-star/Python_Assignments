# -*- coding: utf-8 -*-
"""
Created on Sat May 18 17:05:16 2024

@author: Lenovo
"""

import pandas as pd 
import numpy as np 

df  = pd.read_csv(r"D:\EXCELR\Assignments\EDA1\Cardiotocographic.csv")
print(df.head())


"""
Task 1: Data Cleaning and Preparation:

    Load the dataset into a DataFrame or equivalent data structure.
    Handle missing values appropriately (e.g., imputation, deletion).
    Identify and correct any inconsistencies in data types (e.g., numerical values stored as strings).
    Detect and treat outliers if necessary.

"""
df.dtypes

headers = list(df.columns.values)
print(headers)

                                                            

# Check data types again
df.dtypes

# ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Tendency', 'NSP']
data = df['LB']
mean = np.mean(data)
std = np.std(data)
 
threshold = 3
outliers = []
for x in data:
    z_score = (x - mean) / std
    if abs(z_score) > threshold:
        outliers.append(x)
print("Mean: ",mean)
print("\nStandard deviation: ",std)
print("\nOutliers  : ", outliers)

# ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Tendency', 'NSP']
# df['LB'].fillna(value=df['LB'].mean(), inplace = True)

"""#####################################################################################
  Method 1: Z-score
import numpy as np
 
data =[1, 2, 3, 4,5,6,7, 8, 9,10,1000]
 
mean = np.mean(data)
std = np.std(data)
 
threshold = 3
outliers = []
for x in data:
    z_score = (x - mean) / std
    if abs(z_score) > threshold:
        outliers.append(x)
print("Mean: ",mean)
print("\nStandard deviation: ",std)
print("\nOutliers  : ", outliers)
Here’s a quick explanation of the above code.

Import numpy module and its alias np
Z-score method uses standard deviation to determine outliers
Calculated z-score > threshold is considered an outlier
Threshold generally lies between 2 to 3
To calculate outlier, initiate for loop with z-score formula (x – mean) / std
Calculate mean and standard deviation beforehand
If absolute value of z-score > threshold, return outliers
Code also returns mean and standard deviation


Method 2: Interquartile Range (IQR)
In this method, we would first calculate the IQR of the given array by subtracting q1 from q3 .If the value/data point is more than 1.5 times the iqr it will be considered an outlier.

import numpy as np
 
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
 
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
threshold = 1.5 * iqr
outliers = np.where((data < q1 - threshold) | (data > q3 + threshold))
 
print("Outliers of array ",data,"is : \n", data[outliers])
We import numpy module for calculating the IQR for the same, we first calculate the percentile of 25th and 75th stored in variables q1 and q2 respectively. iqr is calculated by q3-q1 .We then set the threshold to 1.5 times iqr .   

Method 3: Tukey’s Fences
This method is similar to Interquartile Range (IQR) method used earlier. The only difference is unlike IQR Method this method doesn’t have a single threshold of 1.5 times the IQR, it calculates lower and upper fences based on quartiles, and if the data points/ values lie beyond this range is considered an outlier.


import numpy as np
from scipy import stats
 
data = np.array([1, 20, 20, 20, 21, 100])
 
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1
lower_fence = q1 - 1.5 * iqr
upper_fence = q3 + 1.5 * iqr
outliers = np.where((data < lower_fence) | (data > upper_fence))
 
print("Outliers of array ",data,"is : \n", data[outliers])
Numpy and Scipy assist in calculating fences
Define input array in data
Calculate 25th and 75th percentiles and store in q1 and q3
Calculate IQR as q3-q1
Calculate lower fence as q1 – 1.5 * IQR and upper fence as q3 + 1.5 * IQR
Outliers are values that satisfy the condition to be considered as an outlier
"""     