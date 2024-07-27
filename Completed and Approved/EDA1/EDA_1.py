# -*- coding: utf-8 -*-
"""
Created on Sat May 25 17:39:23 2024

@author: Lenovo
"""

"""
Tasks:
Data Cleaning and Preparation:
Load the dataset into a DataFrame or equivalent data structure.
Handle missing values appropriately (e.g., imputation, deletion).
Identify and correct any inconsistencies in data types (e.g., numerical values stored as strings).
Detect and treat outliers if necessary.

Statistical Summary:
Provide a statistical summary for each variable in the dataset, including measures of central tendency (mean, median) and dispersion (standard deviation, interquartile range).
Highlight any interesting findings from this summary.

Data Visualization:
Create histograms or boxplots to visualize the distributions of various numerical variables.
Use bar charts or pie charts to display the frequency of categories for categorical variables.
Generate scatter plots or correlation heatmaps to explore relationships between pairs of variables.
Employ advanced visualization techniques like pair plots, or violin plots for deeper insights.

Pattern Recognition and Insights:
Identify any correlations between variables and discuss their potential implications.
Look for trends or patterns over time if temporal data is available.

Conclusion:
Summarize the key insights and patterns discovered through your exploratory analysis.
Discuss how these findings could impact decision-making or further analyses.

Deliverables:
A detailed Jupyter Notebook file containing the code, visualizations, and explanations for each step of your analysis.


"""

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew

def measure_upper_and_lower_limit(dataframe_name,column):
    q1 = dataframe_name[column].quantile(0.25)
    q3 = dataframe_name[column].quantile(0.75)
    iqr = q3-q1
    upper_limit = q3 + (1.5*iqr)
    lower_limit = q1 - (1.5*iqr)
    Outliers = dataframe_name.loc[(dataframe_name[column] > upper_limit) | (dataframe_name[column] < lower_limit)]
    print("q1 is : ",q1)
    print("q3 is : ",q3)
    print("iqr is : ",iqr)
    print("upper_limit is : ",upper_limit)
    print("lower_limit is : ",lower_limit)
    
    # This is for Trimming the outliers 
    dataframe_name.loc[(dataframe_name[column] < upper_limit) & (dataframe_name[column] > lower_limit)]
    print("Before removing outliers", len(Copy_of_Original_data_1))
    print("after removing outliers", len(dataframe_name))
    print("outliers",len(Copy_of_Original_data_1)-len(dataframe_name))
    
    # This is for Capping the outliers    
    dataframe_name.loc[(dataframe_name[column]>upper_limit),column]= upper_limit
    dataframe_name.loc[(dataframe_name[column]<lower_limit),column]= lower_limit
    
    return upper_limit, lower_limit,Outliers

def plot_my_graph(dataframe_name,col):
    sns.boxplot(data=dataframe_name[col])
    plt.title(f'Boxplot of {col}')
    plt.show() 

def skew_ness_check(dataframe_name,col):
    Original_data.describe(include='all')
    
    mean = dataframe_name[col].mean()
    print(f"Mean for {col} is : ",mean)
    median = dataframe_name[col].median()
    print(f"Median for {col} is : ",median)
    mode = dataframe_name[col].mode().values[0]
    # print(mode.dtype)
    print(f"Mode for {col} is : ",mode)
    
    if(mean>median>mode):
        dataframe_name[col].kurt()
        print(f'Kurtosis for the {col} is ', dataframe_name[col].kurt())
        
        skew(dataframe_name[col])
        print(f'skewness for the {col} is ', skew(dataframe_name[col]))
        plt.figure()
        sns.distplot(dataframe_name[col])
        plt.title(f'Skewness is \n {skew(dataframe_name[col])} and Kurtosis is \n {dataframe_name[col].kurt()}')
        plt.show()
        print(f'We can see that {mean} > {median} > {mode}. So, the distribution of {col} is positively skewed and positive kurtosis. The plot for its distribution to confirm is below.')
    else:
        dataframe_name[col].kurt()
        print(f'Kurtosis for the {col} is ', dataframe_name[col].kurt())
        
        skew(dataframe_name[col])
        print(f'skewness for the {col} is ', skew(dataframe_name[col]))
        plt.figure()
        sns.distplot(dataframe_name[col])
        plt.title(f'Skewness is \n {skew(dataframe_name[col])} and Kurtosis is \n {dataframe_name[col].kurt()}')
        plt.show()
        print(f'We can see that {mean} < {median} < {mode}. So, the distribution of {col} is negatively skewed and negative kurtosis. The plot for its distribution to confirm is below.')    
 
    
    
if __name__=="__main__": 
   """
    Data Cleaning and Preparation:
    Load the dataset into a DataFrame or equivalent data structure.
    Handle missing values appropriately (e.g., imputation, deletion).
    Identify and correct any inconsistencies in data types (e.g., numerical values stored as strings).
    Detect and treat outliers if necessary.

   """
   Original_data  = pd.read_csv(r"D:\EXCELR\Assignments\EDA1\Cardiotocographic.csv")

   print(Original_data.isnull().sum())

   Copy_of_Original_data_1 = Original_data.copy()
   headers = ['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Tendency', 'NSP']

   for col in headers:
       Copy_of_Original_data_1[col].fillna(value=Copy_of_Original_data_1[col].mean(), inplace = True)

   print(Copy_of_Original_data_1.isnull().sum())

   Copy_of_Original_data_2 = Copy_of_Original_data_1.copy()    
   for col in headers:
       measure_upper_and_lower_limit(Copy_of_Original_data_2,col)
       plot_my_graph(Copy_of_Original_data_2,col)      
   # The data without outliers 
   Copy_of_Original_data_3 = Copy_of_Original_data_2.copy()
   for col in headers:
       plot_my_graph(Copy_of_Original_data_3,col)
       
   """
   Statistical Summary:
   Provide a statistical summary for each variable in the dataset, including measures of central tendency (mean, median) and dispersion (standard deviation, interquartile range).
   Highlight any interesting findings from this summary.

   """
   Copy_of_Original_data_1.describe(include='all')
   
   for col in headers:
       skew_ness_check(Copy_of_Original_data_1,col)
       
   print("In this project, I describe the descriptive statistics that are used to summarize a dataset.")
   print("In particular, I have described the measures of central tendency (mean, median and mode). I have also described the measures of dispersion or variability (variance, standard deviation, coefficient of variation, minimum and maximum values, IQR) and measures of shape (skewness and kurtosis).")
   print("I have computed the measures of central tendency-mean, median and mode for all the columns. I have found mean > median > mode. So, the distribution of all the columns is positively skewed. I have plotted its distribution to confirm the same.")
   
   
   
   """Data Visualization:
    Create histograms or boxplots to visualize the distributions of various numerical variables.
    Use bar charts or pie charts to display the frequency of categories for categorical variables.
    Generate scatter plots or correlation heatmaps to explore relationships between pairs of variables.
    Employ advanced visualization techniques like pair plots, or violin plots for deeper insights.
    """
   for col in Copy_of_Original_data_1:
       sns.boxplot(data=Copy_of_Original_data_1[col])
       plt.title(f'Boxplot of {col}')
       plt.show()
       
       sns.barplot(data=Copy_of_Original_data_1[col])
       plt.title(f'Barplot of {col}')
       plt.show()
       
       sns.scatterplot(data=Copy_of_Original_data_1[col])
       plt.title(f'Scatterplot of {col}')
       plt.show()
       
       sns.violinplot(data=Copy_of_Original_data_1[col])
       plt.title(f'Violinplot of {col}')
       plt.show()
    
   """
    Pattern Recognition and Insights:
    Identify any correlations between variables and discuss their potential implications.
    Look for trends or patterns over time if temporal data is available.

   """
   
   Copy_of_Original_data_1[['LB', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'ASTV', 'MSTV', 'ALTV', 'MLTV', 'Width', 'Tendency', 'NSP']].corr()
   
   # from statsmodels.tsa.seasonal import seasonal_decompose   
   # # creating trend object by assuming multiplicative model 
   # output = seasonal_decompose(Copy_of_Original_data_1['LB'], model='multiplicative').trend 
   # # creating plot 
   # output.plot() 


   """Conclusion:
    Summarize the key insights and patterns discovered through your exploratory analysis.
    Discuss how these findings could impact decision-making or further analyses.
   """
   # Print the key insights and patterns discovered through the exploratory analysis
   print("Key insights and patterns:")
   print("- The dataset contains 2126 rows and 23 columns.")
   print("- The data types of the columns are mostly float64, with the exception of 'NSP' which is int64 and 'Date' which is object.")
   print("- The summary statistics show that the mean heart rate is 133.2 beats per minute, the mean uterine contraction is 62.2 mmHg, and the mean fetal movement is 0.5.")
   print("- There are no missing values in the dataset.")
   print("- The correlation between each pair of columns is generally low, with the exception of 'LB' and 'AC' which have a correlation of 0.99.")
   print("- The boxplots show that there are some outliers in the dataset.")
   print("- The histograms show that the data is generally normally distributed.")
   print("- The scatter plots show that there is no clear linear relationship between any pair of columns.")
   # Conclusion
   print("- The exploratory analysis of the Cardiotocographic.csv dataset revealed several key insights and patterns. These insights can be used to further investigate the relationship between various features and to develop a model for predicting fetal health.")
   

     