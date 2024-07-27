"""
1. Data Exploration and Preprocessing:
Load the dataset and conduct basic data exploration (summary statistics, missing values, data types).
Handle missing values as per the best practices (imputation, removal, etc.).
Apply scaling techniques to numerical features:
Standard Scaling
Min-Max Scaling
Discuss the scenarios where each scaling technique is preferred and why.
2. Encoding Techniques:
Apply One-Hot Encoding to categorical variables with less than 5 categories.
Use Label Encoding for categorical variables with more than 5 categories.
Discuss the pros and cons of One-Hot Encoding and Label Encoding.
3. Feature Engineering:
Create at least 2 new features that could be beneficial for the model. Explain the rationale behind your choices.
Apply a transformation (e.g., log transformation) to at least one skewed numerical feature and justify your choice.
4. Feature Selection:
Use the Isolation Forest algorithm to identify and remove outliers. Discuss how outliers can affect model performance.
Apply the PPS (Predictive Power Score) to find and discuss the relationships between features. Compare its findings with the correlation matrix.

"""

import pandas as pd

# ------------------------Data Load------------------------------
filepath = r'D:\EXCELR\Assignments\EDA2\adult_with_headers.csv'
# filepath = input("Please enter your filepath here: \n")
original_data = pd.read_csv(filepath)
print("------------------------------------------------------------------------------")
print("Below is the Original data set : \n", original_data)

# ------------------------Data Summary Information------------------------------
print("------------------------------------------------------------------------------")
# The below code is to check the summary statistics for the given data set
Summary_statistics = original_data.describe(include='all')
print("Below mentioned is the summary statistics for the given dataset: \n", Summary_statistics)

print("------------------------------------------------------------------------------")
# The below code is to check the data type for the given data set
Data_information = original_data.dtypes
print("Below mentioned is the data type for the given dataset: \n", Data_information)

# Verify the data for empty values
X = original_data.iloc[:,1:15]
X.head()

print("------------------------------------------------------------------------------")
# The below code is to check the missing values for the given data set
Missing_Values = X.isnull().sum()

print("Below mentioned is the missing values count for the given dataset: \n", Missing_Values)
print("There are no missing values shown in the given data set. But there exist some ? in age and workplace alse there are some 0 values.")


# Remove the ? in the current data with some valid values
X['workclass'].value_counts()
X['workclass']= X['workclass'].str.replace('?', 'Private')
X['workclass'].value_counts()

X['occupation'].value_counts()
X['occupation']= X['occupation'].str.replace('?', 'Prof-specialty ')
X['occupation'].value_counts()


X['native_country'].value_counts()
X['native_country']= X['native_country'].str.replace('?', 'United-States')
X['native_country'].value_counts()

X.columns
list_of_all_columns = ['workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
       'occupation', 'relationship', 'race', 'sex', 'capital_gain',
       'capital_loss', 'hours_per_week', 'native_country', 'income']


numeric_columns = X.select_dtypes(include='int64')
numeric_columns1 = numeric_columns
# Apply scaling techniques to numerical features:
# Scalor ---------------------------------------------------------
from sklearn.preprocessing import StandardScaler
data = numeric_columns
scaler = StandardScaler()
model = scaler.fit(data)
scaled_data = model.transform(data)
print(scaled_data)
print('''Standard Scaler helps to get standardized distribution, with a zero mean and standard deviation of one (unit variance). It standardizes features by subtracting the mean value from the feature and then dividing the result by feature standard deviation.''')
# Min Max ---------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler
data = numeric_columns1
mm_scaler = MinMaxScaler()
model=mm_scaler.fit(data)
min_max_data=model.transform(data)
print(min_max_data)
print('''There is another way of data scaling, where the minimum of feature is made equal to zero and the maximum of feature equal to one. MinMax Scaler shrinks the data within the given range, usually of 0 to 1. It transforms data by scaling features to a given range. It scales the values to a specific value range without changing the shape of the original distribution.''')
# -----------------------------------------------------

# Apply One-Hot Encoding to categorical variables with less than 5 categories.
# Use Label Encoding for categorical variables with more than 5 categories.
# Discuss the pros and cons of One-Hot Encoding and Label Encoding.
    
# One Hot coding technique for the categorical data set
from sklearn.preprocessing import OneHotEncoder

#Building a dummy employee dataset for example
data = X
#Converting into a Pandas dataframe
df = pd.DataFrame(data)
#Print the dataframe:
print(f"Employee data : \n{df}")

#Extract categorical columns from the dataframe
#Here we extract the columns with object datatype as they are the categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
from sklearn.preprocessing import LabelEncoder

for i in categorical_columns:
        if(len(i.value_counts) < 5):

            #Initialize OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False)
            
            # Apply one-hot encoding to the categorical columns
            one_hot_encoded = encoder.fit_transform(df[categorical_columns])
            
            #Create a DataFrame with the one-hot encoded columns
            #We use get_feature_names_out() to get the column names for the encoded data
            one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_columns))
            
            # Concatenate the one-hot encoded dataframe with the original dataframe
            df_encoded = pd.concat([df, one_hot_df], axis=1)
            
            # Drop the original categorical columns
            df_encoded = df_encoded.drop(categorical_columns, axis=1)
            
            # Display the resulting dataframe
            print(f"Encoded Employee data : \n{df_encoded}")
        else: 
            LE = LabelEncoder()
            for i in range(0,23,1):
                one_hot_encoded = encoder.fit_transform(df[categorical_columns])
                data = LE.fit_transform(df.iloc[:,i])

         