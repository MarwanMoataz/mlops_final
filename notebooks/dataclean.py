# %%
# I am using 3.9.7.
import sys
print("Python Version is: " + sys.version)

# %%
# importing libraries 
import pandas as pd
import numpy as np

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x) 
#pd.set_option('display.max_rows', None)


# %%
# Loading dataset labeled "Telecom Customer Churn" 
# Link: https://www.mavenanalytics.io/data-playground 

import time
time_begin = time.time()

df = pd.read_csv("telecom_customer_churn.csv")

print(f'Run time: {round(((time.time()-time_begin)/60), 3)} mins') 

# I like to print the time it takes to load a file

# %% [markdown]
# ### Macro Exploring the Data

# %%
df.shape

# %%
df.head(4)

# %%
df.info()
# we can see some columns have missing data. Lets check that out. 

# %%
df.isnull().sum()

# %%
df[df['MultipleLines'].isnull()].head(3)

# %%
df[df['StreamingMusic'].isnull()].head(3)

# We can see that when Internet Service = No then the Internet columns have NaN because it's not available. 

# %%
df.describe(exclude = ['O']) 
# excludes object data types and describes them

# %%
df.describe(include = ['O'])
# Includes object data types and describes them

# %% [markdown]
# ### Data Cleaning

# %%
# A preference of mine is to replace binary columns with 1,0. 
df['Churn']=df['Churn'].replace('Joined', 'Stayed')
df['Churn']=df['Churn'].replace(['Stayed','Churned'], [0,1])
df['Married']=df['Married'].replace(['Yes','No'],[1,0])
df['Gender'] = df['Gender'].replace(['Female','Male'],[0,1]) # 1 = male, # 0 = female
df['PaperlessBilling']=df['PaperlessBilling'].replace(['Yes','No'],[1,0])
df['InternetService']=df['InternetService'].replace(['Yes','No'],[1,0])
df['PhoneService']=df['PhoneService'].replace(['Yes','No'],[1,0]) #1 = Yes, 0 = No


# Filling these specifically with 0 rather than the mean because it's already a "avg".
df['AvgMonthlyGBDownload'] = df['AvgMonthlyGBDownload'].fillna(0)
df['AvgMonthlyLongDistanceCharges'] = df['AvgMonthlyLongDistanceCharges'].fillna(0)

# %%
# Filling all na fields that contain text with "unknown" rather than delete or remove them from the datset.

cols_to_change= ['ChurnCategory','ChurnReason', 'UnlimitedData','StreamingMusic',
                  'StreamingMovies','StreamingTV','PremiumTechSupport','DeviceProtectionPlan'
                  ,'OnlineSecurity','OnlineBackup','InternetType','MultipleLines']

df[cols_to_change]=df[cols_to_change].fillna('Unknown')

# %%
# Double checking my work to ensure they are all zero for NULLS
df.isnull().sum()

# %% [markdown]
# ### Data Exploration

# %%
df['Churn'].value_counts()
# 0 = did not exit

# %%
#Removing outliers from two columns that encompass the entire data set. 

from scipy import stats
df=df[(np.abs(stats.zscore(df['TotalRevenue'])) < 3)]
df=df[(np.abs(stats.zscore(df['Population'])) < 3)]

# %%
df['CustomerID'].count()

# %%
df.head(2)

# %%
df['Churn'].value_counts()
# 0 = did not exit

# %% [markdown]
# ### Exporting Cleaned Data to a new CSV. This will be used for importing to Machine Learning Model

# %%
df.to_csv('CleanedDF.csv', index = False)


