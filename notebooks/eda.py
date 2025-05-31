# %%
import sys
print("Python Version is: " + sys.version)



# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

#SNS Settings 
sns.set(color_codes = True)
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(7,4)})
sns.set_palette("Set3")

# Get multiple outputs in the same cell
#from IPython.core.interactiveshell import InteractiveShell
#InteractiveShell.ast_node_interactivity = "all"

# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x) 
#pd.set_option('display.max_rows', None)


# %%
import time
time_begin = time.time()

df = pd.read_csv("telecom_customer_churn.csv") # data = pd.read_csv("census.csv")

# Convert Churn to numeric: 0 = Stayed/Joined, 1 = Churned
df['Churn'] = df['Churn'].replace({'Stayed': 0, 'Churned': 1, 'Joined': 0})
df['Churn'] = pd.to_numeric(df['Churn'], errors='coerce')

print(f'Run time: {round(((time.time()-time_begin)/60), 3)} mins')

print(df['Churn'].unique())

# %%
df.sample(2)

# %%
# df.describe().transpose()[:-3] # Removing the last three as are "churn predicitons and probabilities "

# %%
df.describe(include = ['O']).transpose()[1:] 
#removes CustomerID from the results
# O = Object

# %%
df.describe(exclude = ['O']).transpose()[:-3]

# %% [markdown]
# ### Determining how many "churned" 

# %%
df['Churn'].value_counts()
# 0 = did not exit

# %% [markdown]
# ### Comparing churned to various features

# %%
df.groupby(['Married']).agg({'Churn': 'mean'}).reset_index().sort_values(by='Churn', ascending=False)
# If you are not married, your chances of churning are on average 33%
# If you are marrried, your chances of churning are on average 20% 

# %%
df.groupby(['Gender']).agg({'Churn': 'mean'}).reset_index().sort_values(by='Churn', ascending=False)
# If you are a female (0), your chances of churning are on average 27%
# If you are a male (1), your chances of churning are on average 26%

# %%
df.groupby(['Gender','Married']).agg({'Churn': 'mean'}).reset_index().sort_values(by='Churn', ascending=False)
# The highest churn grouping is a female who is not married (34%). 
# Next is a male who is not married (32%). 

# %%
# histo_plots = pd.DataFrame(data =df, columns =  ['Total Revenue','Total Long Distance Charges'])

# %%
g = sns.FacetGrid(df, col = "Married", row = 'Gender')
g = g.map(plt.hist, "Age")
# 0 = Female, 1 = Male
# 0 = Not married, 1 = married

# %%
#cols = ['Experience', 'Mortgage']
fig, [ax0, ax1, ax2] = plt.subplots(1,3, figsize = (14,4))

ax0.hist(df['TotalRevenue'])
ax0.set_xlabel('Total Revenue Distribution')
ax0.axvline(df['TotalRevenue'].mean(), color = "black")

ax1.hist(df.TotalCharges)
ax1.set_xlabel('Total Charges Distribution')
ax1.axvline(df["TotalCharges"].mean(), color = "black");

ax2.hist(df.TenureinMonths)
ax2.set_xlabel('Tenure in Month distribution')
ax2.axvline(df['TenureinMonths'].mean(), color = "black");

print("Black lines are means")

# %%
fig, [ax0, ax1, ax2] = plt.subplots(1,3, figsize = (14,4))

ax0.hist(df['Age'])
ax0.set_xlabel('Age Distribution')
ax0.axvline(df['Age'].mean(), color = "black")

ax1.hist(df['TotalLongDistanceCharges'])
ax1.set_xlabel('Total Long Distance Charges Distribution')
ax1.axvline(df["TotalLongDistanceCharges"].mean(), color = "black");

ax2.hist(df['Population'])
ax2.set_xlabel('Population Distribution')
ax2.axvline(df['Population'].mean(), color = "black");

print("Black lines are means")

# %%
sns.countplot(y = df['TotalExtraDataCharges'])

# %%
sns.violinplot(x=df['TotalRefunds'])

# %%
sns.boxplot(x=df['MonthlyCharge'])

# %%
sns.countplot(df['PaymentMethod'])

# %%
sns.countplot(y=df['ChurnCategory'],order = df['ChurnCategory'].value_counts().index)

# %%
plt.figure(figsize=(10,10))
sns.countplot(y=df['ChurnReason'],order = df['ChurnReason'].value_counts().index)

# %%
sns.countplot(df['PhoneService'])

# %%
sns.countplot(data=df, x='PaperlessBilling', hue='Churn')
plt.show()

# %%
sns.countplot(
    data=df,
    x='Contract',
    hue='Churn',
    order=df['Contract'].value_counts().index
)
plt.show()


# %%
sns.countplot(
    data=df,
    x='Offer',
    hue='Churn',
    order=df['Offer'].value_counts().index
)
plt.show()


# %%
sns.countplot(df['InternetType'], order = df['InternetType'].value_counts().index)

# %% [markdown]
# ## Exploratory data analysis

# %% [markdown]
# ###  Is there some association between personal characteristics and those who churned?
# #### QUANTATIVE VARIABLES

# %%
quant_df1 = df[[ #nominal
 'Churn',
 'Age',
 'NumberofDependents',
 'Population',
 'NumberofReferrals',
 'TenureinMonths',
 'AvgMonthlyLongDistanceCharges',
'AvgMonthlyGBDownload',
 'MonthlyCharge',
 'TotalCharges',
 'TotalRefunds',
 'TotalExtraDataCharges',
 'TotalLongDistanceCharges',
 'TotalRevenue'
]].copy()

# categorical columns

quant_df2 = df[[ #binary
#'Churn',
'Gender',
'Married',
'PhoneService',
'InternetService',
'PaperlessBilling'
]].copy()

quant_df3 = df[[ #categories (one-hot-encode)
 #'Churn',
 'Offer',
 #'MultipleLines',
 #'InternetType',
 #'OnlineSecurity',
 'OnlineBackup',
 #'DeviceProtectionPlan',
 'PremiumTechSupport',
 'StreamingTV',
 'StreamingMovies',
 'StreamingMusic',
 #'UnlimitedData',
 'Contract',
 #'PaymentMethod',
]].copy()


# %%
quant_df1.corr()

# %%
plt.figure(figsize=(10,10))
cmap = sns.diverging_palette(250, 10, as_cmap=True)
sns.heatmap(quant_df1.corr(), cmap = cmap, annot = True);


# %%
# get association coefficients for 'Personal Loan' and exclude it's data from series
quant_df1.corr()['Churn'][1:]

# %%
print(quant_df1.columns)

# %%
# quant_df1.corr()['Churn'][1:].plot.bar()
# plt.axhline(y = 0.05);

# %% [markdown]
# Let's check our confidense about this statment with logistic regression model:

# %%


# %%
quant_df1['intercept'] = 1
# First, let's check for missing values
print("Missing values in each column:")
print(quant_df1.isnull().sum())

# Fill missing values with appropriate methods
quant_df1_clean = quant_df1.copy()
# For numerical columns, fill with median
numeric_cols = ['Age', 'NumberofDependents', 'Population', 'NumberofReferrals',
                'TenureinMonths', 'AvgMonthlyLongDistanceCharges', 'AvgMonthlyGBDownload',
                'MonthlyCharge', 'TotalCharges', 'TotalRefunds', 'TotalExtraDataCharges',
                'TotalRevenue']
for col in numeric_cols:
    quant_df1_clean[col] = quant_df1_clean[col].fillna(quant_df1_clean[col].median())

# Now fit the model with cleaned data
log_mod = sm.Logit(quant_df1_clean['Churn'], 
                   quant_df1_clean[['intercept'] + numeric_cols]).fit()
log_mod.summary()

# %%
# p values
log_mod.pvalues[:].plot.bar()
plt.axhline(y = 0.05);

# %%
#coefficient
log_mod.params[:].plot.bar()
plt.axhline(y = 0.05);

# %%
quant_df_main = {}
for i in log_mod.params[:].to_dict().keys():
    if log_mod.pvalues[i] < 0.05:
        quant_df_main[i] = log_mod.params[i]
    else:
        continue
quant_df_main
sorted(quant_df_main.items(), key=lambda x: x[1]) #sorting by highest to lowest value

# %%


# %% [markdown]
# #### Compute the odds

# %%
quant_df_main_odds = {k : np.exp(v) for k, v in quant_df_main.items()}
sorted(quant_df_main_odds.items(), key=lambda x: x[1]) #sorting by highest to lowest value

# %% [markdown]
# ### Analysis
# NumberofDependents, NumberofReferrals,TenureinMonths, AvgMonthlyLongDistanceCharges,TotalCharges are all negatively associated
# 
# Population, TotalRevenue, Age, MonthlyCharge are positively associated

# %%
quant_df1['intercept'] = 1
log_mod2 = sm.Logit(quant_df1['Churn'], quant_df1[['intercept','Age','Population', 'MonthlyCharge','TotalRevenue']]).fit()
log_mod2.summary()

# %% [markdown]
# ### Log Mod 2 (Binary)

# %%
quant_df2.info()

# %%
# For the binary variables model (quant_df2)
# First check the data types
print("\nData types of binary variables:")
print(quant_df2.dtypes)

# Convert binary variables to numeric if they aren't already
binary_cols = ['Gender', 'Married', 'PhoneService', 'InternetService', 'PaperlessBilling']
quant_df2_clean = quant_df2.copy()

# Check for missing values before conversion
print("\nMissing values before conversion:")
print(quant_df2_clean.isnull().sum())

# Convert to numeric and handle missing values
for col in binary_cols:
    if quant_df2_clean[col].dtype == 'object':
        # First convert to numeric, coercing errors to NaN
        quant_df2_clean[col] = pd.to_numeric(quant_df2_clean[col], errors='coerce')
        # Then fill NaN with the most common value (mode)
        mode_value = quant_df2_clean[col].mode()
        if not mode_value.empty:
            quant_df2_clean[col] = quant_df2_clean[col].fillna(mode_value.iloc[0])
        else:
            # If no mode exists, fill with 0 (assuming binary variables)
            quant_df2_clean[col] = quant_df2_clean[col].fillna(0)

# Check for missing values after conversion
print("\nMissing values after conversion:")
print(quant_df2_clean.isnull().sum())

# Add intercept
quant_df2_clean['intercept'] = 1

# Now fit the model with cleaned binary data
log_mod2 = sm.Logit(quant_df1['Churn'], quant_df2_clean).fit()
log_mod2.summary()

# %%
quant_df_main2 = {}
for i in log_mod2.params[:].to_dict().keys():
    if log_mod2.pvalues[i] < 0.05:
        quant_df_main2[i] = log_mod2.params[i].round(4)
    else:
        continue
quant_df_main2
sorted(quant_df_main2.items(), key=lambda x: x[1]) #sorting by highe

# %%
quant_df_main_odds2 = {k : np.exp(v) for k, v in quant_df_main2.items()}
sorted(quant_df_main_odds2.items(), key=lambda x: x[1]) #sorting by highest to lowest value

# %% [markdown]
# ### Log Mod 3 (Category)

# %%
quant_df3 = df[[ #categories (one-hot-encode)
 'Churn',
 'Offer',
 #'MultipleLines',
 'InternetType',
 #'OnlineSecurity',
 #'OnlineBackup',
 #'DeviceProtectionPlan',
 #'PremiumTechSupport',
 #'StreamingTV',
 #'StreamingMovies',
 #'StreamingMusic',
 #'UnlimitedData', #Seems to be the issue
 'Contract',
 'PaymentMethod',
]].copy()

# %%
quant_df3.head()

# %%
quant_df3 = pd.get_dummies(quant_df3)

# %%
#for i in quant_df3:
#    quant_df3[i] = quant_df3[i]*100
    


# %%
#quant_df3['Churn']=quant_df3['Churn']/100

# %%
quant_df3.describe().transpose()

# %%
quant_df3.shape

# %%
print(quant_df3.columns)


# %%
#quant_df3['intercept'] = 1
log_mod3 = sm.Logit(quant_df3['Churn'], quant_df3[['Offer_Offer A','Offer_Offer B',
                                                   'Offer_Offer C','Offer_Offer E','Offer_Offer D']]).fit()
log_mod3.summary()

# %%
#quant_df3['intercept'] = 1
log_mod3 = sm.Logit(quant_df3['Churn'], quant_df3[['Offer_Offer A','Offer_Offer B',
                                                   'Offer_Offer C','Offer_Offer E','Offer_Offer D']]).fit()
log_mod3.summary()

# we get the dummy variable trip. Research said to drop intercept 
# https://www.algosome.com/articles/dummy-variable-trap-regression.html

# %%
quant_df_main = {}
for i in log_mod3.params[:].to_dict().keys():
    if log_mod3.pvalues[i] < 0.05:
        quant_df_main[i] = log_mod3.params[i].round(4)
    else:
        continue
quant_df_main
sorted(quant_df_main.items(), key=lambda x: x[1]) #sorting by highest to lowest value

# %%
quant_df3['intercept'] = 1
log_mod3 = sm.Logit(quant_df3['Churn'], quant_df3[['Contract_Month-to-Month',
                                                   'Contract_One Year','Contract_Two Year']]).fit()
log_mod3.summary()

# %%
quant_df3['intercept'] = 1
log_mod3 = sm.Logit(quant_df3['Churn'], quant_df3[['PaymentMethod_Bank Withdrawal','PaymentMethod_Credit Card',
                                                  'PaymentMethod_Mailed Check']]).fit()
log_mod3.summary()

# %%


# %%
quant_df3['intercept'] = 1
log_mod3 = sm.Logit(quant_df1['Churn'], quant_df3[['InternetType_Cable','InternetType_DSL',
                                                  'InternetType_Fiber Optic','InternetType_Unknown']]).fit()
log_mod3.summary()

# %%


# %%
quant_df1_filled = quant_df1.fillna(0)



