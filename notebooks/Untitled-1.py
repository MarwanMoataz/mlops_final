# %%
import sys
print("Python Version is: " + sys.version)



# %%
import time
import pandas as pd
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib #load model
import subprocess
from sklearn import feature_selection
#
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
#
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression,RidgeClassifier,SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.preprocessing import OneHotEncoder , OrdinalEncoder,StandardScaler , MinMaxScaler, MaxAbsScaler
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import RFE
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, confusion_matrix, classification_report,f1_score,accuracy_score 
from sklearn.dummy import DummyClassifier
from catboost import CatBoostClassifier

#imblen learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline
 
# Get multiple outputs in the same cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
 
# Ignore all warnings
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x) #goes to two decimal places

# %%
import time
time_begin = time.time()

df = pd.read_csv("CleanedDF.csv") # data = pd.read_csv("census.csv")

print(f'Run time: {round(((time.time()-time_begin)/60), 3)} mins')

# %%
X = df
y = df['Churn']

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.3, 
                                                    random_state=42,
                                                    stratify=y,
                                                    shuffle = True
                                                   )

# %%
remove = ['CustomerID','ZipCode','City', 'ChurnCategory','ChurnReason','Churn']

# numercical columns
num_feats = [ 
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
]
# categorical columns
cat_feats = [ 
 'Gender',
 'Offer',
 'Married',
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'InternetService',
 'InternetType',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtectionPlan',
 'PremiumTechSupport',
 'StreamingTV',
 'StreamingMovies',
 'StreamingMusic',
 'UnlimitedData',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod',
]

# %%
X_test['Churn'].value_counts()

# %%
X_testcopy = X_test.copy()
X_testcopy.sample(2)

# %%
X_test.drop(remove, axis = 1, inplace = True)
X_train.drop(remove, axis = 1, inplace = True)

# %%
#X_test.drop(['Churn'], axis = 1, inplace = True)
#X_train.drop(['Churn'], axis = 1, inplace = True)
#X_test.sample(2)

# %%
def get_pipeline(X, model): 

    numeric_pipeline = SimpleImputer(strategy='mean')
    categorical_pipeline = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_pipeline, num_feats),
            ('categorical', categorical_pipeline, cat_feats),
            ], remainder='passthrough'
    )

    bundled_pipeline = imbpipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('scaler', MinMaxScaler()),
        ('model', model)
    ])
    
    return bundled_pipeline

# %%
def select_model(X, y, pipeline=None):  
    classifiers = {}
    classifiers.update({"DummyClassifier": DummyClassifier(strategy='most_frequent')})
    classifiers.update({"XGBClassifier": XGBClassifier(use_label_encoder=False, 
                                                       eval_metric='logloss',
                                                       objective='binary:logistic',
                                                      )})
    classifiers.update({"LGBMClassifier": LGBMClassifier()})
    classifiers.update({"RandomForestClassifier": RandomForestClassifier()})
    classifiers.update({"DecisionTreeClassifier": DecisionTreeClassifier()})
    classifiers.update({"ExtraTreeClassifier": ExtraTreeClassifier()})
    #classifiers.update({"ExtraTreesClassifier": ExtraTreeClassifier()})    
    classifiers.update({"AdaBoostClassifier": AdaBoostClassifier()})
    classifiers.update({"KNeighborsClassifier": KNeighborsClassifier()})
    classifiers.update({"RidgeClassifier": RidgeClassifier()})
    classifiers.update({"SGDClassifier": SGDClassifier()})
    classifiers.update({"BaggingClassifier": BaggingClassifier()})
    classifiers.update({"BernoulliNB": BernoulliNB()})
    classifiers.update({"SVC": SVC()})
    classifiers.update({"CatBoostClassifier":CatBoostClassifier(silent=True)})
    
    # Stacking
    models = []

    models = []
    models.append(('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')))
    models.append(('CatBoostClassifier', CatBoostClassifier(silent=True)))
    models.append(('BaggingClassifier', BaggingClassifier()))
    classifiers.update({"VotingClassifier (XGBClassifier, CatBoostClassifier, BaggingClassifier)": VotingClassifier(models)})

    models = []
    models.append(('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')))
    models.append(('LGBMClassifier', LGBMClassifier()))
    models.append(('CatBoostClassifier', CatBoostClassifier(silent=True)))
    classifiers.update({"VotingClassifier (XGBClassifier, LGBMClassifier, CatBoostClassifier)": VotingClassifier(models)})
    
    models = []
    models.append(('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')))
    models.append(('RandomForestClassifier', RandomForestClassifier()))
    models.append(('DecisionTreeClassifier', DecisionTreeClassifier()))
    classifiers.update({"VotingClassifier (XGBClassifier, RandomForestClassifier, DecisionTreeClassifier)": VotingClassifier(models)})

    models = []
    models.append(('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')))
    models.append(('AdaBoostClassifier', AdaBoostClassifier()))
    #models.append(('ExtraTreeClassifier', ExtraTreeClassifier()))
    classifiers.update({"VotingClassifier (XGBClassifier, AdaBoostClassifier)": VotingClassifier(models)})
    
    models = []
    models.append(('XGBClassifier', XGBClassifier(use_label_encoder=False, eval_metric='logloss', objective='binary:logistic')))
    #models.append(('ExtraTreesClassifier', ExtraTreesClassifier()))
    classifiers.update({"VotingClassifier (XGBClassifier)": VotingClassifier(models)})    
    
    df_models = pd.DataFrame(columns=['model', 'run_time', 'accuracy'])

    for key in classifiers:
        
        start_time = time.time()

        pipeline = get_pipeline(X_train, classifiers[key])
        
        cv = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')

        row = {'model': key,
               'run_time': format(round((time.time() - start_time)/60,2)),
               'accuracy': cv.mean(),
        }

        df_models = pd.concat([df_models, pd.DataFrame([row])], ignore_index=True)
        
    df_models = df_models.sort_values(by='accuracy', ascending=False)
    return df_models

# %%
models = select_model(X_train, y_train)

# %%
models.sort_values(by=['accuracy','run_time'], ascending=False)

# %%
basemodel = XGBClassifier(use_label_encoder = False, eval_metric='logloss', objective='binary:logistic')

# %%
bundled_pipeline = get_pipeline(X_train, basemodel)

# %%
bundled_pipeline.fit(X_train, y_train)

# %%
basemodel_y_pred = bundled_pipeline.predict(X_test)

# %%
print(classification_report(y_test, basemodel_y_pred))
print(confusion_matrix(y_test, basemodel_y_pred))

# %% [markdown]
# ### Runing Test/Train split through pipeline w/hyper-parameter tuning

# %%
time_begin = time.time() #starts timer

#Loan Model

model = XGBClassifier(
    use_label_encoder = False, eval_metric='logloss', objective='binary:logistic', learning_rate = 0.1,
                     n_estimators = 1000, max_depth = 9, min_child_weight = 1, gamma = 0.4, colsample_bytree = 0.8, 
                      subsample = 0.9, reg_alpha = 1, scale_pos_weight = 1)

model = get_pipeline(X_train,model)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

#predict target probabilities
test_prob = model.predict_proba(X_test)[:,1]

test_pred = np.where(test_prob > 0.45, 1, 0) #sets the probability threshhold and can be tweaked

#test set metrics
roc_auc_score(y_test, test_pred)
recall_score(y_test, test_pred)
confusion_matrix(y_test, test_pred)

print(classification_report(y_test,test_pred))

print(f'Run time: {round(((time.time()-time_begin)/60), 3)} mins')

# adding predictions and their probabilities to the original test Data frame
X_testcopy['predictions'] = test_pred
X_testcopy['pred_probabilities'] = test_prob

high_churn_list = X_testcopy[X_testcopy.pred_probabilities > 0.0].sort_values(by=['pred_probabilities'], ascending = False
                                                                             ).reset_index().drop(columns=['index'],axis=1)


# %%
high_churn_list.to_csv('high_churn_list_model.csv', index = False)

# %%



