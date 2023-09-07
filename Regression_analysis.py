#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import string
from datetime import datetime

from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col
from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df_reviews = pd.read_csv("reviews_final.csv")
df_listings = pd.read_csv("listings_final.csv")
df_sentiment = pd.read_csv("sentiment_all.csv")


# In[3]:


df_listings = df_listings[['id', 'host_id', 'host_since',
                           'host_response_time', 'host_response_rate', 'host_acceptance_rate', 
                           'host_is_superhost', 'host_verifications', 
                           'host_has_profile_pic', 'host_identity_verified',
                           'price', 'review_scores_rating', 'instant_bookable']]


# In[4]:


# Null Value Percentage Per column
percent_missing = df_listings.isnull().sum() * 100 / len(df_listings)
missing_value_df = pd.DataFrame({'column_name': df_listings.columns,
                                 'percent_missing': percent_missing})
missing_value_df


# In[5]:


df_listings.head()


# # Cleaning data

# In[6]:


# Convert columns to the appropriate data types
df_listings['host_response_rate'] = df_listings['host_response_rate'].astype(str).str.replace('%', '').astype(float) / 100
df_listings['host_acceptance_rate'] = df_listings['host_acceptance_rate'].astype(str).str.replace('%', '').astype(float) / 100
df_listings['review_scores_rating'] = df_listings['review_scores_rating'].astype(float)


# In[7]:


# fill null value by mean imputation
columns = ['review_scores_rating', 'host_acceptance_rate', 'host_response_rate' ]

for column in columns:
    mean = df_listings[column].mean()
    df_listings[column].fillna(mean, inplace=True)


# In[8]:


# change variable 't' and 'f' to 1 and 2
variables_to_replace = ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic', 'instant_bookable']

for variable in variables_to_replace:
    df_listings[variable] = df_listings[variable].replace({'t': 1, 'f': 2})


# In[9]:


# Adjusted price 
df_listings['price'] = df_listings['price'].apply(lambda x: float(x.replace('$', '').replace(',', '')) if isinstance(x, str) else x)


# In[10]:


# Convert 'host_since' column to datetime format
df_listings['host_since'] = pd.to_datetime(df_listings['host_since'])

# Define the target date
target_date = datetime(2023, 3, 29)

# Calculate the number of days until the target date
df_listings['host_since_days'] = (target_date - df_listings['host_since']).dt.days


# In[11]:


df_listings.isnull().sum()


# In[12]:


df1 = df_listings[['id','review_scores_rating','host_is_superhost', 
                   'host_response_rate', 'host_acceptance_rate', 'host_identity_verified', 'price',
                   'host_has_profile_pic', 'instant_bookable', 'host_since_days'
                  ]]


# In[13]:


df1.info()


# # test with sentiment score

# In[14]:


df_sentiment['listing_id'].isin(df1['id']).value_counts()


# In[15]:


df1['id'].isin(df_sentiment['listing_id']).value_counts()


# In[16]:


df_test = pd.merge(df1[['id',
                         'review_scores_rating', 
                        'host_is_superhost', 'host_response_rate', 
                        'host_acceptance_rate', 'host_identity_verified', 
                        'price', 'host_has_profile_pic', 'instant_bookable', 'host_since_days']],
                   
                   
                    df_sentiment[['listing_id','polarity', 'sentiment']],
              left_on = 'id', 
              right_on = 'listing_id')


# df_test.info()

# In[17]:


df_test.isnull().sum()


# # Check multicollinearity

# In[18]:


X = df_test[['host_is_superhost', 'host_response_rate', 'host_acceptance_rate',
        'host_identity_verified', 'price', 'host_has_profile_pic', 'instant_bookable', 'host_since_days']]


# Calculate the correlation matrix
corr_matrix = X.corr()

# Perform the eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(corr_matrix)

print("Eigenvalues:")
print(eigenvalues)

print("Eigenvectors:")
print(eigenvectors)


# In[21]:


# Create a heatmap
sns.heatmap(corr_matrix, annot=True)

# Show the plot
plt.show()


# # Regression w/ rating score

# In[22]:


# Define the dependent variable
y = df_test['review_scores_rating']

# Define the independent variables (excluding non-numeric columns)
X = df_test[['host_is_superhost', 'host_response_rate', 'host_acceptance_rate',
        'host_identity_verified', 'price', 'host_has_profile_pic', 'instant_bookable', 'host_since_days']]

# Define the variable names
var_names = ['const'] + list(X.columns)

# Step 1: Normalize Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: OLS Regression with clustered standard errors
X_with_const = sm.add_constant(X_scaled)  # Adding a constant term
cluster_var = df_test['listing_id']
model = sm.OLS(y, X_with_const).fit(cov_type = 'cluster', cov_kwds = {'groups': cluster_var})

# Print the regression summary
print('Model 1: Review rating score - Host Reputation Indicators')
print(model.summary(xname = var_names))


# # Regression w/ polarity

# In[23]:


# Define the dependent variable
y = df_test['polarity']

# Define the independent variables (excluding non-numeric columns)
X = df_test[['host_is_superhost', 'host_response_rate', 'host_acceptance_rate',
        'host_identity_verified', 'price', 'host_has_profile_pic', 'instant_bookable', 'host_since_days']]

# Define the variable names
var_names = ['const'] + list(X.columns)

# Step 1: Normalize Z-score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: OLS Regression with clustered standard errors
X_with_const = sm.add_constant(X_scaled)  # Adding a constant term
cluster_var = df_test['listing_id']
model = sm.OLS(y, X_with_const).fit(cov_type='cluster', cov_kwds={'groups': cluster_var})

# Print the regression summary
print('Model 2: Sentiment - Host Reputation Indicators')
print(model.summary(xname=var_names))

