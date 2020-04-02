# AirBnb-price-prediction
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:53:46 2020

@author: nikitagupta
"""

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns


# Importing the dataset

data_train = pd.read_csv('train10k.csv')

data_test = pd.read_csv('test2k.csv')


print("\n Train data size : {} ".format(data_train.shape))
print("Test data size : {} ".format(data_test.shape))

data_train.info()
data_test.head()
y=data_train.loc[:,'log_price']
print(data_train.info())

data_train[['city', 'log_price']].groupby(['city'], as_index=False).mean().sort_values(by='log_price',ascending=False)
g=sns.regplot(x=data_train['review_scores_rating'],y=data_train['log_price'],fit_reg=True)

#group property_type with log price
data_train[['property_type', 'log_price']].groupby(['property_type'], as_index=False).mean().sort_values(by='log_price',ascending=False).head(n=10)


#group accommodates with log price
data_train[['accommodates', 'log_price']].groupby(['accommodates'], as_index=False).mean().sort_values(by='log_price',ascending=False)

#identify the distance to center on the basis of latitude and longitude values from dataset
def lat(row):
    if (row['city']=='NYC'):
        return 40.72
    
def long(row):
    if (row['city']=='NYC'):
        return -74
#for training data
data_train['lat']=data_train.apply(lambda row: lat(row), axis=1)
data_train['long']=data_train.apply(lambda row: long(row), axis=1)

data_train['distance from center']=np.sqrt((data_train['lat']-data_train['latitude'])**2+(data_train['long']-data_train['longitude'])**2)

data_test['lat']=data_test.apply(lambda row: lat(row), axis=1)
data_test['long']=data_test.apply(lambda row: long(row), axis=1)

#for testing data
data_test['distance from center']=np.sqrt((data_test['lat']-data_test['latitude'])**2+(data_test['long']-data_test['longitude'])**2)

data_train.drop(["lat","long","log_price","host_response_rate","name","number_of_reviews","thumbnail_url","first_review","host_since","last_review","longitude","zipcode","latitude"], axis = 1, inplace = True)
data_test.drop(["lat","long","host_response_rate","name","number_of_reviews","thumbnail_url","first_review","host_since","last_review","longitude","zipcode","latitude"], axis = 1, inplace = True)

#label encoder to create categorical variable and fill 0 for blank values

from sklearn.preprocessing import LabelEncoder 
  
le = LabelEncoder() 

#encoding for train data
data_train['property_type']= le.fit_transform(data_train['property_type']) 
data_train['room_type']= le.fit_transform(data_train['room_type'])
data_train['amenities']= le.fit_transform(data_train['amenities'])
data_train['bed_type']= le.fit_transform(data_train['bed_type'])
data_train['cancellation_policy']= le.fit_transform(data_train['cancellation_policy'])
data_train['cleaning_fee']= le.fit_transform(data_train['cleaning_fee'])
data_train['city']= le.fit_transform(data_train['city'])
data_train['description']= le.fit_transform(data_train['description'])
data_train['neighbourhood']=data_train['neighbourhood'].fillna('na')
data_train['neighbourhood']= le.fit_transform(data_train['neighbourhood'])
data_train['host_has_profile_pic']=data_train['host_has_profile_pic'].fillna('na')
data_train['host_identity_verified']=data_train['host_identity_verified'].fillna('na')
data_train['instant_bookable']=data_train['instant_bookable'].fillna('na')
data_train['host_has_profile_pic']= le.fit_transform(data_train['host_has_profile_pic'])
data_train['host_identity_verified']= le.fit_transform(data_train['host_identity_verified'])
data_train['instant_bookable']= le.fit_transform(data_train['instant_bookable'])
data_train['review_scores_rating']=data_train['review_scores_rating'].fillna(0)
data_train['bedrooms']=data_train['bedrooms'].fillna(0)
data_train['beds']=data_train['beds'].fillna(0)
data_train['bathrooms']=data_train['bathrooms'].fillna(0)
data_train['distance from center']=data_train['distance from center'].fillna(0)

#encoding for test data
data_test['property_type']= le.fit_transform(data_test['property_type']) 
data_test['room_type']= le.fit_transform(data_test['room_type'])
data_test['amenities']= le.fit_transform(data_test['amenities'])
data_test['bed_type']= le.fit_transform(data_test['bed_type'])
data_test['cancellation_policy']= le.fit_transform(data_test['cancellation_policy'])
data_test['cleaning_fee']= le.fit_transform(data_test['cleaning_fee'])
data_test['city']= le.fit_transform(data_test['city'])
data_test['description']= le.fit_transform(data_test['description'])
data_test['neighbourhood']=data_test['neighbourhood'].fillna('na')
data_test['neighbourhood']= le.fit_transform(data_test['neighbourhood'])
data_test['host_has_profile_pic']=data_test['host_has_profile_pic'].fillna('na')
data_test['host_identity_verified']=data_test['host_identity_verified'].fillna('na')
data_test['instant_bookable']=data_test['instant_bookable'].fillna('na')
data_test['host_has_profile_pic']= le.fit_transform(data_test['host_has_profile_pic'])
data_test['host_identity_verified']= le.fit_transform(data_test['host_identity_verified'])
data_test['instant_bookable']= le.fit_transform(data_test['instant_bookable'])
data_test['review_scores_rating']=data_test['review_scores_rating'].fillna(0)
data_test['bedrooms']=data_test['bedrooms'].fillna(0)
data_test['beds']=data_test['beds'].fillna(0)
data_test['bathrooms']=data_test['bathrooms'].fillna(0)
data_test['distance from center']=data_test['distance from center'].fillna(0)

data_train.info()
data_test.info()


#from sklearn.ensemble import RandomForestRegressor

#model = RandomForestClassifier(n_estimators=100,bootstrap = True,max_features = 'sqrt')

X_train, X_test, y_train, y_test = train_test_split(data_train, y, test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestRegressor
    
reg = RandomForestRegressor(random_state = 0, n_estimators = 10)
#fitting the model on X_train, y_train
reg.fit(X_train,y_train)
#prediction the train model
y_pred1=reg.predict(X_test)
#predicting the test model
y_pred = reg.predict(data_test)

#Evaluating the root mean squared error
rmse = str(np.sqrt(np.mean((y_test - y_pred1)**2)))
    
print("RMSE value: " + rmse)

#Create the Submission file with predicted data and id.
    
    pred_df = pd.DataFrame(y_pred, index=data_test["id"])    
    pred_df.to_csv('Nikita_Submission.csv', header=True, index_label='id') 


