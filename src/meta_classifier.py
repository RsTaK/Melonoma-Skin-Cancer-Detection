
import re
import os

import math
import time

import random

import numpy as np
import pandas as pd

import seaborn as sns
import plotly.express as px

import joblib

import lightgbm as lgb

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec

from sklearn.metrics import roc_auc_score

from tqdm import tqdm
from tqdm.keras import TqdmCallback

import warnings
seed_val = 42
random.seed(seed_val)
np.random.seed(seed_val)

df_2=pd.read_csv("/content/b0_384_oof (1).csv",index_col="image_name")
df_1=pd.read_csv("./train.csv")
df_2=pd.read_csv("./train2.csv")

df_train=df_1.append(df_2)
di = {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:11,12:12,13:13,14:14,15:0,16:1,17:2,18:3,19:4,20:5,21:6,22:7,23:8,24:9,25:10,26:11,27:12,28:13,29:14}

df_train = df_train.replace({'tfrecord':di})
df_train = df_train[df_train.patient_id!=-1]

img_dir="/content/train"
df_test = pd.read_csv('test.csv')

def missing_percentage(df):

    total = df.isnull().sum().sort_values(
        ascending=False)[df.isnull().sum().sort_values(ascending=False) != 0]
    percent = (df.isnull().sum().sort_values(ascending=False) / len(df) *
               100)[(df.isnull().sum().sort_values(ascending=False) / len(df) *
                     100) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_train = missing_percentage(df_train)
missing_test = missing_percentage(df_test)

print('----Train----')
print(missing_train)
print('\n\n')
print('----Test----')
print(missing_test)

# Imputing missing values

# 1) anatom_site_general_challenge
for df in [df_train, df_test]:
  df['anatom_site_general_challenge'].fillna('unknwon', inplace=True)

# 2) sex
df_train['sex'].fillna(df_train['sex'].mode()[0], inplace=True)

# 3) age_approx
df_train['age_approx'].fillna(df_train['age_approx'].median(), inplace=True) 

print(
    f'Train missing value count: {df_train.isnull().sum().sum()}\nTest missing value count: {df_test.isnull().sum().sum()}'
)

print(
    f'Number of unique Patient ID\'s in train set: {df_train.patient_id.nunique()}, Total: {df_train.patient_id.count()}\nNumber of unique Patient ID\'s in test set: {df_test.patient_id.nunique()}, Total: {df_test.patient_id.count()}'
)

df_train['age_min'] = df_train['patient_id'].map(df_train.groupby(['patient_id']).age_approx.min())
df_train['age_max'] = df_train['patient_id'].map(df_train.groupby(['patient_id']).age_approx.max())

df_test['age_min'] = df_test['patient_id'].map(df_test.groupby(['patient_id']).age_approx.min())
df_test['age_max'] = df_test['patient_id'].map(df_test.groupby(['patient_id']).age_approx.max())

df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())


# Making the red, green and blue parameters

train_path = './train/'
test_path = './test/'

df_train['red'] = -1
df_train['blue'] = -1
df_train['green'] = -1
df_train['mean_colors'] = -1
df_test['red'] = -1
df_test['blue'] = -1
df_test['green'] = -1
df_test['mean_colors'] = -1


for i in tqdm(range(len(df_train))):
  img = mpimg.imread(f'{train_path}{df_train.iloc[i]["image_name"]}.jpg')
  r = img[:,:,0].mean()
  g = img[:,:,1].mean()
  b = img[:,:,2].mean()
  mean = (r+g+b)/3.0
  df_train.loc[i,'red'] = r
  df_train.loc[i,'green'] = g
  df_train.loc[i,'blue'] = b
  df_train.loc[i,'mean_colors'] = mean

for i in tqdm(range(len(df_test))):
  img = mpimg.imread(f'{test_path}{df_test.iloc[i]["image_name"]}.jpg')
  r = img[:,:,0].mean()
  g = img[:,:,1].mean()
  b = img[:,:,2].mean()
  mean = (r+g+b)/3.0
  df_test.loc[i,'red'] = r
  df_test.loc[i,'green'] = g
  df_test.loc[i,'blue'] = b
  df_test.loc[i,'mean_colors'] = mean


# df_train.to_csv('train_rgb.csv',index=False)
# df_test.to_csv('test_rgb.csv', index=False)
df_train = pd.read_csv('train_rgb.csv')
df_test = pd.read_csv('test_rgb.csv')

# Load the folds
temp = pd.read_csv("b0_384_oof.csv")
temp = temp[temp.folds!=-1]
df_train['folds'] = temp.folds

df_train = df_train[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'target', 'tfrecord', 'folds', 'age_min', 'age_max', 'red', 'green', 'blue', 'mean_colors']]
df_test = df_test[['image_name', 'patient_id', 'sex', 'age_approx', 'anatom_site_general_challenge', 'age_min', 'age_max', 'red', 'green', 'blue', 'mean_colors']]

print(f'train columns : {df_train.columns}\n')
print(f'test columns : {df_test.columns}')

# Label Encoding

def label_encode(train, test):
  lbe = LabelEncoder()
  lbe.fit(train.values)
  train = lbe.transform(train.values)
  test = lbe.transform(test.values)
  return lbe,train,test

# Handling Sex
# df_train['sex'] = df_train['sex'].astype('category')
# df_test['sex'] = df_test['sex'].astype('category')
# df_train['anatom_site_general_challenge'] = df_train['anatom_site_general_challenge'].astype('category')
# df_test['anatom_site_general_challenge'] = df_test['anatom_site_general_challenge'].astype('category')

lbe_sex, df_train['sex'], df_test['sex'] = label_encode(df_train['sex'], df_test['sex'])
lbe_site, df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge'] = label_encode(df_train['anatom_site_general_challenge'], df_test['anatom_site_general_challenge'])

#joblib.dump(lbe_sex,'./drive/My Drive/lbe_sex.bin')
#joblib.dump(lbe_site, './drive/My Drive/lbe_site.bin')

# Handling age_approx, age_min, age_max, width, 

def perform_minmax(train, test):
  mms = MinMaxScaler()
  mms.fit(train.values.reshape(-1,1))
  train = mms.transform(train.values.reshape(-1,1))
  test = mms.transform(test.values.reshape(-1,1))
  return mms, train, test

mms_age_approx, df_train['age_approx'], df_test['age_approx'] = perform_minmax(df_train['age_approx'], df_test['age_approx'])
mms_age_min, df_train['age_min'], df_test['age_min'] = perform_minmax(df_train['age_min'], df_test['age_min'])
mms_age_max, df_train['age_max'], df_test['age_max'] = perform_minmax(df_train['age_max'], df_test['age_max'])
mms_red, df_train['red'], df_test['red'] = perform_minmax(df_train['red'], df_test['red'])
mms_green, df_train['green'], df_test['green'] = perform_minmax(df_train['green'], df_test['green'])
mms_blue, df_train['blue'], df_test['blue'] = perform_minmax(df_train['blue'], df_test['blue'])
mms_mean_colors, df_train['mean_colors'], df_test['mean_colors'] = perform_minmax(df_train['mean_colors'], df_test['mean_colors'])
"""
pre_path = './drive/My Drive/'
joblib.dump(mms_age_approx, f'{pre_path}mms_approx.bin')
joblib.dump(mms_age_min, f'{pre_path}mms_min.bin')
joblib.dump(mms_age_max, f'{pre_path}mms_max.bin')
joblib.dump(mms_red, f'{pre_path}mms_red.bin')
joblib.dump(mms_green, f'{pre_path}mms_green.bin')
joblib.dump(mms_blue, f'{pre_path}mms_blue.bin')
joblib.dump(mms_mean_colors, f'{pre_path}mms_mean_colors.bin')"""

df_train = df_train[['image_name', 'patient_id', 'sex', 'age_approx', 
                     'anatom_site_general_challenge', 'age_min', 'age_max', 
                     'red', 'green', 'blue', 'mean_colors', 'target', 
                     'tfrecord', 'folds']]


def train_multi_fold_lgb(df_train, num_folds=5):

  model = lgb.LGBMClassifier(**{
      'learning_rate': 0.004,
      'num_leaves': 31,
      'max_bin': 1023,
      'min_child_samples': 700,
      'reg_alpha': 0.1,
      'reg_lambda': 0.2,
      'feature_fraction': 1.0,
      'bagging_freq': 1,
      'bagging_fraction': 0.85,
      'objective': 'binary',
      'n_jobs': -1,
      'n_estimators':600,})

  models = {}

  for fold in range(num_folds):
    # train
    train_x = df_train[df_train.folds!=fold].values[:,2:-3] # Excluding the folds and target column
    train_y = df_train[df_train.folds!=fold].target.values
    # valid
    valid_x = df_train[df_train.folds==fold].values[:,2:-3]
    valid_y = df_train[df_train.folds==fold].target.values

    model = model.fit(
        train_x, train_y, eval_set = (valid_x, valid_y),
        verbose=0, eval_metric='auc', early_stopping_rounds=100
    )

    models['fold_'+str(fold)] = model

  return models

def logit(p):
  return np.log(p) - np.log(1-p)

# Final results

def return_final_result(df_train, df_test, num_folds=5):

  oof_preds = {}
  test_preds = {}
  models = train_multi_fold_lgb(df_train)
  for fold in range(num_folds):
    oof_preds['fold_'+str(fold)] = models['fold_'+str(fold)].predict_proba(df_train.values[:,2:-3])
    test_preds['fold_'+str(fold)] = models['fold_'+str(fold)].predict_proba(df_test.values[:,2:])
  final_oof_preds = None
  final_test_preds = None
  for fold in range(num_folds):

    if final_oof_preds is None:
      final_oof_preds = logit(oof_preds['fold_'+str(fold)])
    else:
      final_oof_preds += logit(oof_preds['fold_'+str(fold)])

    if final_test_preds is None:
      final_test_preds = logit(test_preds['fold_'+str(fold)])
    else:
      final_test_preds += logit(test_preds['fold_'+str(fold)])

  final_oof_preds /= num_folds
  final_test_preds /= num_folds

  return final_oof_preds, final_test_preds

final_oof, final_test_preds = return_final_result(df_train, df_test)

sample=pd.read_csv("sample_submission.csv")

a=df_train.copy()

a.target=final_oof[:,1]


oof_pred = a.target.rank()
oof_target = df_train.target.rank()
roc_auc_score(oof_target,oof_pred)

oof_pred = pd.Series(final_oof[:,1]).rank()
oof_target = df_train.target.rank()