import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import seaborn as sns
import matplotlib.pyplot as plt

import gc

import lightgbm as lgb
from scipy.stats import rankdata

from sklearn.metrics import roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

b0=pd.read_csv("/content/b0_384_oof.csv")
b1=pd.read_csv("/content/b1_384_oof.csv")
b2=pd.read_csv("/content/b2_384_oof.csv")
b3=pd.read_csv("/content/b3_384_oof.csv")
b4=pd.read_csv("/content/b4_384_oof.csv")
b5=pd.read_csv("/content/b5_384_oof.csv")
b6=pd.read_csv("/content/b6_384_oof.csv")

b0_test=pd.read_csv("/content/b0_384_test.csv")
b1_test=pd.read_csv("/content/b1_384_test.csv")
b2_test=pd.read_csv("/content/b2_384_test.csv")
b3_test=pd.read_csv("/content/b3_384_test.csv")
b4_test=pd.read_csv("/content/b4_384_test.csv")
b5_test=pd.read_csv("/content/b5_384_test.csv")
b6_test=pd.read_csv("/content/b6_384_test.csv")

ab0=pd.read_csv("/content/b0_512_oof.csv")
ab1=pd.read_csv("/content/b1_512_oof.csv")
ab2=pd.read_csv("/content/b2_512_oof.csv")
ab3=pd.read_csv("/content/b3_512_oof.csv")
ab4=pd.read_csv("/content/b4_512_oof.csv")
ab5=pd.read_csv("/content/b5_512_oof.csv")
ab6=pd.read_csv("/content/b6_512_oof.csv")

ab0_test=pd.read_csv("/content/b0_512_test.csv")
ab1_test=pd.read_csv("/content/b1_512_test.csv")
ab2_test=pd.read_csv("/content/b2_512_test.csv")
ab3_test=pd.read_csv("/content/b3_512_test.csv")
ab4_test=pd.read_csv("/content/b4_512_test.csv")
ab5_test=pd.read_csv("/content/b5_512_test.csv")
ab6_test=pd.read_csv("/content/b6_512_test.csv")

base_predictions_train = pd.DataFrame( {'b0': b0.pred.ravel(),
     'b1': b1.pred.ravel(),
     'b2': b2.pred.ravel(),
    'b3': b3.pred.ravel(),
         'b4': b4.pred.ravel(),
     'b5': b5.pred.ravel(),
     'b6': b6.pred.ravel(),

     'ab0': ab0.pred.ravel(),
     'ab1': ab1.pred.ravel(),
     'ab2': ab2.pred.ravel(),
    'ab3': ab3.pred.ravel(),
         'ab4': ab4.pred.ravel(),
     'ab5': ab5.pred.ravel(),
     'ab6': ab6.pred.ravel(),
     
    })

base_predictions_test = pd.DataFrame( {'b0_test': b0_test.target.ravel(),
     'b1_test': b1_test.target.ravel(),
     'b2_test': b2_test.target.ravel(),
    'b3_test': b3_test.target.ravel(),
         'b4_test': b4_test.target.ravel(),
     'b5_test': b5_test.target.ravel(),
     'b6_test': b6_test.target.ravel(),
     'ab0_test': ab0_test.target.ravel(),
     'ab1_test': ab1_test.target.ravel(),
     'ab2_test': ab2_test.target.ravel(),
    'ab3_test': ab3_test.target.ravel(),
         'ab4_test': ab4_test.target.ravel(),
     'ab5_test': ab5_test.target.ravel(),
     'ab6_test': ab6_test.target.ravel(),
    })


sns.set(style="white")
plt.rcParams['figure.figsize'] = (10, 10) 
sns.heatmap(base_predictions_train.corr(), annot = True, linewidths=.5, cmap="YlGnBu")
plt.title('Corelation Between Features', fontsize = 10)
plt.show()

sns.set(style="white")
plt.rcParams['figure.figsize'] = (10, 10) 
sns.heatmap(base_predictions_test.corr(), annot = True, linewidths=.5, cmap="YlGnBu")
plt.title('Corelation Between Features', fontsize = 10)
plt.show()

base_predictions_train["folds"] = b0.folds
base_predictions_train["target"] = b0.target

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
MODELS=[]

for i in range(5):

  train_x = base_predictions_train[base_predictions_train.folds!=i].iloc[:,:14]
  train_x = np.array(train_x)
  train_y = base_predictions_train[base_predictions_train.folds!=i].target

  valid_x = base_predictions_train[base_predictions_train.folds==i].iloc[:,:14]
  valid_x = np.array(valid_x)
  valid_y = base_predictions_train[base_predictions_train.folds==i].target

  print("Fold : {}".format(i))


  model = model.fit( train_x, train_y,
                    eval_set = (valid_x, valid_y),
                    verbose = 10,
                    eval_metric='auc',
                    early_stopping_rounds=100)
  MODELS.append( model )

a = MODELS[0].predict_proba(np.array(base_predictions_test))
b = MODELS[1].predict_proba(np.array(base_predictions_test))
c = MODELS[2].predict_proba(np.array(base_predictions_test))
d = MODELS[3].predict_proba(np.array(base_predictions_test))
e = MODELS[4].predict_proba(np.array(base_predictions_test))

def logit(p):
    return np.log(p) - np.log(1 - p)
final_test=(logit(a)+logit(b)+logit(c)+logit(d)+logit(e))/5


a = MODELS[0].predict_proba(np.array(base_predictions_train.iloc[:,:14]))
b = MODELS[1].predict_proba(np.array(base_predictions_train.iloc[:,:14]))
c = MODELS[2].predict_proba(np.array(base_predictions_train.iloc[:,:14]))
d = MODELS[3].predict_proba(np.array(base_predictions_train.iloc[:,:14]))
e = MODELS[4].predict_proba(np.array(base_predictions_train.iloc[:,:14]))

final=(logit(a)+logit(b)+logit(c)+logit(d)+logit(e))/5

sample=pd.read_csv("./b1_512_oof.csv")

sample.target=final[:,1]
roc_auc_score(base_predictions_train.target.rank(), sample.target.rank())

sample=pd.read_csv("/content/b1_512_test.csv")
sample.target=final_test[:,1]

sample.target=sample.target.rank()

sample.to_csv("stacked.csv", index=False)