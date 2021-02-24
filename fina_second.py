# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 01:41:20 2019

@author: Radha Rani
"""

import pandas as pd
file=r'F:\finacus job hackathon\python'
train=pd.read_csv(file+'\s_train.csv')
test=pd.read_csv(file+'\s_test.csv')

forpre=pd.read_csv(file+'\\finatest_std.csv')

from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(train,test_size=0.2,random_state=2)
x_train=x_train.reset_index(drop=True)
y_train=(x_train['Target']).astype(int)
y_test=(x_test['Target']).astype(int)

x_train=x_train.drop(['Target'],axis=1)
x_test=x_test.drop(['Target'],axis=1)


train['Target'].value_counts()
test['Target'].value_counts()
y_train=(train['Target']).astype(int)
y_test=(test['Target']).astype(int)

x_train=train.drop(['Target'],axis=1)
x_test=test.drop(['Target'],axis=1)






from sklearn import tree
from sklearn.metrics import roc_auc_score

dtree=tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=10)
x_train=x_train.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)

dtree.fit(x_train,y_train)
dtree.predict(x_train)
train_score=dtree.predict_proba(x_train)[:,1]
roc_auc_score(y_train,train_score)

import pandas as pd
import numpy as np

prediction=np.where(train_score>0.009,1,0)


pd.crosstab(y_train,prediction)

train_score=dtree.predict_proba(x_train)[:,1]

testid=(forpre['id']).astype(int)
forpre=forpre.drop(['id'],axis=1)

train_core=dtree.predict_proba(forpre)[:,1]
pred=np.where(train_core>0.009,1,0)

df1=pd.DataFrame({'id':testid,'Target':pred})



train_score
df1.to_csv(file+'\\finaout1.csv')


from operator import itemgetter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomisedsearchCV
rf=RandomForestClassifier(n_estimators=100,
                          verbose=1,
                          criterion='entropy',
                          min_samples_split=2,
                          bootstrap=True,
                          max_depth=None,
                          max_features=20,
                          min_samples_leaf=1,
                          class_weight='balanced')

rf.fit(x_train,y_train)

predicted=rf.predict_proba(x_test)[:,1]
roc_auc_score(y_test,predicted)


prediction=np.where(predicted>0.00801,1,0)


pd.crosstab(y_test,prediction)

predicted=rf.predict_proba(forpre)[:,1]
testid=(forpre['id']).astype(int)
forpre=forpre.drop(['id'],axis=1)

train_core=dtree.predict_proba(forpre)[:,1]
pred=np.where(train_core>0.009,1,0)

df1=pd.DataFrame({'id':testid,'Target':pred})



train_score
df1.to_csv(file+'\\finaoutrand1.csv')








train_score>0.2

predicted_probs>0.51
#prediction=np.where(predicted_probs>0.49,1,0)
roc_auc_score(y_test,predicted_probs)
roc_auc_score(y_test,prediction)

import numpy as np

cutoffs=np.linspace(0.0001,0.0999,999)
train_score=rf.predict_proba(x_train)[:,1]
#train_score=dtree.predict_proba(x_train)[:,1]
#train_score=logr.predict_proba(article_train)[:,1]
real=y_train

KS_all=[]
for cutoff in cutoffs:
    predicted=(train_score>cutoff).astype(int)
    
    TP=((predicted==1)&(real==1)).sum()
    TN=((predicted==0)&(real==0)).sum()
    FP=((predicted==1)&(real==0)).sum()
    FN=((predicted==0)&(real==1)).sum()
    
    P=TP+FN
    N=TN+FP
    KS=(TP/P)-(FP/N)
    KS_all.append(KS)
    
type(train_score)
cutoffs[KS_all==max(KS_all)]    
logr.intercept_    
list(zip(x_train.columns,logr.coef_[0]))    




prediction=np.where(train_score>0.009,1,0)


pd.crosstab(y_train,prediction)


np.range(train_score)



table()

KS_all.value_counts


import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

cutoffs
logr.fit(x_train,y_train)
logr.predict(x_test)
logr.predict_proba(x_test)
predicted_probs=logr.predict_proba(x_test)[:,1]
roc_auc_score(y_test,predicted_probs)

cutoffs=np.linspace(0,1,100)
train_score=logr.predict_proba(x_train)[:,1]
real=y_train

train_score>0.2

KS_all=[]
for cutoff in cutoffs:
    predicted=(train_score>cutoff).astype(int)
    
    TP=((predicted==1) & (real==1)).sum()
    TN=((predicted==0) & (real==0)).sum()
    FP=((predicted==1) & (real==0)).sum()
    FN=((predicted==0) & (real==1)).sum()
    
    P=TP+FN
    N=TN+FP
    KS=(TP/P)-(FP/N)
    KS_all.append(KS)
    
type(train_score)
cutoffs[KS_all==max(KS_all)]    
logr.intercept_    
list(zip(x_train.columns,logr.coef_[0]))    



pred=(train_score>0.2).astype(int)
min(KS_all)

prob_score=pd.Series(list(zip(*logr.predict_proba(x_train)))[1])

file=r'F:\finacus job hackathon\python\testnaremoved.csv'
test=pd.read_csv(file)
test.columns
cutoffs=np.linspace(0,1,100)
test=test.drop(['id','Group'],axis=1)
test=test.reset_index(drop=True)


test.isnull().sum
test.columns



test.loc[test['per6'].isinfinite(),'Per6']=train['per6'].mean()
for col in [test.columns]:
    
    test.loc[test[col].isnull(),col]=train[col].mean()

test[test==np.inf]=np.nan
test.fillna(test.mean(), inplace=True)

q = test.replace([np.inf, -np.inf], np.nan)
for col in test.columns:
    max(test[col])

logr.predict(test)
test.score=logr.predict_proba(test)[:,1]


KS_cut=[]
for cutoff in cutoffs:
    predicted=pd.Series([0]*len(y_train))
    predicted[prob_score>cutoff]=1
    df=pd.DataFrame(list(zip(y_train,predicted)),columns=["real","predicted"])
    TP=len(df[(df["real"]==1) &(df["predicted"]==1) ])
    FP=len(df[(df["real"]==0) &(df["predicted"]==1) ])
    TN=len(df[(df["real"]==0) &(df["predicted"]==0) ])
    FN=len(df[(df["real"]==1) &(df["predicted"]==0) ])
    P=TP+FN
    N=TN+FP
    KS=(TP/P)-(FP/N)
    KS_cut.append(KS)

cutoff_data=pd.DataFrame(list(zip(cutoffs,KS_cut)),columns=["cutoff","KS"])

KS_cutoff=cutoff_data[cutoff_data["KS"]==cutoff_data["KS"].max()]["cutoff"]



