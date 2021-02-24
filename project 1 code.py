# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:34:09 2018

@author: admin
"""

import os
path=r'D:\mohit gate\edvancer python\Project 1'
files=os.listdir(path+'\\Consumer_Complaints_train.csv')
# LOGISTIV REGRESSION ---------------------------------------------------------------

import numpy as np
import pandas as pd
file=r'D:\mohit gate\edvancer python\Project 1'
cc=pd.read_csv(file+'\\Consumer_Complaints_train.csv')



list(zip(cc.columns,cc.dtypes,cc.nunique()))

for obj in bd.select_dtypes(['object']).columns:
    print(bd[obj].value_counts())
    print('--------------------')

bd=bd.drop(['REF_NO','post_code','post_area'],axis=1)

bd['children']=np.where(bd['children']=='Zero',0,bd['children'])
bd['children']=np.where(bd['children']=="4+",4,bd['children'])
bd['children']=pd.to_numeric(bd['children'],errors='coerce')

bd['family_income'].value_counts()
k=bd['family_income'].str.split('>=',expand=True)
k,k[0],k[1]
#              0       1
#0                35,000
#1     <12,500,   10,000
#2                35,000
#3                35,000
#4     <27,500,   25,000
#5     <25,000,   22,500
k[0]=k[0].str.replace('<',"")
#k[1]=k[1].str.replace('>=',"")
k[0]=k[0].str.replace(',',"")
k[1]=k[1].str.replace(',',"")
k[0]=pd.to_numeric(k[0],errors='coerce')
k[1]=pd.to_numeric(k[1],errors='coerce')
k.loc[k[0].isnull(),0]=k[1]
k.loc[k[1].isnull(),1]=k[0]
bd['fi']=0.5*(k[0]+k[1])
bd['fi'].isnull().sum() # they are unknown
del bd['family_income']

bd.dtypes
bd['Revenue.Grid']=(bd['Revenue.Grid']==1).astype(int) # 0,1 bydefault binary, change to int

cat_vars=bd.select_dtypes(['object']).columns
cat_vars
for col in cat_vars:
    dummy=pd.get_dummies(bd[col],drop_first=True,prefix=col)
    bd=pd.concat([bd,dummy],axis=1)
    del bd[col]
    print(col)
    del dummy

bd.shape    

from sklearn.model_selection import train_test_split
bd_train,bd_test=train_test_split(bd,test_size=0.2,random_state=2)
bd_train.isnull().sum()

# fi has 82 null, we  fill null here
for col in ['fi']:
    bd_train.loc[bd_train[col].isnull(),col]=bd_train[col].mean()
    bd_test.loc[bd_test[col].isnull(),col]=bd_train[col].mean()
    # never put mean of test data in test data
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

x_train=bd_train.drop('Revenue.Grid',axis=1)
y_train=bd_train['Revenue.Grid']
x_test=bd_test.drop('Revenue.Grid',axis=1)
y_test=bd_test['Revenue.Grid']
logr.fit(x_train,y_train)
logr.predict(x_test)
logr.predict_proba(x_test)
predicted_probs=logr.predict_proba(x_test)[:,1]
roc_auc_score(y_test,predicted_probs)

cutoffs=np.linspace(0.01,0.99,99)
train_score=logr.predict_proba(x_train)[:,1]
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

