# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:46:44 2019

@author: admin
"""

train.to_csv('D:\mohit gate\edvancer python\project 2 3\unioncartrain.csv')
test.to_csv('D:\mohit gate\edvancer python\project 2 3\unioncartest.csv')

import numpy as np
import pandas as pd


train=pd.read_csv('D:\mohit gate\edvancer python\project 2 3\unioncartrain.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 2 3\unioncartest.csv')


from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso   #Ridge
from sklearn.model_selection import KFold

ld_train=cr.reset_index(drop=True) # kfold use index to split
x_train=cr.drop('V86',axis=1)
y_train=cr['V86']

lambdas =np.linspace(0.01,0.1,100)
mae_list=[]
for a in lambdas:
    lasso=Lasso(fit_intercept=True,alpha=a)  # a is lambda
    kf=KFold(n_splits=10)
    xval_err=0
    for train, test in kf.split(x_train):   # train test are array of index
        lasso.fit(x_train.loc[train],y_train[train]) # 
        p=lasso.predict(x_train.loc[test])
        xval_err+=mean_absolute_error(y_train[test],p)
    mae_10cv=xval_err/10
    print(a,':',mae_10cv)
    mae_list.extend([mae_10cv])
    
best_alpha=lambdas[mae_list==min(mae_list)]
print('alpha with min mae_10cv is:',best_alpha)
    
lasso=Lasso(fit_intercept=True,alpha=best_alpha)
lasso.fit(x_train,y_train)
p_test=lasso.predict(x_test)
list(zip(x_train.columns,lasso.coef_)) 

mae_lasso=mean_absolute_error(ld_test['Interest.Rate'],p_test)
mae_lasso    # train_test_split gives 80-20% data from ld train

#print(x_train.loc[:100])





from sklearn.model_selection import train_test_split
article_train,article_test=train_test_split(train,test_size=0.2,random_state=2)
article_train=article_train.reset_index(drop=True)
y_train=(article_train['V86']==1).astype(int)
y_test=(article_test['V86']==1).astype(int)

article_train=article_train.drop(['V86'],axis=1)
article_test=article_test.drop(['V86'],axis=1)

test=test.drop(['V86'],axis=1)
test=test.drop(['Consumer disputed?'],axis=1)
#tf.fit(article_train['ccn'])
#train_tf=tf.transform(article_train['ccn'])
#x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
#               
#test_tf=tf.transform(article_test['ccn'])
#x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
#x_train_tf.shape
#x_test_tf.shape


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

#y_train.dtypes
#y_train['cd'].value_counts
#y_train['cd']=(y_train['cd']==1).astype(int) 
logr.fit(article_train,y_train)
logr.predict(article_test)


logr.predict(test)
pre=logr.predict_proba(test)[:,1]
res=pd.DataFrame({'Complaint ID':ID,
                  'Consumer disputed?':prediction})

img=pd.read_csv('D:\mohit gate\edvancer python\Project 1\Result_pr')

res=pd.DataFrame({'V86':pre})
res['V86']=np.where(res['V86']>0.22,1,0)


res.to_csv('D:\mohit gate\edvancer python\project 1\Result_carvan.csv')
predicted_probs=logr.predict_proba(article_test)[:,1]

predicted_probs=logr.predict_proba(article_test)[:,1]
predicted_probs>0.51
prediction=np.where(predicted_probs>0.50,1,0)
roc_auc_score(y_test,predicted_probs)
roc_auc_score(y_test,prediction)

cutoffs=np.linspace(0,1,100)
train_score=logr.predict_proba(article_test)[:,1]
real=y_test

KS_all=[]
#F2_all=[]
for cutoff in cutoffs:
    predicted=(train_score>cutoff).astype(int)
    
    TP=((predicted==1)&(real==1)).sum()
    TN=((predicted==0)&(real==0)).sum()
    FP=((predicted==1)&(real==0)).sum()
    FN=((predicted==0)&(real==1)).sum()
    
    P=TP+FN
    N=TN+FP
    R=TP/P
    KS=(TP/P)-(FP/N)
    KS_all.append(KS)
    #F2=(5*P*R)/((4*P)+R)
    #F2_all.append(F2)
    print(KS)
    
    
type(train_score)
#cutoffs[KS_all==max(KS_all)]    
cutoffs[F2_all==max(F2_all)]

logr.intercept_    
list(zip(x_train.columns,logr.coef_[0]))   

# beta=2
# array([0.        , 0.01010101, 0.02020202])
train
dt=pd.DataFrame({'train_score':train_score})
import seaborn as sns
sns.boxplot(x='train_score',data=dt)









