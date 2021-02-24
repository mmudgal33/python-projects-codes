# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 18:24:57 2018

@author: admin
"""
import pandas as pd
import numpy as np

data=pd.read_csv('D:\mohit gate\edvancer python\project 1\Complaints_Data.csv')



x_train=
train=data[data['ttt']==1]
test=data[data['ttt']==0]

train.columns
test.columns

train.to_csv('D:\mohit gate\edvancer python\project 1\Ttrain.csv')
test.to_csv('D:\mohit gate\edvancer python\project 1\Ttest.csv')

x_train.to_csv('D:\mohit gate\edvancer python\project 1\Tx_train.csv')
y_train.to_csv('D:\mohit gate\edvancer python\project 1\Ty_train.csv')

train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Ttrain.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Ttest.csv')

x_train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Tx_train.csv')
y_train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Ty_train.csv')

x_train.columns
y_train.columns
y_train.head(5)
train=train.drop(['Unnamed: 0'],axis=1)
test=test.drop(['Unnamed: 0'],axis=1)

x_train=x_train.drop(['0'],axis=1)

mydata.shape

pd.crosstab(train['wkds'],train['Consumer disputed?'])
y_train['0.1'].value_counts

#train=train.drop(['Unnamed: 0', 'Unnamed: 0.1','ttt'],axis=1)

x_train=train.drop(['Consumer disputed?'],axis=1)
y_train=train['Consumer disputed?']
mydata=pd.DataFrame({'Consumer disputed?':y_train['0.1']})
mydata=mydata['Consumer disputed?'].append([0])

logr.fit(x_train,y_train)





from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced',penalty='l1')
#x_train=x_train.reset_index(drop=True)

from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(train,test_size=0.2,random_state=2)
x_train=x_train.reset_index(drop=True)
y_train=(x_train['Consumer disputed?']==1).astype(int)
y_test=(x_test['Consumer disputed?']==1).astype(int)

x_train=x_train.drop(['Consumer disputed?'],axis=1)
x_test=x_test.drop(['Consumer disputed?'],axis=1)


#logr.fit(x_train,y_train)

from sklearn.linear_model import Lasso   #Ridge
from sklearn.model_selection import KFold

#x_train=x_train.reset_index(drop=True) # kfold use index to split
#x_train=c_train
y_train=ld_train['Interest.Rate']

lambdas =np.linspace(0.001,1,100)
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






test=test.drop(['ttt'],axis=1)
test=test.drop(['Consumer disputed?'],axis=1)
#tf.fit(article_train['ccn'])
#train_tf=tf.transform(article_train['ccn'])
#x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
#               
#test_tf=tf.transform(article_test['ccn'])
#x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
#x_train_tf.shape
#x_test_tf.shape




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

res['Result']=np.where(res['Result']>0.45,1,0)

res.to_csv('D:\mohit gate\edvancer python\project 1\Result_pr.csv')
predicted_probs=logr.predict_proba(article_test)[:,1]

predicted_probs=logr.predict_proba(article_test)[:,1]
predicted_probs>0.51
prediction=np.where(predicted_probs>0.50,1,0)
roc_auc_score(y_test,predicted_probs)
roc_auc_score(y_test,prediction)

cutoffs=np.linspace(0.01,0.99,99)
train_score=logr.predict_proba(article_test)[:,1]
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced',penalty='l1')
lm.coef_
x_test=ld_test.drop('Interest.Rate',axis=1)
predicted_ir=lm.predict(x_test)




import pandas as pd
import numpy as np
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Ttrain.csv')
train=train.drop(['Unnamed: 0'],axis=1)
train=train.iloc[250000:,:]

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced',penalty='l1')
#x_train=x_train.reset_index(drop=True)

from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(train,test_size=0.2,random_state=2)
x_train=x_train.reset_index(drop=True)
y_train=(x_train['Consumer disputed?']==1).astype(int)
y_test=(x_test['Consumer disputed?']==1).astype(int)

x_train=x_train.drop(['Consumer disputed?'],axis=1)
x_test=x_test.drop(['Consumer disputed?'],axis=1)




from sklearn.linear_model import Ridge  # Lass
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error


#train=train.reset_index(drop=True) # kfold use index to split
#x_train=train.drop('Consumer disputed?',axis=1)
#y_train=train['onsumer disputed?']


lambdas =np.linspace(0.1,1,100)
mae_list=[]
for a in lambdas:
    ridge=Ridge(fit_intercept=True,alpha=a)  # a is lambda
    kf=KFold(n_splits=10)
    xval_err=0
    for train, test in kf.split(x_train):   # train test are array of index
        ridge.fit(x_train.loc[train],y_train[train]) # 
        p=ridge.predict(x_train.loc[test])
        xval_err+=mean_absolute_error(y_train[test],p)
    mae_10cv=xval_err/10
    print(a,':',mae_10cv)
    mae_list.extend([mae_10cv]) 
    
#0,100,100 0.3299784083019851
#0,10,100 0.329896834793494
#0.3,1,100 0.329897255896238
    #0.3292227075437131
    
best_alpha=lambdas[mae_list==min(mae_list)]
print('Alpha with min mae_10cv error is:',best_alpha)

ridge=Ridge(fit_intercept=True,alpha=best_alpha)
ridge.fit(x_train,y_train)
p_test=ridge.predict(x_test)

d=ridge.Decision_function()

list(zip(x_train.columns,ridge.coef_))

mae_ridge=mean_absolute_error(y_test,p_test)
mae_ridge

roc_auc_score(y_test,p_test)
p_test=np.where(p_test>0.50,1,0)

import seaborn as sns
# DENSITY DISTRIBUTION FUNCTION
myplot=sns.distplot(p_test) #kernel density & histogram both
sns.distplot(bd['age'],kde=False)


test.columns
test=test.drop(['Unnamed: 0.2'],axis=1)
test=test.drop(['Unnamed: 0', 'Unnamed: 0.1', 'Unnamed: 0.1.1', 'Consumer disputed?','ttt'],axis=1)



cutoffs=np.linspace(0.01,0.99,99)
roc=[]
cutoff=[]
for x in cutoffs:
    pt_test=np.where(p_test>x,1,0)
    roc.append(roc_auc_score(y_test,p_test))
    cutoff.append(x)
    

max(roc)
roc

test.shape
x_train.shape
# Extract features from all text articles in data

pt_test=pd.DataFrame({'p_test':p_test})
pt_test.dtypes
pt_test.head(5)
roc_auc_score(y_test,pt_test)

#chk=pd.concat([ID,p_test],axis=1)
mydata=pd.DataFrame({'Complaint ID':ID,
                     'Consumer disputed?':pt_test['p_test']})
p_test=ridge.predict(test)
p=ridge.predict_proba(test)
 del p
mydata.to_csv('D:\mohit gate\edvancer python\project 1\Lassoproject1prediction.csv')

    
cutoffs=np.linspace(0.35,0.50)
train_score=ridge.predict(x_test)[:,1]
real=y_test

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

