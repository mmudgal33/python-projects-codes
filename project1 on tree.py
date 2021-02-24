# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 11:09:54 2018

@author: admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
sns.heatmap(train.corr())
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')

train.columns

test.len
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_test_share.csv')
ID=test['Complaint ID']
ID.head(5)
del test
train.columns
test.columns
train=train.drop(['Complaint ID'],axis=1)
test=test.drop(['Complaint ID'],axis=1)

train['rec']=pd.to_datetime(train['Date received'])
train['sent']=pd.to_datetime(train['Date sent to company'])

train['wkdr']=train['rec'].dt.weekofyear
train['wkds']=train['sent'].dt.weekofyear

train['Date received'].head(10)

union['Consumer disputed?']=np.where(union['Consumer disputed?']=='Yes',1,0)

test['Consumer disputed?']=0
train['ttt']='1'
test['ttt']='0'

union=pd.concat([train,test],axis=0)

#union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
#trn=union[union['ttt']==1]
#trn=trn.drop(['ttt'],axis=1)

company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company.csv')
#dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
complaints=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_disputed.csv')
union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
gp=union[['ttt','Consumer disputed?']]
del union
join=pd.concat([complaints,company,union],axis=1)
join.columns
join=join.drop(['Unnamed: 0'],axis=1)

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\Aunion_tt.csv')
#union['Unnamed: 0'].head
#sns.heatmap(union.corr())
union.columns
del union
union['Consumer disputed?'].value_counts()
#union.iloc[union['ttt']=='0','Consumer disputed?']='NA'


#join=pd.concat([train,test],axis=1)

#del train, test
#union.dtypes
list(zip(train.columns,train.dtypes,train.nunique(),train.isnull().sum()))

#company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')
#cc.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction1.csv')
#final=pd.concat([dat,company,com],axis=1)
df=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')

cc.columns
cc.head
#union=union.drop(['Unnamed: 0','Date received','Consumer complaint narrative','ZIP code','Date sent to company','Company'],axis=1)
union=union.drop(['Consumer consent provided?','Sub-issue','Company public response','Tags',],axis=1)
train['State'].value_counts()


extra=union[['Date received',
       'Consumer complaint narrative',
       'Date sent to company']]

union=union.drop(['Date received',
       'Consumer complaint narrative',
       'Date sent to company'],axis=1)
      

train.columns
cc=cc.drop(['Date received',
       'Consumer complaintc narrative', 'Company',
       'State', 'ZIP code',
       'Submitted via', 'Date sent to company',
       'Complaint ID'],axis=1)

#################################################################################

com=pd.read_csv(file+'\\Consumer_Complaints_train.csv')

li=['Product', 'Sub-product', 'Issue',
       'Company', 'State', 'ZIP code',
       'Submitted via', 'Company response to consumer',
       'Timely response?']
list(zip(com.columns,com.dtypes,com.nunique()))

#cc.select_dtypes(['object'])
com['Issue'].head
com['Issue'].value_counts()
#
##bd.select_dtypes(['object']).columns
#for col in cc[list]:
#    print('summary for:'+col)
#    print(cc[col].value_counts())
#    print('--------------')
union['Company'].value_counts()
#cat_cols=cc.select_dtypes(['object']).columns
cat_cols=union[li]
for col in cat_cols:
    freqs=union[col].value_counts()
    k=freqs.index[freqs>20000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        union[name]=(union[col]==cat).astype(int)
    del union[col]
    print(col)

del com
freqs    
del union
join.to_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplitjoin.csv')


union.to_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplit.csv')

train.to_csv('D:\mohit gate\edvancer python\project 1\Train.csv')
test.to_csv('D:\mohit gate\edvancer python\project 1\Test.csv')
pt=pd.read_csv('D:\mohit gate\edvancer python\project 1\Pt.csv')


import numpy as np
import pandas as pd
import seaborn as sns
sns.heatmap(train.corr())

#union=pd.read_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplit.csv')

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplitjoin.csv')
del union
#union=union.drop(['Unnamed: 0'],axis=1)

train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Train.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Test.csv')

list(zip(train.columns,train.dtypes,train.nunique(),train.isnull().sum()))
train["also"].value_counts()
train['also']=train.drop(["'s"],axis=1)

from sklearn.model_selection import train_test_split
x_train,x_test=train_test_split(train,test_size=0.2,random_state=2)
x_train=x_train.reset_index(drop=True)
y_train=(x_train['Consumer disputed?']==1).astype(int)
y_test=(x_test['Consumer disputed?']==1).astype(int)

x_train=x_train.drop(['Consumer disputed?'],axis=1)
x_test=x_test.drop(['Consumer disputed?'],axis=1)


from sklearn import tree
from sklearn.metrics import roc_auc_score

dtree=tree.DecisionTreeClassifier(criterion='entropy',max_leaf_nodes=10)
x_train=x_train.reset_index(drop=True)
x_test=x_test.reset_index(drop=True)

dtree.fit(x_train,y_train)
dtree.predict(x_train)
p=dtree.predict_proba(x_train)[:,1]
roc_auc_score(x_train,p)

#pt['p'].dtype
#pt['p']=np.where(pt['p']>0.45,1,0)
#pt['p'].astype(int)

union.columns
union.head(5)
union=union.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
train=union[union['ttt']==1]
test=union[union['ttt']==0]
train=train.drop(['ttt'],axis=1)
test=test.drop(['ttt'],axis=1)
union['Consumer disputed?'].value_counts
test=test.drop(['Consumer disputed?'],axis=1)
test=test.drop(['Consumer disputed?','Unnamed: 0','ttt'],axis=1)
test=test.drop(['Consumer disputed?', 'ttt'],axis=1)
test.columns

union['ttt'].value_counts()


#article_test=article_test.drop(['Consumer disputed?'],axis=1)

article_train.columns

#article_train=article_train.drop(['ttt','Unnamed: 0'],axis=1)
#test=test.drop(['Consumer disputed?','ttt','Unnamed: 0'],axis=1)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

#y_train.dtypes
#y_train['cd'].value_counts
#y_train['cd']=(y_train['cd']==1).astype(int) 
#y_train.value_counts
logr.fit(article_train,y_train)
logr.predict(article_test)



logr.predict(test)
pre=logr.predict_proba(test)[:,1]
pre=pd.DataFrame({'Complaint ID':ID,
                  'Consumer disputed?':pre})
ded=pd.read_csv('D:\mohit gate\edvancer python\Project 1\Result_pr_after52.csv')
pre['Consumer disputed?']=np.where(pre['Consumer disputed?']>=0.50,1,0)


pre.to_csv('D:\mohit gate\edvancer python\Project 1\Result_pr_after52com.csv')
ID
pre.head(5)
res['Result']=np.where(res['Result']>0.45,1,0)

#res.to_csv('D:\mohit gate\edvancer python\project 1\Result_pr.csv')
predicted_probs=logr.predict_proba(article_test)[:,1]


predicted_probs>0.51
prediction=np.where(predicted_probs>0.49,1,0)
roc_auc_score(y_test,predicted_probs)
roc_auc_score(y_test,prediction)

cutoffs=np.linspace(0.01,0.99,99)
train_score=logr.predict_proba(article_train)[:,1]
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

# Extract features from all text articles in data
tf.fit(cc['Company'])
train_tf=tf.transform(cc['Company'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)