# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 19:37:41 2018

@author: admin
"""

import numpy as np
import pandas as pd
import seaborn as sns
sns.heatmap(union.corr())
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')
#join.to_csv('D:\mohit gate\edvancer python\project 1\Complaints_words.csv')
join=pd.read_csv('D:\mohit gate\edvancer python\project 1\Complaints_words.csv')

train.head(5)

tr=train[['Company','Consumer disputed?']]


comp=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')
join.columns
del train

join=pd.concat([train,test],axis=1)
join.columns
tr=tr.drop(['Consumer complaint narrative','Company'],axis=1)
join=join.drop(['Unnamed: 0','Consumer disputed?.1'],axis=1)


test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_test_share.csv')
ID=test['Complaint ID']
ID.head(5)
del cc
train.columns
test.columns
train=train.drop(['Complaint ID'],axis=1)
test=test.drop(['Complaint ID'],axis=1)

train['']

train['rec']=pd.to_datetime(train['Date received'])
train['sent']=pd.to_datetime(train['Date sent to company'])

train['wkdr']=train['rec'].dt.month
train['wkds']=train['sent'].dt.month

train['Date received'].head(10)

train['Consumer disputed?']=np.where(train['Consumer disputed?']=='Yes',1,0)
sns.jointplot(x='wkdr',y='Consumer disputed?',data=train,kind='kde',size=10)
#sns.lmplot(x='age',y='duration',data=bd.iloc[1:100,:],row='loan',col='default')
myplot=sns.countplot(x='wkdr',data=train)


myplot=sns.countplot(x='Consumer disputed?',data=train)


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
del join

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
join=join.drop(['Product', 'Sub-product', 'Issue',
      'State', 'ZIP code',
       'Submitted via', 'Company response to consumer',
       'Timely response?'],axis=1)

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
      'State', 'ZIP code',
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
join['Company'].value_counts()
#cat_cols=cc.select_dtypes(['object']).columns

tr['Company']=np.where(tr['Company']=='Wells Fargo & Company','Wells Fargo',tr['Company'])



cat_cols=join[['Company']]
for col in cat_cols:
    freqs=join[col].value_counts()
    k=freqs.index[freqs>3000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        join[name]=(join[col]==cat).astype(int)
    del join[col]
    print(col)



join['Company'].value_

del com
freqs    
del union


#y_train['cd'].value_counts()
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
lemma=WordNetLemmatizer()
my_stop=set(stopwords.words('english')+list(punctuation))
stopwords.words('english')
list(punctuation)

lemma.lemmatize('peoples')

def split_into_lemma(message):
    message=message.lower()
    words=word_tokenize(message)
    words_sans_stop=[]
    for word in words:
        if word in my_stop:continue
        words_sans_stop.append(word)
    return[lemma.lemmatize(word) for word in words_sans_stop]
    
words=word_tokenize('if i pass a sentence here, it will break this; into words. ')
for word in words:
    if word in my_stop:continue
    else:print(word)
    
from sklearn.feature_extraction.text import CountVectorizer
tf=CountVectorizer(analyzer=split_into_lemma,
                   min_df=10000,max_df=250000,
                   stop_words=my_stop)

cc.columns
mydata=pd.DataFrame({'ccn':cc['Company'],
                     'cd':cc['Consumer disputed?']})
list(zip(mydata.columns,mydata.dtypes,mydata.nunique()))
mydata.head()

mydata=pd.DataFrame({'ccn':cc['Consumer complaint narrative'],
                     'cd':cc['Consumer disputed?']})

#[print(cc[i]) for i in cc['ccn']]
mydata['ccn']=np.where(mydata['ccn'].isnull(),'',mydata['ccn'])
tr['Consumer complaint narrative']=np.where(tr['Consumer complaint narrative'].isnull(),'',tr['Consumer complaint narrative'])

#mydata.loc['ccn'=='',:].value_counts()

mydata['ccn'][0]
mydata['cd'].value_counts() 


tf.fit(tr['Consumer complaint narrative'])
train_tf=tf.transform(tr['Consumer complaint narrative'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')

               
test_tf=tf.transform(mydata['ccn'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)
x_test_tfidf_scaled=st.transform(x_test_tf)




join.to_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplitjoin.csv')


union.to_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplit.csv')

import numpy as np
import pandas as pd
import seaborn as sns
sns.heatmap(train.corr())

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplit.csv')

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\Percent52_beforesplitjoin.csv')
union=union.drop(['Unnamed: 0'],axis=1)





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

from sklearn.model_selection import train_test_split
article_train,article_test=train_test_split(union,test_size=0.2,random_state=2)
article_train=article_train.reset_index(drop=True)
y_train=(article_train['Consumer disputed?']==1).astype(int)
y_test=(article_test['Consumer disputed?']==1).astype(int)

article_train=article_train.drop(['Consumer disputed?'],axis=1)
article_test=article_test.drop(['Consumer disputed?'],axis=1)
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