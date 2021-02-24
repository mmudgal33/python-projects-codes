# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 15:12:50 2019

@author: admin
"""

import numpy as np
import pandas as pd
import seaborn as sns

train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_test_share.csv')

train['Consumer disputed?']=np.where(train['Consumer disputed?']=='Yes',1,0)
train=train.drop(['Complaint ID'],axis=1)
test=test.drop(['Complaint ID'],axis=1)

test['Consumer disputed?']=0
train['ttt']='1'
test['ttt']='0'
union=pd.concat([train,test],axis=0)

list(zip(train.columns,train.dtypes,train.nunique(),train.isnull().sum()))
list(zip(union.columns,union.dtypes,union.nunique(),union.isnull().sum()))

union.columns
join=join.drop(['Sub-product','Sub-issue',],axis=1)

pd.crosstab([union,union['Sub-issue'])
pd.crosstab(y_test,prediction)

union['Sub-issue'].value_counts()

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
                   min_df=10000,max_df=200000,
                   stop_words=my_stop)

mydata=pd.DataFrame({'ccn':train['Company'],
                     'cd':train['Consumer disputed?']})
list(zip(mydata.columns,mydata.dtypes,mydata.nunique()))
mydata.head()

tr=pd.DataFrame({'ccn':train['Consumer complaint narrative'],
                     'cd':test['Consumer disputed?']})

#[print(cc[i]) for i in cc['ccn']]
mydata['ccn']=np.where(mydata['ccn'].isnull(),'',mydata['ccn'])
tr['Consumer complaint narrative']=np.where(tr['Consumer complaint narrative'].isnull(),'',tr['Consumer complaint narrative'])
tr['ccn']=np.where(tr['ccn'].isnull(),'',tr['ccn'])

#mydata.loc['ccn'=='',:].value_counts()

mydata['ccn'][0]
sum(mydata['ccn'].isnull())
mydata['ccn'].value_counts() 

tf.fit(mydata['ccn'])
train_tf=tf.transform(train['Company'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')

tf.fit(tr['ccn'])
train_tf=tf.transform(tr['ccn'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction2019.csv')

union=pd.concat([x_train_tf,train[]],axis=0)





dat['diff'].describe()






dat=pd.DataFrame({'Date received': union['Date received'],
                 'Date sent to company': union['Date sent to company'],
                 'Consumer disputed?': union['Consumer disputed?']})
    
dat=pd.DataFrame({'Date received': train['Date received'],
                 'Date sent to company': train['Date sent to company'],
                 'Consumer disputed?': train['Consumer disputed?']})    
 
dat['rec']=pd.to_datetime(dat['Date received'])
dat['sent']=pd.to_datetime(dat['Date sent to company'])

dat['wkdr']=dat['rec'].dt.dayofyear
dat['wkdr']=dat['rec'].dt.date
#dat['wkds']=dat['sent'].dt.month
dat['wkds']=dat['sent'].dt.dayofyear

jon['wkwr']=pd.to_numeric(jon['wkwr'])
jon['wkwr']=pd.to_numeric(jon['wkwr'])
jon['wkdr']=jon['wkdr'].astype(int)
jon['wkds']=jon['wkds'].astype(int)    
dat['wkds'].dtypes

dat['diff']=dat['wkds']-dat['wkdr']
pd.crosstab(dat['diff'],dat['Consumer disputed?'])

jon=pd.concat([train,dat],axis=1)
jon.columns    
import seaborn as sns
sns.heatmap(train.corr())
train.dtypes
jon[['wkdr','wkds']].head
    
union.columns
dat['rec']=pd.to_datetime(dat['Date received'])
dat['sent']=pd.to_datetime(dat['Date sent to company'])
dat['wkdr']=dat['rec'].dt.weekday_name
dat['wkds']=dat['sent'].dt.weekday_name

dat['wkdr']=np.where(dat['wkdr'].isin(['Saturday','Sunday']),1,0)
dat['wkds']=np.where(dat['wkds'].isin(['Saturday','Sunday']),1,0)


dat['s']=dat['sent'].dt.date
dat['r']=dat['rec'].dt.date
dat['d']=(dat['s']-dat['r']).astype(str)
dat['d'].dtypes
dat['d'].value_counts()

#mydata['rating_score']=np.where(mydata['rating'].isin(['Good','Excellent']),1,0)
dat['d'].head

k=dat['d'].str.split(' days ',expand=True)
k
dat['d']=k[0]
dat['d']=pd.to_numeric(dat['d'])
dat['d']=np.where(dat['d']<10,1,0)
dat['d'].value_counts()
dat['d'].dtypes


# hh.to_csv('D:\mohit gate\edvancer python\project 1\dat.csv')


#dat['diff']=(dat['wkdts']-dat['wkdtr'])
#dat['diff'].describe()
#lal=['0 days 00:00:00','1 days 00:00:00','2 days 00:00:00','3 days 00:00:00','4 days 00:00:00','5 days 00:00:00','6 days 00:00:00','7 days 00:00:00','-1 days 00:00:00','8 days 00:00:00','9 days 00:00:00','10 days 00:00:00']



dat['wkwr'].dtypes
dat['wkdtr']=dat['rec'].dt.day
dat['wkwr']=np.where(dat['wkdtr']>20,1,0)

dat['wkdts']=dat['sent'].dt.day
dat['wkws']=np.where(dat['wkdts']>20,1,0)
dat['wkws'].head

dat[['rec','sent']]
dat.columns
datli=['wkdr', 'wkds','d','wkwr','wkws']
dated=dat[datli]

#dated.to_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
#
#dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
#
#dat['wkdr'].value_counts()
#dte=['Consumer disputed?', 'Date received', 'Date sent to company', 'rec',
#       'sent', 'wkdr', 'wkds', 'wkwr', 'wkws']
#dat=dat.drop(['Consumer disputed?', 'Date received', 'Date sent to company', 'rec',
#       'sent'],axis=1)
#dat.to_csv('D:\mohit gate\edvancer python\project 1\date_Extraction.csv')
#union=pd.concat([union,dat],axis=1)
#
#train.columns
#join.columns
#pd.crosstab(join['credit'],join['Consumer disputed?'])
#
#import seaborn as sns
#myplot=sns.countplot(x='diff',hue='Consumer disputed?',data=dat)
## dat['wkdr1']=dat['rec'].dt.weekday
## dat['wkds2']=dat['sent'].dt.weekday
#dat.head
# dat=dat.drop(['wkdr1','wkds2'],axis=1)
# dat['wkdr']=np.where(dat['wkdr'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday']),1,0)

# mydata['rating_score']=np.where(mydata['rating'].isin(['Good','Excellent']),1,0)

# 'Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday'


