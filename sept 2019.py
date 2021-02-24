# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:05:42 2019

@author: admin
"""
import numpy as np
import pandas as pd
#import seaborn as sns
#sns.heatmap(train.corr())
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_test_share.csv')

train=train.drop(['Complaint ID'],axis=1)
test=test.drop(['Complaint ID'],axis=1)

train['Consumer disputed?']=(train['Consumer disputed?']=='Yes').astype(int)
train['Consumer disputed?'].head(50)

'ZIP code'
train=train.drop(['Tags','Consumer complaint narrative','Date received','Date sent to company','ZIP code'],axis=1)
train=train.drop(['ZIP code'],axis=1)

te=train.columns
test.columns

test['Consumer disputed?']=0

list(zip(train.columns,train.dtypes,train.nunique(),train.isnull().sum()))

#[('Date received', dtype('O'), 1759L, 0L),
# ('Product', dtype('O'), 12L, 0L),
# ('Sub-product', dtype('O'), 47L, 138473L),
# ('Issue', dtype('O'), 95L, 0L),
# ('Sub-issue', dtype('O'), 68L, 292625L),
# ('Consumer complaint narrative', dtype('O'), 74019L, 403327L),
# ('Company public response', dtype('O'), 10L, 388029L),
# ('Company', dtype('O'), 3276L, 0L),
# ('State', dtype('O'), 62L, 3839L),
# ('ZIP code', dtype('O'), 25962L, 3848L),
# ('Tags', dtype('O'), 3L, 411215L),
# ('Consumer consent provided?', dtype('O'), 4L, 342934L),
# ('Submitted via', dtype('O'), 6L, 0L),
# ('Date sent to company', dtype('O'), 1706L, 0L),
# ('Company response to consumer', dtype('O'), 7L, 0L),
# ('Timely response?', dtype('O'), 2L, 0L),
# ('Consumer disputed?', dtype('O'), 2L, 0L),
# ('Complaint ID', dtype('int64'), 478421L, 0L)]

'Sub-product','Sub-issue','Company public response','Company','ZIP code','Consumer consent provided?',
'Submitted via','Company response to consumer','Timely response?','State','Product'

'Consumer disputed?','Timely response?','Issue'

'Consumer complaint narrative'   # later

'Date received','Date sent to company'    # later

'Tags','Complaint ID'      # drop


zp=train['Issue'].value_counts() #take greater than 2000 freq
pd.crosstab(train['Issue'],train['Consumer disputed?'])
cat_cols=train[['Issue']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>9000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)



zp=train['State'].value_counts() #take greater than 2000 freq
pd.crosstab(train['State'],train['Consumer disputed?'])
cat_cols=train[['State']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>9000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)



zp=train['Product'].value_counts() #take greater than 2000 freq
pd.crosstab(train['Product'],train['Consumer disputed?'])
cat_cols=train[['Product']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>2000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)


#zp=train['ZIP code'].value_counts() #take greater than 500 freq
#pd.crosstab(train['ZIP code'],train['Consumer disputed?'])
#cat_cols=train[['ZIP code']]
#for col in cat_cols:
#    freqs=train[col].value_counts()
#    k=freqs.index[freqs>800][:-1]
#    for cat in k:
#        name=col+'_'+cat[:5]
#        train[name]=(train[col]==cat).astype(int)
#    del train[col]
#    print(col)


zp=train['Consumer consent provided?'].value_counts() #take 4-1 dummies
pd.crosstab(train['Consumer consent provided?'],train['Consumer disputed?'])    
cat_cols=train[['Consumer consent provided?']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>4000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)

zp=train['Submitted via'].value_counts() #take 6-1 dummies
pd.crosstab(train['Submitted via'],train['Consumer disputed?']) 
cat_cols=train[['Submitted via']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>7000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)


zp=train['Company response to consumer'].value_counts() #take 6-1 dummies
pd.crosstab(train['Company response to consumer'],train['Consumer disputed?'])
cat_cols=train[['Company response to consumer']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>4000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)
 

zp=train['Timely response?'].value_counts() #take 1 dummy
pd.crosstab(train['Timely response?'],train['Consumer disputed?'])
cat_cols=train[['Timely response?']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>4000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)


pd.crosstab(train['Sub-product'],train['Consumer disputed?'])



#train['Sub-product']=np.where(train['Sub-product']==NaN,'Other mortgage',train['Sub-product'])
train.loc[train['Sub-product'].isnull(),'Sub-product']='unknown_Sub-product'
zp=train['Sub-product'].value_counts() #take freq > 1000
#pd.crosstab(train['Sub-product'],train['Consumer disputed?'])
cat_cols=train[['Sub-product']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>10000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)
    
    
train['Consumer disputed?'].head(50)

pd.crosstab(train['Sub-issue'],train['Consumer disputed?'])
zp=train['Sub-issue'].value_counts()    #2600 freq cutoff
train.loc[train['Sub-issue'].isnull(),'Sub-issue']='unknown_Sub-issue'
cat_cols=train[['Sub-issue']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>5000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)




pd.crosstab(train['Company public response'],train['Consumer disputed?'])
zp=train['Company public response'].value_counts()    #20 freq cutoff
train.loc[train['Company public response'].isnull(),'Company public response']='unknown_Company public response'
cat_cols=train[['Company public response']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>1000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)






pd.crosstab(train['Company'],train['Consumer disputed?'])
zp=train['Company'].value_counts()    #1000 freq cutoff
#train.loc[train['Company'].isnull(),'Company']='unknown_Company'
cat_cols=train[['Company']]
for col in cat_cols:
    freqs=train[col].value_counts()
    k=freqs.index[freqs>10000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        train[name]=(train[col]==cat).astype(int)
    del train[col]
    print(col)


test=test.drop(['Complaint ID'],axis=1)

#for col in train.select_dtypes(['object']).columns:
#    print('summary for:'+col)
#    print(train[col].value_counts())
#    print(pd.crosstab(train[col],train['Consumer disputed?']))
#    print('--------------')


train=train.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)
list(zip(train.columns,train.dtypes,train.nunique(),train.isnull().sum()))


train.to_csv('D:\mohit gate\edvancer python\project 1\sept2019.csv')

import pandas as pd
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\sept2019.csv')


















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
mydata=pd.DataFrame({'ccn':train['Company'],
                     'cd':train['Consumer disputed?']})
list(zip(mydata.columns,mydata.dtypes,mydata.nunique()))
mydata.head()

mydata=pd.DataFrame({'ccn':train['Consumer complaint narrative'],
                     'cd':train['Consumer disputed?']})

#[print(cc[i]) for i in cc['ccn']]
mydata['ccn']=np.where(mydata['ccn'].isnull(),'',mydata['ccn'])
#tr['Consumer complaint narrative']=np.where(tr['Consumer complaint narrative'].isnull(),'',tr['Consumer complaint narrative'])

#mydata.loc['ccn'=='',:].value_counts()

mydata['ccn'].head(50)
mydata['cd'].value_counts() 


tf.fit(mydata['ccn'])
train_tf=tf.transform(mydata['ccn'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\ccn_Extraction_sept2019.csv')


tf.vocabulary_
{"'s": 0,
 '``': 1,
 'account': 2,
 'also': 3,
 'amount': 4,
 'asked': 5,
 'back': 6,
 'balance': 7,
 'bank': 8,
 u'call': 9,
 'called': 10,
 'card': 11,
 'collection': 12,
 'company': 13,
 'contacted': 14,
 'could': 15,
 'credit': 16,
 'date': 17,
 u'day': 18,
 'debt': 19,
 'due': 20,
 'even': 21,
 'fee': 22,
 'get': 23,
 'help': 24,
 'home': 25,
 'information': 26,
 u'know': 27,
 'letter': 28,
 u'loan': 29,
 'made': 30,
 'make': 31,
 'money': 32,
 'month': 33,
 'mortgage': 34,
 "n't": 35,
 'need': 36,
 'never': 37,
 'number': 38,
 'one': 39,
 'paid': 40,
 'pay': 41,
 'payment': 42,
 'phone': 43,
 'received': 44,
 'report': 45,
 'said': 46,
 'sent': 47,
 'service': 48,
 'since': 49,
 'still': 50,
 u'time': 51,
 'told': 52,
 'would': 53,
 'xx/xx/xxxx': 54,
 'xxxx': 55,
 'year': 56}

#ttt=CountVectorizer(analyzer=split_into_lemma,stop_words=my_stop)
#ttt.fit(mydata['ccn'])
print tf.vocabulary_
               
test_tf=tf.transform(mydata['ccn'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)
x_test_tfidf_scaled=st.transform(x_test_tf)






'Date received','Date sent to company'
import numpy as np
import pandas as pd
#import seaborn as sns
#sns.heatmap(train.corr())
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')



mydata=pd.DataFrame({'dr':train['Date received'],
                     'dstc':train['Date sent to company'],
                     'cd':train['Consumer disputed?']})



mydata['rec']=pd.to_datetime(mydata['dr'])
mydata['sent']=pd.to_datetime(mydata['dstc'])

mydata['rec'].head(5)
mydata['wkdr']=mydata['rec'].dt.weekday_name
mydata['wkds']=mydata['sent'].dt.weekofyear

train['www']=np.where(train['wkdr']>25,1,0)

pd.crosstab(train['wkds'],train['Consumer disputed?'])

train['Date received'].head(10)


mydata['weekd']=mydata.rec.dt.weekday_name

mydata['ydrec']=mydata.rec.dt.dayofyear
mydata['ydsent']=mydata.sent.dt.dayofyear
mydata['yddiff']=mydata['ydsent']-mydata['ydrec']
ydiff=pd.crosstab(mydata['yddiff'],mydata['cd'])
mydata['www']=np.where(mydata['wkdr']>25,1,0)

ddf1=np.arange(-364,-329)
ddf2=np.arange(0,50)
mydata['weekend']=np.where(mydata['yddiff'].isin([]),1,0)

mydata['weekd'].value_counts()
pd.crosstab(mydata['weekd'],mydata['cd'])



x = pd.datetime.now() 
x.month, x.year
# Create date and time with dataframe 
rng = pd.DataFrame() 
rng['date'] = pd.date_range('1/1/2011', periods = 72, freq ='H') 
  
# Print the dates in dd-mm-yy format 
rng[:5] 
  
# Create features for year, month, day, hour, and minute 
rng['year'] = rng['date'].dt.year 
rng['month'] = rng['date'].dt.month 
rng['day'] = rng['date'].dt.day 
rng['hour'] = rng['date'].dt.hour 
rng['minute'] = rng['date'].dt.minute 
  
# Print the dates divided into features 
rng.head(3)
# Input present datetime using Timestamp 
t = pandas.tslib.Timestamp.now() 
t
# Convert timestamp to datetime 
t.to_datetime()
# Directly access and print the features 
t.year 
t.month 
t.day 
t.hour 
t.minute 
t.second 
import pandas as pd 
  
url = 'http://bit.ly/uforeports'
  
# read csv file 
df = pd.read_csv(url)            
df.head() 
# Convert the Time column to datatime format 
df['Time'] = pd.to_datetime(df.Time) 
  
df.head() 
# shows the type of each column data 
df.dtypes 
# Get hour detail from time data 
df.Time.dt.hour.head() 
# Get name of each date 
df.Time.dt.weekday_name.head() 
# Get ordinal day of the year 
df.Time.dt.dayofyear.head()










from sklearn.model_selection import train_test_split
bd_train,bd_test=train_test_split(train,test_size=0.3,random_state=2)
bd_train.isnull().sum()

# fi has 82 null, we  fill null here
#for col in ['fi']:
#    bd_train.loc[bd_train[col].isnull(),col]=bd_train[col].mean()
#    bd_test.loc[bd_test[col].isnull(),col]=bd_train[col].mean()
    # never put mean of test data in test data
    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

x_train=bd_train.drop('Consumer disputed?',axis=1)
y_train=bd_train['Consumer disputed?']
x_test=bd_test.drop('Consumer disputed?',axis=1)
y_test=bd_test['Consumer disputed?']
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



for col in train.select_dtypes(['int']).columns:
    print('summary for:'+col)
    print(train[col].value_counts())
    print(pd.crosstab(train[col],train['Consumer disputed?']))
    print('--------------')


train['Unnamed: 0'].head(50)
