# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 21:34:09 2018

@author: admin
"""

#import os
#path=r'D:\mohit gate\edvancer python\Project 1'
#files=os.listdir(path+'\\Consumer_Complaints_train.csv')
## LOGISTIV REGRESSION ---------------------------------------------------------------

#################################################################################

import numpy as np
import pandas as pd
file=r'D:\mohit gate\edvancer python\Project 1'
cc=pd.read_csv(file+'\\Consumer_Complaints_train.csv')

list(zip(cc.columns,cc.dtypes,cc.nunique()))

#company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')
#cc.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction1.csv')
#final=pd.concat([dat,company,com],axis=1)
df=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')

cc.columns
cc.head
df=df.drop(['Date received','Consumer complaint narrative','ZIP code','Date sent to company','Complaint ID'],axis=1)
df=df.drop(['Company'],axis=1)

cc.columns
cc=cc.drop(['Date received',
       'Consumer complaintc narrative', 'Company',
       'State', 'ZIP code',
       'Submitted via', 'Date sent to company',
       'Complaint ID'],axis=1)

#################################################################################

com=pd.read_csv(file+'\\Consumer_Complaints_train.csv')

li=['Product', 'Sub-product', 'Issue', 'Sub-issue',
        'Company public response',
       'State', 'Tags', 'Consumer consent provided?',
       'Submitted via','Company response to consumer',
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

#cat_cols=cc.select_dtypes(['object']).columns
cat_cols=df[li]
for col in cat_cols:
    freqs=df[col].value_counts()
    k=freqs.index[freqs>10000][:-1]
    for cat in k:
        name=col+'_'+cat[:5]
        df[name]=(df[col]==cat).astype(int)
    del df[col]
    print(col)

del com
freqs    

df.to_csv('D:\mohit gate\edvancer python\project 1\other_Information.csv')


###################################################################################

dat=pd.DataFrame({'Date received': cc['Date received'],
                 'Date sent to company': cc['Date sent to company'],
                 'Consumer disputed?': cc['Consumer disputed?']})

dat['rec']=pd.to_datetime(dat['Date received'])
dat['sent']=pd.to_datetime(dat['Date sent to company'])
dat['wkdr']=dat['rec'].dt.weekday_name
dat['wkds']=dat['sent'].dt.weekday_name

dat['wkwr']=dat['rec'].dt.day
dat['wkwr']=pd.to_numeric(dat['wkwr'])
dat['wkws'].dtypes
dat['wkwr']=np.where(dat['wkwr']>15,1,0)
dat['wkws']=dat['sent'].dt.day
dat['wkws']=pd.to_numeric(dat['wkws'])
dat['wkws']=np.where(dat['wkws']>15,1,0)

dat.columns

dat['wkwr'].value_counts()
dte=['Consumer disputed?', 'Date received', 'Date sent to company', 'rec',
       'sent', 'wkdr', 'wkds', 'wkwr', 'wkws']
dat=dat.drop(['Consumer disputed?', 'Date received', 'Date sent to company', 'rec',
       'sent'],axis=1)
dat.to_csv('D:\mohit gate\edvancer python\project 1\date_Extraction.csv')


# dat['wkdr1']=dat['rec'].dt.weekday
# dat['wkds2']=dat['sent'].dt.weekday
dat.head
# dat=dat.drop(['wkdr1','wkds2'],axis=1)
# dat['wkdr']=np.where(dat['wkdr'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday']),1,0)
dat['wkdr']=np.where(dat['wkdr'].isin(['Saturday','Sunday']),1,0)
dat['wkds']=np.where(dat['wkds'].isin(['Saturday','Sunday']),1,0)
# mydata['rating_score']=np.where(mydata['rating'].isin(['Good','Excellent']),1,0)

# 'Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday'


####################################################################################



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
                   min_df=50000,max_df=200000,
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
#mydata.loc['ccn'=='',:].value_counts()

mydata['ccn'][0]
mydata['cd'].value_counts() 


tf.fit(mydata['ccn'])
train_tf=tf.transform(mydata['ccn'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
company.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')

               
test_tf=tf.transform(mydata['ccn'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)
x_test_tfidf_scaled=st.transform(x_test_tf)

##################################################################################

tf.fit(cc['Company'])
train_tf=tf.transform(cc['Company'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
final.to_csv('D:\mohit gate\edvancer python\project 1\Final_Extraction.csv')
final['cd'].dtypes
dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\date_Extraction.csv')
dat.columns
company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')
company.columns
complaints=pd.read_csv('D:\mohit gate\edvancer python\project 1\Complaints_Extraction.csv')
df=pd.read_csv('D:\mohit gate\edvancer python\project 1\other_Information.csv')


import pandas as pd
import numpy as np
final=pd.read_csv('D:\mohit gate\edvancer python\project 1\Final_Extraction.csv')
final['cd'].head
list(zip(final.columns,final.dtypes,final.nunique()))
#final['cd']=(final['cd']=='Yes').astype(int)
#y_train=(article_train['cd']=='Yes').astype(int)
#y_test=(article_test['cd']=='Yes').astype(int)

final=final.drop(['cd'],axis=1)
final['Consumer disputed?']=final['cd']


#final=pd.concat([dat,company,complaints,cd],axis=1)
#final=pd.concat([df,final],axis=1)
#cd=pd.DataFrame({'cd':cc['Consumer disputed?']})
#final=pd.concat([final,cd],axis=1)    


from sklearn.model_selection import train_test_split
article_train,article_test=train_test_split(final,test_size=0.2,random_state=2)
article_train=article_train.reset_index(drop=True)
y_train=(article_train['cd']==1).astype(int)
y_test=(article_test['cd']==1).astype(int)

article_train=article_train.drop(['cd'],axis=1)
article_test=article_test.drop(['cd'],axis=1)

tf.fit(article_train['ccn'])
train_tf=tf.transform(article_train['ccn'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
               
test_tf=tf.transform(article_test['ccn'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

#y_train.dtypes
#y_train['cd'].value_counts
#y_train['cd']=(y_train['cd']==1).astype(int) 
logr.fit(article_train,y_train)
logr.predict(article_test)
predicted_probs=logr.predict_proba(article_test)[:,1]
predicted_probs>0.51
prediction=np.where(predicted_probs>0.51,1,0)
roc_auc_score(y_test,predicted_probs)
roc_auc_score(y_test,prediction)

cutoffs=np.linspace(0.01,0.99,99)
train_score=logr.predict_proba(x_train_tfidf_scaled)[:,1]
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
