## -*- coding: utf-8 -*-
#"""
#Created on Mon Oct 15 02:34:37 2018
#
#@author: admin
#"""
data.to_csv('D:\mohit gate\edvancer python\project 1\Complaints_Data.csv')
#
#
import numpy as np
import pandas as pd
import seaborn as sns
sns.heatmap(train.corr())
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_test_share.csv')
train['Consumer disputed?']
test['Consumer disputed?']=0
ID=test['Complaint ID']
cmnt1=train[['Consumer complaint narrative','Consumer disputed?']]
cmnt2=test[['Consumer complaint narrative','Consumer disputed?']]
cmntm=pd.concat([cmnt1,cmnt2],axis=0)

train.columns
test.columns
train=train.drop(['Complaint ID'],axis=1)
test=test.drop(['Complaint ID'],axis=1)

comp=pd.read_csv('D:\mohit gate\edvancer python\project 1\Complaints_Extractionn.csv')
union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
data=pd.concat([union,comp],axis=1)
data.columns

union['ttt'].value_counts()

train['rec']=pd.to_datetime(train['Date received'])
train['sent']=pd.to_datetime(train['Date sent to company'])

train['rec'].head(5)
train['wkdr']=train['rec'].dt.weekofyear
train['wkds']=train['sent'].dt.weekofyear

train['www']=np.where(train['wkdr']>25,1,0)

pd.crosstab(train['wkds'],train['Consumer disputed?'])

train['Date received'].head(10)

test['Consumer disputed?']=0
train['ttt']='1'
test['ttt']='0'

union=pd.concat([train,test],axis=0)

union.to_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
trn=union[union['ttt']==1]
trn=trn.drop(['ttt'],axis=1)

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\Aunion_tt.csv')
union['Unnamed: 0'].head
#sns.heatmap(union.corr())
union.columns
del union
union['Consumer disputed?'].value_counts()
#union.iloc[union['ttt']=='0','Consumer disputed?']='NA'
union['Consumer disputed?']=np.where(union['Consumer disputed?'].isin(['Yes','No']),union['Consumer disputed?'],'NA')
train['Consumer disputed?']=np.where(train['Consumer disputed?'].isin(['Yes']),1,0)


join=pd.concat([train,test],axis=1)

#del train, test
union.dtypes
list(zip(train.columns,train.dtypes,train.nunique(),train.isnull().sum()))

#company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')
#cc.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction1.csv')
#final=pd.concat([dat,company,com],axis=1)
df=pd.read_csv('D:\mohit gate\edvancer python\project 1\Consumer_Complaints_train.csv')

cc.columns
cc.head
union=union.drop(['Unnamed: 0','Date received','Consumer complaint narrative','ZIP code','Date sent to company','Company'],axis=1)
union=union.drop(['Unnamed: 0'],axis=1)

union.columns
cc=cc.drop(['Date received',
       'Consumer complaintc narrative', 'Company',
       'State', 'ZIP code',
       'Submitted via', 'Date sent to company',
       'Complaint ID'],axis=1)

##################################################################################

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
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Complaints_Extractionn.csv')


####################################################################################
#
#dat=pd.DataFrame({'Date received': union['Date received'],
#                 'Date sent to company': union['Date sent to company'],
#                 'Consumer disputed?': union['Consumer disputed?']})
#    
#dat=pd.DataFrame({'Date received': train['Date received'],
#                 'Date sent to company': train['Date sent to company'],
#                 'Consumer disputed?': train['Consumer disputed?']})    
# 
#dat['rec']=pd.to_datetime(dat['Date received'])
#dat['sent']=pd.to_datetime(dat['Date sent to company'])
#
#dat['wkdr']=dat['rec'].dt.weekofyear
#dat['wkds']=dat['sent'].dt.month
#
#
#jon['wkwr']=pd.to_numeric(jon['wkwr'])
#jon['wkwr']=pd.to_numeric(jon['wkwr'])
#jon['wkdr']=jon['wkdr'].astype(int)
#jon['wkds']=jon['wkds'].astype(int)    
#jon['wkwr'].dtypes
#
#jon=pd.concat([train,dat],axis=1)
#jon.columns    
#import seaborn as sns
#sns.heatmap(train.corr())
#train.dtypes
#jon[['wkdr','wkds']].head
#    
#union.columns
#dat['rec']=pd.to_datetime(dat['Date received'])
#dat['sent']=pd.to_datetime(dat['Date sent to company'])
#dat['wkdr']=dat['rec'].dt.weekday_name
#dat['wkds']=dat['sent'].dt.weekday_name
#
#dat['wkdr']=np.where(dat['wkdr'].isin(['Saturday','Sunday']),1,0)
#dat['wkds']=np.where(dat['wkds'].isin(['Saturday','Sunday']),1,0)
#
#
#dat['s']=dat['sent'].dt.date
#dat['r']=dat['rec'].dt.date
#dat['d']=(dat['s']-dat['r']).astype(str)
#dat['d'].dtypes
#dat['d'].value_counts()
#
##mydata['rating_score']=np.where(mydata['rating'].isin(['Good','Excellent']),1,0)
#dat['d'].head
#
#k=dat['d'].str.split(' days ',expand=True)
#k
#dat['d']=k[0]
#dat['d']=pd.to_numeric(dat['d'])
#dat['d']=np.where(dat['d']<10,1,0)
#dat['d'].value_counts()
#dat['d'].dtypes
#
#
## hh.to_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
#
#
##dat['diff']=(dat['wkdts']-dat['wkdtr'])
##dat['diff'].describe()
##lal=['0 days 00:00:00','1 days 00:00:00','2 days 00:00:00','3 days 00:00:00','4 days 00:00:00','5 days 00:00:00','6 days 00:00:00','7 days 00:00:00','-1 days 00:00:00','8 days 00:00:00','9 days 00:00:00','10 days 00:00:00']
#
#
#
#dat['wkwr'].dtypes
#dat['wkdtr']=dat['rec'].dt.day
#dat['wkwr']=np.where(dat['wkdtr']>20,1,0)
#
#dat['wkdts']=dat['sent'].dt.day
#dat['wkws']=np.where(dat['wkdts']>20,1,0)
#dat['wkws'].head
#
#dat[['rec','sent']]
#dat.columns
#datli=['wkdr', 'wkds','d','wkwr','wkws']
#dated=dat[datli]
#
##dated.to_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
##
##dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
##
##dat['wkdr'].value_counts()
##dte=['Consumer disputed?', 'Date received', 'Date sent to company', 'rec',
##       'sent', 'wkdr', 'wkds', 'wkwr', 'wkws']
##dat=dat.drop(['Consumer disputed?', 'Date received', 'Date sent to company', 'rec',
##       'sent'],axis=1)
##dat.to_csv('D:\mohit gate\edvancer python\project 1\date_Extraction.csv')
##union=pd.concat([union,dat],axis=1)
##
##train.columns
##join.columns
##pd.crosstab(join['credit'],join['Consumer disputed?'])
##
##import seaborn as sns
##myplot=sns.countplot(x='diff',hue='Consumer disputed?',data=dat)
### dat['wkdr1']=dat['rec'].dt.weekday
### dat['wkds2']=dat['sent'].dt.weekday
##dat.head
## dat=dat.drop(['wkdr1','wkds2'],axis=1)
## dat['wkdr']=np.where(dat['wkdr'].isin(['Monday','Tuesday','Wednesday','Thursday','Friday']),1,0)
#
## mydata['rating_score']=np.where(mydata['rating'].isin(['Good','Excellent']),1,0)
#
## 'Saturday','Sunday','Monday','Tuesday','Wednesday','Thursday','Friday'
#
#
#####################################################################################
#
#del union

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
                   min_df=20000,max_df=250000,
                   stop_words=my_stop)


mydata=pd.DataFrame({'ccn':cmntm['Company'],
                     'cd':cmntm['Consumer disputed?']})
list(zip(mydata.columns,mydata.dtypes,mydata.nunique()))
mydata.head()

mydata=pd.DataFrame({'ccn':cmntm['Consumer complaint narrative'],
                     'cd':cmntm['Consumer disputed?']})

#[print(cc[i]) for i in cc['ccn']]
mydata['ccn']=np.where(mydata['ccn'].isnull(),'',mydata['ccn'])
#mydata.loc['ccn'=='',:].value_counts()

mydata['ccn'][500000:]
mydata['ccn'].head
mydata['ccn'].value_counts() 
mydata['ccn'].dtypes

tf.fit(mydata['ccn'])
train_tf=tf.transform(mydata['ccn'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape

x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Complaints_Extractionn.csv')

               
test_tf=tf.transform(mydata['ccn'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)
x_test_tfidf_scaled=st.transform(x_test_tf)

###################################################################################
#
#tf.fit(cc['Company'])
#train_tf=tf.transform(cc['Company'])
#x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
#final.to_csv('D:\mohit gate\edvancer python\project 1\Final_Extraction.csv')
#final['cd'].dtypes
#dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\date_Extraction.csv')
#dat.columns
#company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')
#company.columns
#complaints=pd.read_csv('D:\mohit gate\edvancer python\project 1\Complaints_Extraction.csv')
#df=pd.read_csv('D:\mohit gate\edvancer python\project 1\other_Information.csv')


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

join.columns
join.head(5)
join=join.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1)

import pandas as pd
import numpy as np
import seaborn as sns
sns.heatmap(train.corr())
chk=pd.read_csv('D:\mohit gate\edvancer python\project 1\chk.csv')
cor=join[['Consumer disputed?','ttt']]
chk=pd.concat([dat,company,cor],axis=1)

trn=cor[cor['ttt']==1]

trn.head(5)
trn=trn.drop(['ttt'],axis=1)

join['Consumer disputed?']=np.where(join['Consumer disputed?'].isin(['Yes']),1,0)

join.dtypes
company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company.csv')
dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
complaints=pd.read_csv('D:\mohit gate\edvancer python\project 1\Complaints_Extractionn.csv')

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')


gp=union[['ttt','Consumer disputed?']]
del union
join=pd.concat([dat,complaints,company],axis=1)
join=pd.concat([union,company])
chk=pd.concat([dat,company,cor],axis=1)
gp=pd.concat([dat,gp],axis=1)

train=train.drop(['ttt'],axis=1)
train=gp[gp['ttt']==1]
test=join[join['ttt']==0]
train.to_csv('D:\mohit gate\edvancer python\project 1\Final_train.csv')
test.to_csv('D:\mohit gate\edvancer python\project 1\Final_test.csv')
join.to_csv('D:\mohit gate\edvancer python\project 1\Final_join.csv')
join=pd.read_csv('D:\mohit gate\edvancer python\project 1\Final_join.csv')
train=pd.read_csv('D:\mohit gate\edvancer python\project 1\Final_train.csv')
test=pd.read_csv('D:\mohit gate\edvancer python\project 1\Final_test.csv')

union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
union.columns
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
train.columns

train.dtypes
union['ttt'].value_counts()
train=train.drop(['Unnamed: 0','Unnamed: 0.1.1.1'],axis=1)
test=test.drop(['Unnamed: 0','Unnamed: 0.1.1.1'],axis=1)
test.isnull().sum()






#print(x_train.loc[:100])
#
#kf=KFold(n_splits=10)
#train, test in kf.split(x_train)

# LASSO REGULARISATION
 
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso   #Ridge
from sklearn.model_selection import KFold

ld_train=ld_train.reset_index(drop=True) # kfold use index to split
x_train=ld_train.drop('Interest.Rate',axis=1)
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




from sklearn.model_selection import train_test_split
article_train,article_test=train_test_split(train,test_size=0.2,random_state=2)
article_train=article_train.reset_index(drop=True)
y_train=(article_train['Consumer disputed?']==1).astype(int)
y_test=(article_test['Consumer disputed?']==1).astype(int)

article_train=article_train.drop(['Consumer disputed?'],axis=1)
article_test=article_test.drop(['Consumer disputed?'],axis=1)

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


from sklearn.linear_model import Ridge  # Lass
from sklearn.model_selection import KFold

train=train.reset_index(drop=True) # kfold use index to split
x_train=train.drop('Consumer disputed?',axis=1)
y_train=train['Consumer disputed?']

lambdas =np.linspace(1,100,100)
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
    
best_alpha=lambdas[mae_list==min(mae_list)]
print('Alpha with min mae_10cv error is:',best_alpha)

ridge=Ridge(fit_intercept=True,alpha=best_alpha)
ridge.fit(x_train,y_train)
p_test=ridge.predict(x_test)
list(zip(x_train.columns,ridge.coef_))

mae_ridge=mean_absolute_error(ld_test['Interest.Rate'],p_test)
mae_ridge



# Extract features from all text articles in data
tf.fit(cc['Company'])
train_tf=tf.transform(cc['Company'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)