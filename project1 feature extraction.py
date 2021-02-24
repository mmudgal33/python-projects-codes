# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 19:52:38 2018

@author: admin
"""
tf.fit(cc['Company'])
train_tf=tf.transform(cc['Company'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.to_csv('D:\mohit gate\edvancer python\project 1\Company_Extraction.csv')

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)


cc=pd.concat([cc,x_train_tf],axis=1)
x_train_tfidf_scaled.shape
cc.shape
#mydata=pd.DataFrame(list(zip(cc['Consumer complaint narrative'],cc['Consumer disputed?'])))
#mydata
mydata=pd.DataFrame({'ccn':cc['Company'],
                     'cd':cc['Consumer disputed?']})
list(zip(mydata.columns,mydata.dtypes,mydata.nunique()))
mydata.head()

#[print(cc[i]) for i in cc['ccn']]
mydata['ccn']=np.where(mydata['ccn'].isnull(),'',mydata['ccn'])
#mydata.loc['ccn'=='',:].value_counts()

mydata['ccn'][0]
mydata['cd'].value_counts()    


from sklearn.model_selection import train_test_split
article_train,article_test=train_test_split(mydata,test_size=0.2,
                                            random_state=2)
article_train=article_train.reset_index(drop=True)
y_train=(article_train['cd']=='Yes').astype(int)
y_test=(article_test['cd']=='Yes').astype(int)
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
    
cc['Company'].nunique()    
    
from sklearn.feature_extraction.text import CountVectorizer
tf=CountVectorizer(analyzer=split_into_lemma,
                   min_df=20000,max_df=100000,
                   stop_words=my_stop)

tf.fit(article_train['ccn'])
train_tf=tf.transform(article_train['ccn'])
x_train_tf=pd.DataFrame(train_tf.toarray(),columns=tf.get_feature_names())

test_tf=tf.transform(article_test['ccn'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)
x_test_tfidf_scaled=st.transform(x_test_tf)




    
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
logr=LogisticRegression(class_weight='balanced')

#y_train.dtypes
#y_train['cd'].value_counts
#y_train['cd']=(y_train['cd']==1).astype(int) 
logr.fit(x_train_tf,y_train)
logr.predict(x_test_tfidf_scaled)
predicted_probs=logr.predict_proba(x_test_tfidf_scaled)[:,1]
#predicted_probs>0.51
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