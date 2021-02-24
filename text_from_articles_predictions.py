# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:36:27 2018

@author: admin
"""

import numpy as np
import pandas as pd

import os
path=r'D:\mohit gate\edvancer python'
files=os.listdir(path+'\\reuters_data')

#pth=r'D:\movies\The Conjuring (2013) 720p Blu-Ray\LOST.DIR'
#fi=os.listdir(pth)
#import re
#fi
#os.listdir(pth)=((i+'.jpg') for i in os.listdir(pth))
#    
#out=(re.sub('\\d{3,5}','MATCH',elem) for elem in mylist)
#for value in out:
#    print (value)    
#f.write('this text to my file') # write first open later during update
#f=open('my_first_file.txt','w')
#f.write('he'+'\n')
#help(open)

f=open(path+'\\reuters_data\\training_crude_127.txt','r',encoding='latin-1')
text=""
for line in f:
    if line.strip()=="":continue   #remove blank lines else single text without \n
    else:
        text+=''+line.strip()
print(text)
f.close()


#for line in f:
#    print(line)
f#
#line='i am mohit, i had lot of work to do, but i feel asleep'
#line.strip()

target=[]
article_text=[]
for file in files:
    if '.txt' not in file: continue #print(file)
    f=open(path+'\\reuters_data'+'\\'+file,encoding='latin-1')
    article_text.append(" ".join([line.strip() for line in f if line.strip()!=""]))
    if 'crude' in file:
        target.append('crude')
    else:
        target.append('money')
    f.close

mydata=pd.DataFrame({'target':target,
                     'article_text':article_text})
mydata.head()


mydata['article_text'][0]
mydata['target'].value_counts()    

from sklearn.model_selection import train_test_split
article_train,article_test=train_test_split(mydata,test_size=0.2,
                                            random_state=2)
article_train=article_train.reset_index(drop=True)
y_train=(article_train['target']=='money').astype(int)
y_test=(article_test['target']=='money').astype(int)

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')

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
                   min_df=20,max_df=300,
                   stop_words=my_stop)

tf.fit(article_train['article_text'])
train_tf=tf.transform(article_train['article_text'])
x_train_tf=pd.DataFrame(train_tf.toarray(),
                        columns=tf.get_feature_names())
test_tf=tf.transform(article_test['article_text'])
x_test_tf=pd.DataFrame(test_tf.toarray(),columns=tf.get_feature_names())
x_train_tf.shape
x_test_tf.shape

from sklearn.preprocessing import StandardScaler
st=StandardScaler()
st.fit(x_train_tf)
x_train_tfidf_scaled=st.transform(x_train_tf)
x_test_tfidf_scaled=st.transform(x_test_tf)
