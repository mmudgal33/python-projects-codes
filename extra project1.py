# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 06:56:50 2018

@author: admin
"""
chk.columns
chk=chk.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
chk=pd.read_csv('D:\mohit gate\edvancer python\project 1\chk.csv')
import pandas as pd
import numpy as np
chk=pd.read_csv('D:\mohit gate\edvancer python\project 1\chk.csv')
trn=chk[chk['ttt']==1]
trn.drop(['ttt'],axis=1)
import pandas as pd
import numpy as np
import seaborn as sns
chk=pd.read_csv('D:\mohit gate\edvancer python\project 1\chk.csv')
trn=chk[chk['ttt']==1]
trn.head(5)
join=pd.read_csv('D:\mohit gate\edvancer python\project 1\Final_join.csv')
import pandas as pd
import numpy as np
join=pd.read_csv('D:\mohit gate\edvancer python\project 1\Final_join.csv')
join['Consumer disputed?']=np.where(join['Consumer disputed?'].isin(['Yes']),1,0)
cor=join['Consumer disputed?','ttt']
cor=join[['Consumer disputed?','ttt']]
dat=pd.read_csv('D:\mohit gate\edvancer python\project 1\dat.csv')
company=pd.read_csv('D:\mohit gate\edvancer python\project 1\Company.csv')
join=pd.concat([dat,company,cor],axis=1)
chk=pd.concat([dat,company,cor],axis=1)
trn=chk[chk['ttt']==1]
trn.head(5)
trn=trn.drop(['ttt'],axis=1)
trn.to_csv('D:\mohit gate\edvancer python\project 1\Trn.csv')
sns.heatmap(trn.corr())
import seaborn as sns
sns.heatmap(trn.corr())
union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
union.columns
union=union.drop(['Unnamed: 0','Date received','Consumer complaint narrative','ZIP code','Date sent to company','Company'],axis=1)
union=union.drop(['Unnamed: 0', 'Unnamed: 0.1'],axis=1)
union.dtypes
union['Consumer disputed?']=np.where(union['Consumer disputed?'].isin(['Yes']),1,0)
union['Consumer disputed?']=union['Consumer disputed?'].astype(int)
import seaborn as sns
sns.heatmap(union.corr())
union.to_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
sns.heatmap(union.corr())

## ---(Tue Oct 16 06:26:28 2018)---
union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
import numpy as np
import pandas as pd
union=pd.read_csv('D:\mohit gate\edvancer python\project 1\AUnion.csv')
union.dtypes
union['Unnamed: 0'].head
union=union.drop(['Unnamed: 0'],axis=1)
import seaborn as sns
sns.heatmap(union.corr())
trn=union[union['ttt']==1]
trn=trn.drop(['ttt'],axis=1)
sns.heatmap(trn.corr())
import pandas as pd