# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 20:25:36 2018

@author: admin
"""

import math as m
m.log(34)
m.log(34,10)
x=True
y=False
x and y
x or y
x="python"
y="py"
y in x
y not in x
len(x),len(y)
x='34'
x+3
x*3
int(x)+3
type(x)
x,y=2,19
a,b,c,d=x+y,x-y,x*y,x/y
a,b,c,d
x**y,x^y
z=(x/y)**(x*y+3)
z
x+' and '+y
x.replace('y','i')
x="StringIndex"
x[4]
x[-5]
x[2:7]
x[2:]
x[-2:3:-1]
x[2::-1]
x[::-1]
x="lalit-sachan"
x.split('-')
z=(20,92,43,83,"lalit","a","c",45)
z[2:7]
z[4]
z[4]=19 #error
z=[20,92,43,83,"lalit","a","c",45]
z[3]
z[3]='change'
z
z[1:4]=70#error
z[1:4]=[70]
z
v=[20,92,43,83,"lalit","a","c",45]
v[2:5]="python"
v[2:5]=["python"]
v

l=[2,4,6]
v[2,4,6]
for i in enumerate(v):  #how to get multiple non continuous elements?
    print(v[i])
    i+=1
    
v.append([23,-18,'two'])    
v
v[-1][0]    

x=[20,92,43,85,'lalit','a','c',45,100]
x=x+[10,20,30]
x  #explicit
x.extend(['one','two','three'])
x  #implace change
print(x)
x.insert(3,'this')  #at position 3, insert element  

x.pop() #remove last
x
x.pop(4) #remove by position
x
x.remove('lalit') #remove by value but only first occurance
x
y=[2,3,40,2,14,2,3,11,71,26]
y.remove(3)
y
y.remove(3)
y #error not in list
y.remove(3)
y
y.reverse()
y  #means implace change
y.sort()   #first read y then output sorted result implace saved
y
y.sort(reverse=True)
y
y[::-1]   #no implace change after run,it's explicit
y
x=[2,3,40,2,14,2,3,11,71,26]
x.reverse()
x
a=x.reverse()
a #they return none,means implace can't be saved in other variable
x=x.reverse()
x #return none, so nothing assign,implace change function can't be assign 
x.sort()   #NoneType can't be assigned
import pandas as pd
file=r'D:\mohit gate\edvancer python\moon_data.csv'
moon1=pd.read_csv(file)

x=6
if x>10:
    print("my value is larger than 10")
elif x>7:
    print("x smaller than 10 and greater than 7")
else:
    diff=10-x
    print("x is smaller than 10 by "+str(diff))
    
x=12
if x>10:
    print("A")
    if x<15:
        print("B")
        
x=12
if x>10:
    print("A")
elif x<15:     #condition met but not run
    print("B")
    
x=(5,40,12,-10,0,32,4,3,6,72)
for value in x:    #list to go through
    if value>10:
        print("x greater than 10")
        print("A block")
    else:
        print("x less than 10")
        print("block B")
print("out")

cities=("mumbai","london","banglore","pune","hyderabad")
for i in cities:
    print("hello")
    num_chars=len(i)
    print("num of char in citiy"+i+' : '+str(num_chars))
    
x=[[1,2],["one","two"],[True,False]]
for i,j in x:
    print(i,j)
    
cities=("mumbai","london","banglore","pune","hyderabad")
list(enumerate(cities))

x=10  
while(x>3):    #conditional loop based
    print(str(x)+" is greater than 3")
    x-=1
    
mylist=range(10,3,-1)
for value in mylist:
    print(value)
    
### game of dice
mysum=0
throws=0
import numpy as np
while mysum<=20 and throws<6:
    throw=np.random.randint(high=7,low=1)
    mysum+=throw
    throws+=1
    print(throw)
    
if throws>6 and mysum<20:
    print('loss')
else:
    print('win')

import math as m    
x=[2,3,4,5,6,8,9,10,1,33,44,12]
logs=[]
for num in x:
    logs.append(m.log(num))
print(x)
print(logs)
    
logs=[m.log(num) for num in x]  #list comprehension   
y=[2,3,4,5,6,8,9,10,0,33,44,12] 
logs=[m.log(num) for num in y] #error due to 0

logs=[]
for num in y:
    if num>0:
        logs.append(m.log(num))
    else:
        logs.append("out of domain error")
        
logs=[m.log(num) for num in y if num>0] #no error came for 0 here

                 #dictionary
#curly bracket, key:value anything, no order internally no iteration,search by key
# key unique, value can same,  

my_dict={'name':'lalit','city':'hyderabad','locality':'gauchi','number of vehicle':2,3:78,4:[3,4,5,6]}
del my_dictr   
my_dict[1] #error no iteration    
my_dict['name'] #call by key
type(my_dict)
my_dict.values
my_dict.keys
del my_dict[4]
my_dict[4] #key error, no key present
for itr in my_dict:
    #print(itr)
    #print(my_dict[itr])
    print(itr,my_dict[itr])
    
tuple=my_dict.items()    #tuples
type(tuple) #dic_items
for key,value in my_dict.items():
    print('key,',key,'value,',value)
    
#   set--------unordered pair of unique values
names={'lalit','spandan','decpta','manoj'}
type(names)     #set
'lalit' in names   #True
'harsh' not in names  #True
names[3] #not indexing
names(3) #no call support
names.add('harsh') #add new element
names.remove('lalit')
for name in names: print(name) 
a={1,2,3,4,5,6}
b={5,6,7,8,9,10}
a not in b
a.union(b)
a.intersection(b)
a.difference(b)
b.difference(a)
a.symmetric_difference(b) #remove intersect elements

        # tuples
t=(34,56,12)   # () for tuple
t1=34,56,12    # tuple too
type(t1)
type(t)

x=7
fib_series=[1,1]
ctr=2
while ctr<=x:
    fib_series.append(fib_series[-1]+fib_series[-2])
    ctr+=1
fib_series


# functions
def fib_series_creater(num_elements):
    fib_series=[1,1]
    ctr=2
    while ctr<num_elements:
        fib_series.append(fib_series[-1]+fib_series[-2])
        ctr+=1
    return(fib_series)
    
fib_series_creater(80)

def mysum(x,y,z):
    return(2*x,3*y,4*z)
mysum(1,2,3)

mysum(y=6,x=3,z=1) # provide names to all
mysum(y=1,z=0) # error, must provide name to all or none
mysum(3,z=-2)  #error again

def test_func(*args):   # don't know how many elements 
    for a in args:
        print(a)
          
test_func(a,78,8,90,4)

def mysum(*args):   # don't know how many elements 
    sum=0       # equate to 0, as it's not list, it's numeric variable
    for arg in args:
        sum+=arg
        #arg+=1
    return(sum)  # return should be in same identation 
        
test_func(a,78,8,90,4) # int and set not add error
test_func(9,78,8,90,4) # 189
sum(9,78,8,90,4)   # error add only 2 numbers together

# class
c1_name='lalit'
c1_balance=100
c1_last_wd=10
 # different variables, different values, but mutual properties, class->object->variable
c2_name='mohit'
c2_balance=90
c2_last_wd=8

class customer():
    def _init_(self,name,balance,last_wd):
        self.name=name
        self.balance=balance
        self.last_wd=last_wd
    def withdraw(self,amount):
        if(amount>self.balance):
            print("insufficient balance")
        else:
            self.balance-=amount
            self.last_wd=amount    
    def deposit(self,amount):
        self.balance+=amount
        return(self.balance)
    


def customer_object(name,balance,last_wd):
    customer_ob=customer()
    customer_ob.name=name
    customer_ob.balance=balance
    customer_ob.last_wd=last_wd
    return(customer_ob)
    
c1=customer_object("mohit",98,67)
c2=customer_object("rohit",68,6)
c2.balance
c1.balance
c1=customer('lalit',100,10)


        
c1.withdraw(20)
c1.balance
c1.last_wd    

c2.deposit(100)


c3=customer_object('saurav',60,7)

# data handling with python
import numpy as np
b=np.array([[2,80,90],[12,8,-10],[2,67,90],[32,18,69]])
             # row 1    row 2      row 3     row 4
    # column 0,1,2      0,1,2     0,1,2      0,1,2
b.shape # 4 rows 3 columns, asymetri array possible in numpy
b[2,1] # start from 0
b[:,1] 
b[1,:]
b[1]
b[[0,1,1],[1,2,1]] #(0,1),(1,2),(1,1)
b%2==0 # logical output
b[b%2==0]
b[2]>=10
b[:,b[2]>=10] # where row 2 >= 10, those columns
b[:0]>=10 # all rows,of column 0, where >= 10 

b[b[:,0]>=10,:]
 
# apply functions on array
import math as m
m.log(b[1]) # math not used for multiple dimentions
# use numpy here
print(b[1])
np.log(b[1])
np.sqrt(b)
np.sum(b) # sum all rows and columns together, we want horizontal vertical sum
np.sum(b,axis=0) #horizontal sum
np.sum(b,axis=1) #vertical sum

# sequential arrays
np.arange(6) # 0 to 5 array
np.arange(3,9) # 3 to 8 array
np.arange(2,10,2) # 2 to 10, step 2, means 2 to 8 result
np.arange(1,9,0.5) # 1 to 8.5
np.linspace(start=2,stop=10,num=20) # start wih 2 to end with 10, 20 numbers total equal step
np.round(np.linspace(start=2,stop=10,num=20),2)
np.random.randint(high=10,low=1,size=(3,)) # between 1 to 10, 3 numbers 
np.random.random(size=(3,4)) # between 0 to 1, 12 numbers, 3 rows 4 columns

# array from manual list
x=[2,10,-19,34,56,76,23,92]
np.random.choice(x,6) # 6 random numbers from array x
help(np.random.choice)

np.random.seed(2)  # seed reproducible functions
np.random.choice(x,4)

np.random.choice(x,4,replace=False) # 4 numbers from x, no repetation
y=np.random.choice(['default','non default'],1000)
np.unique(y)
np.unique(y,return_counts=True)
y=np.random.choice(['default','non default'],1000,p=[0.05,0.95])
np.unique(y,return_counts=True)
x=np.random.randint(high=100,low=12,size=(3,4))
x.sort()  # every list in array sort
x[1].argsort()
x[:,x[1].argsort()] # column move, row same

# pandas : DataFframe creation and its functions
import pandas as pd
age=np.random.randint(low=16,high=80,size=[20,]) # means [20,]=[20]
city=np.random.choice(['mumbai','delhi','chennai','kolkata'],20)
default=np.random.choice([0,1],20)
mydata=pd.DataFrame({'age':age,'city':city,'default':default})
mydata
list(zip([1,2,3],['a','b','c'])) # ist elements zipped together
list(range(6))
list(zip(age,city,default)) # list of list output
mydata=pd.DataFrame(list(zip(age,city,default)))
mydata

# DataFrame from outer data
import numpy as np
import pandas as pd
file=r'D:\mohit gate\edvancer python'
ld=pd.read_csv(file+'/loan_data_train.csv')
ld
ld.head()
ld.shape
ld.columns
ld.dtypes
ld.iloc[5:7,:]
ld.columns
ld['FICO.Range']
ld[['FICO.Range','Interest.Rate']]
ld[(ld['Loan.Purpose']=='credit_card')&(ld['Monthly.Income']>5000)]
ld.loc[(ld['Loan.Purpose']=='Credit_Card')&(ld['Monthly.Income']>5000),['Loan.Purpose','state']]
logic=(ld['Loan.Purpose']=='Credit_Card')&(ld['Monthly.Income']>5000)
np.unique(logic,return_counts=True)
# iLocation based boolean indexing on an integer type is not available
ld[~(ld['Loan.Purpose']=='credit_card')]
ld.drop(['Home.Ownership','Monthly.Income'],axis=1) # explicit, show other columns only
ld=ld.drop(['Home.Ownership','Monthly.Income'],axis=1)
ld.columns # dropped implace now
ld.drop(['Loan.Length','State'],axis=1,implace=True)
ld.columns
del ld['Debt.To.Income.Ratio']

# creating our own dataframe
age=np.random.choice([15,20,30,45,12,10,15,38,7,'missing'],50) # int
fico=np.random.choice(['100-150','150-200','200-250','250-300'],50) #object
city=np.random.choice(['mumbai','delhi','chennai','kolkata'],50) #float
ID=np.arange(50) # int
rating=np.random.choice(['Excellent','Good','Bad','Pathetic'],50) # object
balance=np.random.choice([1000,2000,3000,4000,np.nan,5000,6000],50) # object
children=np.random.randint(high=5,low=0,size=(50,)) # object
mydata=pd.DataFrame({'age':age,'fico':fico,'city':city,'ID':ID,'rating':rating,'balance':balance,'children':children})
mydata.dtypes
mydata['age']=pd.to_numeric(mydata['age'])
#ValueError: Unable to parse string "missing" at position 13
mydata['age']=pd.to_numeric(mydata['age'],errors='coerce')
mydata['balance']=np.log(mydata['balance']) # use any algebric function
mydata
mydata['var2']=mydata['age']/mydata['children']
mydata.isnull().sum()
mydata.loc[mydata['age'].isnull(),'age']=mydata['age'].mean()
mydata['rating_score']=np.where(mydata['rating'].isin(['Good','Excellent']),1,0)
mydata
mydata['rating_score']=np.where(mydata['rating']=='Bad',1,0)
mydata.loc[mydata['rating']=='Pathetic','rating_score']=-1 # capital P
mydata
mydata['fico'].str.split('-') # use str.split 
k=mydata['fico'].str.split('-',expand=True).astype(int) # 0     [150, 200] list of lists
#      0    1    columns
# 0   150  200   first split table format
k.dtypes
mydata['f1'],mydata['f2']=k[0],k[1]
mydata
del mydata['fico']

# FLAG variables
np.unique(mydata['city'],return_counts=True)
mydata['city_mumbai']=np.where(mydata['city']=='mumbai',1,0)
# create dummy column, but what about freequency based creation
mydata['city_kolkata']=np.where(mydata['city']=='kolkata',1,0)
mydata['city_chennai']=np.where(mydata['city']=='chennai',1,0)
(mydata['city']=='chennai').astype(int) # return 1 where chennai
mydata['city_chennai']=(mydata['city']=='chennai').astype(int)
# second method of dummy creation
# only for 0,1 in return but np.wher put anything
mydata['cities']=np.where(mydata['city']=='mumbai','bollywood','free')
mydata
del mydata['cities']
del mydata['city']

# too many unique values, then dummies creation
pd.get_dummies(mydata['rating'])
#       Bad  Excellent  Good  Pathetic
# 0     0          1     0         0
# 1     1          0     0         0
# 2     0          1     0         0
pd.get_dummies(mydata['rating'],prefix='rating',drop_first=True)
#             rating_Excellent  rating_Good  rating_Pathetic
# 0                  1            0                0
# 1                  0            0                0
# 2                  1            0                0
dummy=pd.get_dummies(mydata['rating'],prefix='rating',drop_first=True)
mydata=pd.concat([mydata,dummy],axis=1)

etc=['city_mumbai','city_kolkata','city_chennai']
for col in etc:
    del mydata[col]
mydata
del mydata['rating']
mydata.columns

x=pd.Series(['zero','one','one','alpha','alpha'])
pd.get_dummies(x)
#     alpha  one  zero # alphabetical order in column names
# 0      0    0     1
# 1      0    1     0
# 2      0    1     0
# 3      1    0     0
# 4      1    0     0

df=pd.DataFrame(np.random.randint(2,8,(20,4)),columns=list('ABCD'))
df
# Group_Sort
df.sort_values(['A','B'])
# similar to R sort A then within it's group sort B in ascending order
df
# IDX A  B  C  D   
# 3   2  3  2  6
# 5   2  7  5  4
# 14  2  7  3  7
# 8   3  2  5  2
# 2   3  4  3  5
# 12  4  2  4  2
# 11  4  3  7  3
# 0   4  7  6  6
# 6   4  7  4  3
    
df.sort_values(['A','B'],ascending=[True,False])

# JOINING & MERGING
df1=pd.DataFrame([['a',1],['b',2]],columns=['letter','number'])
df2=pd.DataFrame([['c',3,'cat'],['d',4,'dog']],columns=['letter','number','animal'])
df3=pd.DataFrame([['bird','polly'],['monkey','george'],['tiger','john']],columns=['animal','name'])

df1

#      letter  number
# 0      a       1
# 1      b       2

df2

#     letter  number animal
# 0      c       3    cat
# 1      d       4    dog

pd.concat([df1,df2],axis=0) # ,ignore_index=True) remove index
# axis=0 means rows merge

#   animal letter  number
# 0    NaN      a       1
# 1    NaN      b       2
# 0    cat      c       3
# 1    dog      d       4

df3

#    animal    name
# 0    bird   polly
# 1  monkey  george
# 2   tiger    john

pd.concat([df1,df3],axis=1)
# axis=1 means columns merge

#   letter  number  animal    name
# 0      a     1.0    bird   polly
# 1      b     2.0  monkey  george
# 2    NaN     NaN   tiger    john

df1=pd.DataFrame({'cust id':[1,2,3,4,5],
                  'product':['radio','radio','fridge','fridge','phone']})
df2=pd.DataFrame({'cust id':[3,4,5,6,7],
                  'state':['vp','vp','vp','mh','mh']})
df1
#      cust id product
# 0        1   radio
# 1        2   radio
# 2        3  fridge
# 3        4  fridge
# 4        5   phone

df2
#     cust id state
#0        3    vp
#1        4    vp
#2        5    vp
#3        6    mh
#4        7    mh
pd.merge(df1,df2,on=['cust id'],how='inner')
help(pd.merge)

#    cust id product state
# 0        3  fridge    vp
# 1        4  fridge    vp
# 2        5   phone    vp

pd.merge(df1,df2,on=['cust id'],how='outer')
#    cust id product state
# 0        1   radio   NaN
# 1        2   radio   NaN
# 2        3  fridge    vp
# 3        4  fridge    vp
# 4        5   phone    vp
# 5        6     NaN    mh
# 6        7     NaN    mh

pd.merge(df1,df2,on=['cust id'],how='left')
#    cust id product state
# 0        1   radio   NaN
# 1        2   radio   NaN
# 2        3  fridge    vp
# 3        4  fridge    vp
# 4        5   phone    vp

pd.merge(df1,df2,on=['cust id'],how='right')
#   cust id product state
# 0        3  fridge    vp
# 1        4  fridge    vp
# 2        5   phone    vp
# 3        6     NaN    mh
# 4        7     NaN    mh

help(pd.merge)
#A              >>> B
#        lkey value         rkey value
#    0   foo  1         0   foo  5
#    1   bar  2         1   bar  6
#    2   baz  3         2   qux  7
#    3   foo  4         3   bar  8
#    
#A.merge(B, left_on='lkey', right_on='rkey', how='outer')
#       lkey  value_x  rkey  value_y
#    0  foo   1        foo   5
#    1  foo   4        foo   5
#    2  bar   2        bar   6
#    3  bar   2        bar   8
#    4  baz   3        NaN   NaN
#    5  NaN   NaN      qux   7
#    
#    Returns

#  Numeric Summary
file=r'D:\mohit gate\edvancer python'
bd=pd.read_csv(file+'/bank-full.csv',delimiter=';')
bd
bd.describe()
bd['age'].describe()
#                age        balance           day      duration      campaign  \
# count  45211.000000   45211.000000  45211.000000  45211.000000  45211.000000   
#mean      40.936210    1362.272058     15.806419    258.163080      2.763841   
# std       10.618762    3044.765829      8.322476    257.527812      3.098021   
#min       18.000000   -8019.000000      1.000000      0.000000      1.000000   
# 25%       33.000000      72.000000      8.000000    103.000000      1.000000   
# 50%       39.000000     448.000000     16.000000    180.000000      2.000000   
# 75%       48.000000    1428.000000     21.000000    319.000000      3.000000   
# max       95.000000  102127.000000     31.000000   4918.000000     63.000000   

#              pdays      previous  
# count  45211.000000  45211.000000  
# mean      40.197828      0.580323  
# std      100.128746      2.303441  
# min       -1.000000      0.000000  
# 25%       -1.000000      0.000000  
# 50%       -1.000000      0.000000  
# 75%       -1.000000      0.000000  
# max      871.000000    275.000000

bd.nunique()  
#age            77
#job            12
#marital         3
#education       4
#default         2
#balance      7168
#housing         2
#loan            2
#contact         3
#day            31
#month          12
#duration     1573
#campaign       48
#pdays         559
#previous       41
#poutcome        4
#y               2
#dtype: int64
list(zip(bd.dtypes,bd.nunique())) # 'O' -> character
#[(dtype('int64'), 77),
# (dtype('O'), 12),
# (dtype('O'), 3),
# (dtype('O'), 4),
# (dtype('O'), 2),
# (dtype('int64'), 7168),
# (dtype('O'), 2),
# (dtype('O'), 2),
# (dtype('O'), 3),
# (dtype('int64'), 31),
# (dtype('O'), 12),
# (dtype('int64'), 1573),
# (dtype('int64'), 48),
# (dtype('int64'), 559),
# (dtype('int64'), 41),
# (dtype('O'), 4),
# (dtype('O'), 2)]
pd.crosstab(bd['default'],bd['housing'])
#housing     no    yes
#default              
#no       19701  24695
#yes        380    435
bd.select_dtypes(['object']).columns
#Index(['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
#       'month', 'poutcome', 'y'],
#      dtype='object')

# summary categorical columns data
for col in bd.select_dtypes(['object']).columns:
    print('summary for:'+col)
    print(bd[col].value_counts())
    print('--------------')
    
#summary for:job
#blue-collar      9732
#management       9458
#technician       7597
#admin.           5171
#services         4154
#retired          2264
#self-employed    1579
#entrepreneur     1487
#unemployed       1303
#housemaid        1240
#student           938
#unknown           288
#Name: job, dtype: int64
#--------------
#summary for:marital
#married     27214
#single      12790
#divorced     5207
#Name: marital, dtype: int64
#--------------
#summary for:education
#secondary    23202
#tertiary     13301
#primary       6851
#unknown       1857
#Name: education, dtype: int64
#--------------
#summary for:default
#no     44396
#yes      815
#Name: default, dtype: int64
#--------------
#summary for:housing
#yes    25130
#no     20081
#Name: housing, dtype: int64
#--------------
#summary for:loan
#no     37967
#yes     7244
#Name: loan, dtype: int64
#--------------
#summary for:contact
#cellular     29285
#unknown      13020
#telephone     2906
#Name: contact, dtype: int64
#--------------
#summary for:month
#may    13766
#jul     6895
#aug     6247
#jun     5341
#nov     3970
#apr     2932
#feb     2649
#jan     1403
#oct      738
#sep      579
#mar      477
#dec      214
#Name: month, dtype: int64
#--------------
#summary for:poutcome
#unknown    36959
#failure     4901
#other       1840
#success     1511
#Name: poutcome, dtype: int64
#--------------
#summary for:y
#no     39922
#yes     5289
#Name: y, dtype: int64
#--------------
    
bd.groupby(['job']).mean()
#                     age      balance        day    duration  campaign  \
#job                                                                      
#admin.         39.289886  1135.838909  15.564301  246.896732  2.575324   
#blue-collar    40.044081  1078.826654  15.442561  262.901562  2.816995   
#entrepreneur   42.190989  1521.470074  15.702085  256.309348  2.799597   
#housemaid      46.415323  1392.395161  16.002419  245.825000  2.820968   
#management     40.449567  1763.616832  16.114189  253.995771  2.864348   
#retired        61.626767  1984.215106  15.439488  287.361307  2.346731   
#self-employed  40.484484  1647.970868  16.027866  268.157061  2.853072   
#services       38.740250   997.088108  15.635532  259.318729  2.718344   
#student        26.542644  1388.060768  14.897655  246.656716  2.299574   
#technician     39.314598  1252.632092  16.408582  252.904962  2.906805   
#unemployed     40.961627  1521.745971  15.498081  288.543361  2.432080   
#unknown        47.593750  1772.357639  14.642361  237.611111  3.309028   
#
#                   pdays  previous  
#job                                 
#admin.         47.859021  0.671630  
#blue-collar    44.033498  0.505138  
#entrepreneur   32.486214  0.478144  
#housemaid      21.505645  0.371774  
#management     38.665468  0.668006  
#retired        37.443905  0.638693  
#self-employed  34.747308  0.551615  
#services       41.995185  0.501204  
#student        57.041578  0.953092  
#technician     37.195077  0.574569  
#unemployed     34.146585  0.466616  
#unknown        20.982639  0.319444  

bd.groupby(['job'])['age'].mean()
#job
#admin.           39.289886
#blue-collar      40.044081
#entrepreneur     42.190989
#housemaid        46.415323
#management       40.449567
#retired          61.626767
#self-employed    40.484484
#services         38.740250
#student          26.542644
#technician       39.314598
#unemployed       40.961627
#unknown          47.593750
#Name: age, dtype: float64

bd.pivot_table(values='age',columns='job',index='loan',aggfunc='mean')
#job      admin.  blue-collar  entrepreneur  housemaid  management    retired  \
#loan                                                                           
#no    39.250718    40.161282     42.270557  46.702206   40.330652  62.549872   
#yes   39.455096    39.483967     41.938202  44.361842   41.228252  55.786408   
#
#job   self-employed   services    student  technician  unemployed   unknown  
#loan                                                                         
#no        40.542963  38.850814  26.535637   39.300095   41.041876  47.68662  
#yes       40.139738  38.301435  27.083333   39.384263   40.082569  41.00000

bd.aggregate({'housing':'count','balance':'mean','age':'unique'})  
bd.groupby(['loan']).aggregate({'housing':'count','balance':'mean','age':'unique'})  
#      housing      balance                                                age
#loan                                                                         
#no      37967  1474.453631  [58, 44, 47, 33, 35, 42, 43, 41, 29, 53, 57, 5...
#yes      7244   774.309912  [33, 28, 32, 40, 52, 36, 57, 60, 24, 38, 35, 3...

# VISUALISATION WITH PYTHON
# how variable each other, effect together to response
# ggplot not used now in python, some issues
import pandas as pd
import numpy as np
file=r'D:\mohit gate\edvancer python'
bd=pd.read_csv(file+'/bank-full.csv',delimiter=';')
import seaborn as sns
# DENSITY DISTRIBUTION FUNCTION
sns.distplot(bd['age'])
sns.distplot(bd['age'],kde=False)
sns.distplot(bd['age'],kde=False,norm_hist=True)
sns.distplot(bd['age'],kde=False,norm_hist=True,bins=10)
