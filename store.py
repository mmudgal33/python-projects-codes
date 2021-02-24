# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

data_file=r'D:\mohit gate\edvancer python\store_train'
import pandas as pd
import math
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np
from sklearn.cross_validation import KFold
%matplotlib inline
ld = pd.read_csv('diabetic_data.csv')
view(ld)
head(ld)
