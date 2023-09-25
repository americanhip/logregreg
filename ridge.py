#trying ridge regession
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn import metrics
import time



print("whatever")
start = time.time()
pd.set_option('display.max_columns', 60)

data = pd.read_csv('dataformat.csv')
print("input df", data.head())

#drop nulls 
data = data.dropna()
print(data.shape)

headers = list(data)


#


headers_keep = ['Age at Sx','Sex', 'BMI', 'Pre mHHS', 'Pre NAHS', 'Pre HOS-SSS', 'Pre VAS', 'WC', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)',  'Lateral Imping', 'Anterior Impinge_1', 'Anterior Impinge_2']
print(len(headers))
print(len(headers_keep))
#preop tonniskjlkjlkjlkjlkjlkjlkjkjl.jlkkjlkkjlkjlkjlkjljljljljlj
x_pick = data.drop(headers_keep, axis=1)
x_pick = x_pick.drop(['dmHHS', 'dNAHS', 'dHOS', 'dVAS'], axis = 1)

#split data function
def splittrain(x_col, y_col):
    X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.25, random_state=16)
    # y_train = y_train.ravel() #create 1D array
    # y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test

# logistic regression function
def ridgereg(x_col, y_col): #dep_var needs to be a string
    # split X and y into training and testing sets
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    reg = Ridge(alpha=.5)
    fitridge = reg.fit(x_col, y_col)
    scoreridge = reg.score(x_col, y_col)
    return fitridge, scoreridge
    

#random forest classifier feature selection
def clf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    shapex = X_train.shape
    size = round(shapex[1] * 0.66)
    #print(size)
    sfs1 = sfs(clf, k_features= size, forward=True, floating=False, verbose=2, scoring='accuracy', cv=5)
    #selected_features = sfs.fit(pro_change, mhhsy)
    sfs1 = sfs1.fit(X_train, y_train)
    feat_cols = list(sfs1.k_feature_idx_)
    #print(feat_cols)
    return feat_cols


#putting it all together
def ml(PRO): #<-- PRO is a string
    factor = 'd' + PRO
    x_train, x_test, y_train, y_test = splittrain(x_pick, data[factor])
    feat_cols = clf(x_train, y_train)
    test = x_pick.iloc[:, feat_cols]
    heads = list(test)
    headers_keep.extend(heads)
    x_col = data.loc[:, headers_keep]
    fitpro, scorepro = ridgereg(x_col, data[factor])
    return fitpro, scorepro
    #logregreg(x_pick, data[factor], PRO)


fit, score = ml('mHHS')

"""
ml('NAHS')

ml('HOS')

ml('VAS')

end = time.time()
print("\nthis took",(end-start), "s")
"""
