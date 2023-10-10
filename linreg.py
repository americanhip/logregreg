#Defining machine learning models and preprocessing functions
#uuuuhghghghugghghhghghguuuuuhghghgughgughghgugu

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.impute import KNNImputer
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

print("yay! yippee! yay!!!")
#start = time.time()
pd.set_option('display.max_columns', 60)
data = pd.read_csv('dataformat.csv')

##### preprocessing ################################################################################

headers_keep = ['WC', 'Age at Sx', 'BMI', 'GM Repair', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Lateral Imping', 'Side_R', 'Anterior Impinge_Negative', 'Anterior Impinge_Positive', 'Sex_Male']

#defining dataframes
def df_pro(PRO):
    prePRO = 'Pre ' + PRO
    twoyPRO = '2y ' + PRO
    deltapro = 'd' + PRO
    
    #print(prePRO)
    #print(twoyPRO)
    #print(deltapro)
    headers_keep.append(prePRO)
    headers_keep.append(twoyPRO)
    df_PRO = data.loc[:, headers_keep]
    df_PRO = df_PRO.dropna(subset=[twoyPRO, prePRO])
    # put an imputer here?
    #df_PRO = df_PRO.dropna() <-- what nulls am i dropping here?
    df_PRO[deltapro] = (df_PRO[twoyPRO] - df_PRO[prePRO])
    df_PRO.drop([twoyPRO], axis=1, inplace=True)
    headers_keep.remove(prePRO)
    headers_keep.remove(twoyPRO)
    return df_PRO

#return a categorical change dataframe instead of a number
def df_procat(PRO):
    prePRO = 'Pre ' + PRO
    twoyPRO = '2y ' + PRO
    deltapro = 'd' + PRO
    
    #print(prePRO)
    #print(twoyPRO)
    #print(deltapro)
    headers_keep.append(prePRO)
    headers_keep.append(twoyPRO)
    df_PRO = data.loc[:, headers_keep]
    df_PRO = df_PRO.dropna(subset=[twoyPRO, prePRO])
    # put an imputer here?
    #df_PRO = df_PRO.dropna() <-- what nulls am i dropping here?
    df_PRO[deltapro] = (df_PRO[twoyPRO] - df_PRO[prePRO] > 0)
    df_PRO.drop([twoyPRO], axis=1, inplace=True)
    headers_keep.remove(prePRO)
    headers_keep.remove(twoyPRO)
    return df_PRO

#split fcn into training/testing sets, ravel into 1D array
def splittrain(x_col, y_col):
    X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.25, random_state=16)
    y_train = y_train.ravel() #create 1D array
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test

#impute data in between to complete dataset
def impute(dfPRO):
    df_num = dfPRO.select_dtypes(include='number')
    df_cat = dfPRO.select_dtypes(include='bool')
    #numerical imputer 
    heads_num = list(df_num)
    imputer = KNNImputer(n_neighbors = 2)
    arrayimputed = imputer.fit_transform(df_num)
    df_num = pd.DataFrame(arrayimputed, columns=heads_num)
    #categorical imputer
    df_cat.replace(True, 1)
    df_cat.replace(False, 0)
    heads_cat = list(df_cat)
    #print('dfcat', df_cat)
    imputer = KNNImputer(n_neighbors = 2)
    arrayimputed = imputer.fit_transform(df_cat)
    df_cat = pd.DataFrame(arrayimputed, columns=heads_cat)
    df_cat.replace(1, True)
    df_cat.replace(0, False)
    #join imputer
    df_full = df_num.join(df_cat)
    return df_full


##### model ################################################################################

# decision tree -- categorical
def treereg(x_col, y_col, dep_var):
    #fjalskdfjs
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    #tree.plot_tree(clf)
    #plt.figure()
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    #print(y_pred_proba)
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title("Goodness of Fit of Logistic Regression Model For " + dep_var)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
#literally 0.5 for all PROs.

# logistic regression function
def logregreg(x_col, y_col, dep_var): #dep_var needs to be a string
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    logreg = LogisticRegression(solver = 'lbfgs', random_state=16, max_iter = 2500, class_weight = 'balanced')
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title("Goodness of Fit of Logistic Regression Model For " + dep_var)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
#bad. worse than random

#random forest classifier function
def randforestclass(x_col, y_col, dep_var):
    #DOES NOT ACCEPT NAN
    
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    x = clf.score(X_test, y_test)

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[::,1]
    #print(y_pred_proba)
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title("Goodness of Fit of Logistic Regression Model For " + dep_var)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    return x
    # figure out how to visualize/validate
#does really good but categorical

#K nearest neighbor regressor
def KNR(x_col, y_col):
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    neigh = KNeighborsRegressor(n_neighbors=2)
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    return score
#sucks ASS. literally negative

#multilayer perceptron regressor
def neuralnet(x_col, y_col):
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    regr = MLPRegressor(random_state=1, max_iter=5000, activation='logistic', solver='lbfgs', verbose = False).fit(X_train, y_train)
    regr.predict(X_test)
    score = regr.score(X_test, y_test)
    return score
#horrible. literally bad and i cant get it working. sad!

#epsilon-support vector regression -- in progress
def supvec(x_col, y_col):
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    regr = SVR(kernel='poly', degree = 24)
    regr.fit(X_test, y_test)
    #regr.predict(X_test)
    x = regr.score(X_test, y_test)
    return x
#at least it's positive this time! still max 0.2 though and thats like after overfitting like hell

#Random forest regressor
def randforestreg(x_col, y_col):
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    clf = RandomForestRegressor()
    clf.fit(X_train, y_train)
    x = clf.score(X_test, y_test)
    return x
#NEGATIVE.....

"""
#print(dfmHHS.isin(['N/a']).any())
#print(dfmHHS.dtypes)
mHHSout = ml('NAHS', dfNAHS)
print(mHHSout)
mHHSout = ml('HOS-SSS', dfHOS)
print(mHHSout)
mHHSout = ml('VAS', dfVAS)
print(mHHSout)

#ml('NAHS', dfNAHS)
#ml('HOS-SSS', dfHOS)
#ml('VAS', dfVAS)

data.to_csv('datatest.csv')
headers = ['WC', 'Age at Sx', 'BMI', 'GM Repair', 'Pre mHHS', '2y mHHS', 'Pre NAHS', '2y NAHS', 'Pre HOS-SSS', '2y HOS-SSS', 'Pre VAS', '2y VAS', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Lateral Imping', 'Side_R', 'Anterior Impinge_Mildly Positive', 'Anterior Impinge_Negative', 'Anterior Impinge_Positive', 'Sex_Male']
"""
