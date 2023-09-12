import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, mean_absolute_error
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn import metrics
import time


print("whatever")
start = time.time()
pd.set_option('display.max_columns', 60)
#pd.set_option('display.max_rows', 150)
#implementing a logistic regression model using everything else to predict whether there will be a change in PROs.

#read in xlsx file
data = pd.read_csv('datatrimmed2.csv')
print("input df", data.head())

#drop nulls 
data = data.dropna()
print(data.shape)

#create dummy variables
onehotdata = pd.get_dummies(data, columns = ['Gait', 'Tonnis Grade (Pre-op)', 'Tonnis Grade (Post-op)'], drop_first = True)
headers = list(onehotdata)
#print(headers)
pro_change = onehotdata.drop(['1y mHHS','dmHHS', '1y NAHS', 'dNAHS', '1y HOS-SSS','1y VAS','dVAS','dHOS','PRO change'], axis=1)
headers_keep = ['Age at Sx', 'Sex', 'BMI', 'MRI Generalized chondral damage', 'MRI Localized chondral defect (not degenerative)', 'MRI Subchondral cyst - Femur central compartment', 'MRI Subchondral cyst - Femur peripheral compartment', 'MRI Subchondral cyst - Acetabulum central compartment', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Crossover (Post-op)', 'Lateral CEA  (Post-op)', 'Anterior CEA (Post-op)', 'Alpha Angle (Post-op)', 'Tonnis Grade (Pre-op)_1']
#preop tonnis
x_pick = pro_change.drop(headers_keep, axis=1)

#split data function
def splittrain(x_col, y_col):
    X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.25, random_state=16)
    y_train = y_train.ravel() #create 1D array
    y_test = y_test.ravel()
    return X_train, X_test, y_train, y_test

# logistic regression function
def logregreg(x_col, y_col, dep_var): #dep_var needs to be a string
    # split X and y into training and testing sets
    
    
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)


    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(solver = 'lbfgs', random_state=16, max_iter = 2500)

    # fit the model with data
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    # import the metrics class


    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title("Goodness of Fit of Logistic Regression Model For " + dep_var)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def linreg(x_col, y_col):
    X_train, X_test, y_train, y_test = splittrain(x_col, y_col)
    linregreg = LinearRegression().fit(X_train, y_train)
    linregreg.score(X_train, y_train)

#random forest classifier feature selection
def clf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    shapex = X_train.shape
    size = round(shapex[1] * 0.75)
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
    x_train, x_test, y_train, y_test = splittrain(x_pick, onehotdata[factor])
    feat_cols = clf(x_train, y_train)
    test = x_pick.iloc[:, feat_cols]
    heads = list(test)
    headers_keep.extend(heads)
    x_col = pro_change.loc[:, headers_keep]
    logregreg(x_col, onehotdata[factor], PRO)

ml('mHHS')

ml('NAHS')

ml('HOS')

ml('VAS')
end = time.time()
print("\nthis took",(end-start), "s")

"""
#mHHS
x_trainm, x_testm, y_trainm, y_testm = splittrain(x_pick, onehotdata['dmHHS'])
feat_colsmhhs = clf(x_trainm, y_trainm)
test = x_pick.iloc[:, feat_colsmhhs]
heads = list(test)
headers_keep.extend(heads)
x_col = pro_change.loc[:, headers_keep]
logregreg(x_col, onehotdata['dmHHS'], "mHHS")


#NAHS
x_trainm, x_testm, y_trainm, y_testm = splittrain(x_pick, onehotdata['dNAHS'])
feat_colsmhhs = clf(x_trainm, y_trainm)
test = x_pick.iloc[:, feat_colsmhhs]
heads = list(test)
headers_keep.extend(heads)
x_col = pro_change.loc[:, headers_keep]
logregreg(x_col, onehotdata['dNAHS'], "NAHS")



#iHOT
x_trainm, x_testm, y_trainm, y_testm = splittrain(x_pick, onehotdata['dHOS'])
feat_colsmhhs = clf(x_trainm, y_trainm)
test = x_pick.iloc[:, feat_colsmhhs]
heads = list(test)
headers_keep.extend(heads)
x_col = pro_change.loc[:, headers_keep]
logregreg(x_col, onehotdata['dHOS'], "HOS")


#VAS
x_trainm, x_testm, y_trainm, y_testm = splittrain(x_pick, onehotdata['dVAS'])
feat_colsmhhs = clf(x_trainm, y_trainm)
test = x_pick.iloc[:, feat_colsmhhs]
heads = list(test)
headers_keep.extend(heads)
x_col = pro_change.loc[:, headers_keep]
logregreg(x_col, onehotdata['dVAS'], "VAS")

##### NOTES #####


#define independent and dependent variables
onehotdatax = onehotdata.drop('PRO change', axis=1)
headers = list(onehotdatax)


headers.remove('Sex')

headermhhs = headers
headermhhs.remove('1y mHHS')
headermhhs.remove('1y NAHS')
headermhhs.remove('dNAHS')
headermhhs.remove('1y HOS-SSS')
headermhhs.remove('1y VAS')
headermhhs.remove('dVAS')
headermhhs.remove('dHOS')
headermhhs.remove('PRO change')
mhhsdata = onehotdata[headermhhs]
print(list(mhhsdata))
headermhhsy = headermhhs
headermhhsy.remove('dmHHS')
print(headermhhsy)


onehotdatax = onehotdata.drop('PRO change', axis=1)
headers = list(onehotdatax)
#data.rename({"Unnamed: 0":"ok"}, axis="columns",inplace=True)
#data.drop(["ok"], axis = 1, inplace=True)


# list of full headers below
# headers = ['WC', 'Age at Sx,' 'Sex', 'Age at onset', 'BMI', 'Radiating Pain', 'Back Pain', 'Acute Injury', 'Pre mHHS', '1y mHHS', 'dmHHS', 'Pre NAHS', '1y NAHS', 'dNAHS', 'Pre HOS-SSS', '1y HOS-SSS', 'dHOS', 'Pre VAS', '1y VAS', 'dVAS', 'PRO change', '1y Satisfaction', 'MRI Alpha angle value', 'MRI Femoral Version value', 'MRI Ligamentum teres tear', 'MRI AVN', 'MRI Gluteus medius pathology', 'MRI Hamstring tendon pathology', 'MRI Trochanteric Bursitis', 'MRI Generalized chondral damage', 'MRI Localized chondral defect (not degenerative)', 'MRI Subchondral cyst - Femur central compartment', 'MRI Subchondral cyst - Femur peripheral compartment', 'MRI Subchondral cyst - Acetabulum central compartment', 'MRI Perilabral cyst', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Tonnis Grade (Post-op)', 'Ischial Spine (Post-op)',  'Crossover (Post-op)', 'Lateral CEA  (Post-op)', 'Acetabular Inclination (Post-op)', 'Joint Space - Medial (Post-op)', 'Joint Space - Central (Post-op)', 'Joint Space - Lateral (Post-op)', 'Neck-Shaft Angle (Post-op)', 'Coxa Profunda (Post-op)', 'Anterior CEA (Post-op)', 'Alpha Angle (Post-op)', 'Femoral Offset (Post-op)', 'Gait']

#gait levels --> 0 = normal, 1 = right antalgic, 2 = right trendelenberg, 3 = left antalgic, 4 = left trendelenberg, 5 = other
"""
