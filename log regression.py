import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
import statsmodels.api as sm

#from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn import metrics

start = time.time()
print("whatever")
pd.set_option('display.max_columns', 60)
#pd.set_option('display.max_rows', 150)
# implementing a logistic regression model using everything else to predict whether there will be a change in PROs.

#read in xlsx file
data = pd.read_csv('datatrimmed2.csv')
#print("input df", data.head())

#drop nulls 
data = data.dropna()
#print(data.shape)

#todo
#clean up splitting data in logregreg function

#create dummy variables
onehotdata = pd.get_dummies(data, columns = ['Gait', 'Tonnis Grade (Pre-op)', 'Tonnis Grade (Post-op)'], drop_first = True)
#print(onehotdata.head)
headers = list(onehotdata)
#print(headers)
pro_change = onehotdata.drop(['1y mHHS','dmHHS', '1y NAHS', 'dNAHS', '1y HOS-SSS','1y VAS','dVAS','dHOS','PRO change'], axis=1)
headers_keep = ['Age at Sx', 'Sex', 'BMI', 'MRI Generalized chondral damage', 'MRI Localized chondral defect (not degenerative)', 'MRI Subchondral cyst - Femur central compartment', 'MRI Subchondral cyst - Femur peripheral compartment', 'MRI Subchondral cyst - Acetabulum central compartment', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)','Ischial Spine (Post-op)', 'Crossover (Post-op)', 'Lateral CEA  (Post-op)', 'Joint Space - Medial (Post-op)', 'Joint Space - Central (Post-op)', 'Joint Space - Lateral (Post-op)', 'Coxa Profunda (Post-op)', 'Anterior CEA (Post-op)', 'Alpha Angle (Post-op)', 'Tonnis Grade (Pre-op)_1', 'Tonnis Grade (Pre-op)_2', 'Tonnis Grade (Post-op)_1', 'Tonnis Grade (Post-op)_2']
x_pick = pro_change.drop(headers_keep, axis=1)


#hard code implementation
x_picked = x_pick.iloc[:, [1, 3, 5]]
print("list to pick from")
x_pickheads = list(x_picked)
print(list(x_picked))
x_pickheads.extend(headers_keep)
print(x_pickheads)
print(onehotdata[x_pickheads])

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

    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    cnf_matrix

    y_pred_proba = logreg.predict_proba(X_test)[::,1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    plt.legend(loc=4)
    plt.title("Goodness of Fit of Logistic Regression Model For " + dep_var)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()



#def featselect(X_train, X_test, y_train, y_test):
    #log = 

"""
#mHHS
#x_trainm, x_testm, y_trainm, y_testm = splittrain(pro_change, onehotdata['dmHHS'])
#feat_colsmhhs = clf(x_trainm, y_trainm)
x_col = pro_change.iloc[:, feat_cols]
print(x_col.head())
logregreg(x_col, onehotdata['dmHHS'], "mHHS")



#NAHS
X_trainN, X_testN, y_trainN, y_testN = train_test_split(pro_change, onehotdata['dNAHS'], test_size=0.25, random_state=16)
y_trainN = y_trainN.ravel()
y_testN = y_testN.ravel()

feat_colsnahs = clf(X_trainN, y_trainN)
x_col = pro_change.iloc[:, feat_colsnahs]
print(x_col.head())
logregreg(x_col, onehotdata['dNAHS'], "NAHS")


#iHOT
X_trainH, X_testH, y_trainH, y_testH = train_test_split(pro_change, onehotdata['dHOS'], test_size=0.25, random_state=16)
y_trainH = y_trainH.ravel()
y_testH = y_testH.ravel()

feat_colsHOS = clf(X_trainH, y_trainH)
x_colh = pro_change.iloc[:, feat_colsHOS]
print(x_colh.head())
logregreg(x_colh, onehotdata['dHOS'], "HOS")


#VAS
X_trainV, X_testV, y_trainV, y_testV = train_test_split(pro_change, onehotdata['dVAS'], test_size=0.25, random_state=16)
y_trainV = y_trainV.ravel()
y_testV = y_testV.ravel()

feat_colsVAS = clf(X_trainV, y_trainV)
x_colv = pro_change.iloc[:, feat_colsVAS]
print(x_colv.head())
logregreg(x_colv, onehotdata['dVAS'], "VAS")





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

end = time.time()
print("this shit took",(end-start), "s")