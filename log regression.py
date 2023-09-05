import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn import metrics

print("whatever")
pd.set_option('display.max_columns', 60)
#pd.set_option('display.max_rows', 150)
# implementing a logistic regression model using everything else to predict whether there will be a change in PROs.
# fuck my life
#

# todo
# figure out stepwise????
# set up regression for each variable --> 4 variates, NAHS, mHHS, IHOT, VAS
# --> set up function for regression
# --> implement for each
# debug regression as needed


#read in xlsx file
data = pd.read_csv('datatrimmed2.csv')
#data.rename({"Unnamed: 0":"ok"}, axis="columns",inplace=True)
#data.drop(["ok"], axis = 1, inplace=True)
print("input df", data.head())

#drop nulls 
data = data.dropna()
print(data.shape)

#headers = ['WC', 'Age at Sx', 'Sex', 'Age at onset', 'BMI', 'Radiating Pain', 'Back Pain', 'Acute Injury', 'Pre mHHS', '1y mHHS', 'dmHHS', 'Pre NAHS', '1y NAHS', 'dNAHS', 'Pre HOS-SSS', '1y HOS-SSS', 'dHOS', 'Pre VAS', '1y VAS', 'dVAS', '1y Satisfaction', 'MRI Alpha angle value', 'MRI Femoral Version value', 'MRI Ligamentum teres tear', 'MRI AVN', 'MRI Gluteus medius pathology', 'MRI Hamstring tendon pathology', 'MRI Trochanteric Bursitis', 'MRI Generalized chondral damage', 'MRI Localized chondral defect (not degenerative)', 'MRI Subchondral cyst - Femur central compartment', 'MRI Subchondral cyst - Femur peripheral compartment', 'MRI Subchondral cyst - Acetabulum central compartment', 'MRI Perilabral cyst', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Tonnis Grade (Post-op)', 'Ischial Spine (Post-op)',  'Crossover (Post-op)', 'Lateral CEA  (Post-op)', 'Acetabular Inclination (Post-op)', 'Joint Space - Medial (Post-op)', 'Joint Space - Central (Post-op)', 'Joint Space - Lateral (Post-op)', 'Neck-Shaft Angle (Post-op)', 'Coxa Profunda (Post-op)', 'Anterior CEA (Post-op)', 'Alpha Angle (Post-op)', 'Femoral Offset (Post-op)', 'Gait']

#print(len(headers))

#create dummy variables
onehotdata = pd.get_dummies(data, columns = ['Gait', 'Tonnis Grade (Pre-op)', 'Tonnis Grade (Post-op)'], drop_first = True)
#print(onehotdata.head)
headers = list(onehotdata)

# logistic regression function
def logregreg(x_col, y_col, dep_var): #dep_var needs to be a string
    # split X and y into training and testing sets
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(x_col, y_col, test_size=0.25, random_state=16)

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

pro_change = onehotdata.drop(['1y mHHS','dmHHS', '1y NAHS', 'dNAHS', '1y HOS-SSS','1y VAS','dVAS','dHOS','PRO change'], axis=1)
#mHHS

#x = onehotdata[list(mhhsx)]
mhhsy = onehotdata[['dmHHS']]
sfs = SequentialFeatureSelector(LogisticRegression(), k_features=25, forward=True, scoring='accuracy',cv=None)
selected_features = sfs.fit(pro_change, mhhsy)
#logregreg(selected_features, mhhsy, "mHHS")



"""
#NAHS
nahsy = onehotdata[['dNAHS']]
logregreg(pro_change, nahsy, "NAHS")

#iHOT
ihoty = onehotdata[['dHOS']]
logregreg(pro_change, ihoty, "HOS")

#VAS
vasy = onehotdata[['dVAS']]
logregreg(pro_change, vasy, "VAS")





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


# list of full headers below
# headers = ['WC', 'Age at Sx,' 'Sex', 'Age at onset', 'BMI', 'Radiating Pain', 'Back Pain', 'Acute Injury', 'Pre mHHS', '1y mHHS', 'dmHHS', 'Pre NAHS', '1y NAHS', 'dNAHS', 'Pre HOS-SSS', '1y HOS-SSS', 'dHOS', 'Pre VAS', '1y VAS', 'dVAS', 'PRO change', '1y Satisfaction', 'MRI Alpha angle value', 'MRI Femoral Version value', 'MRI Ligamentum teres tear', 'MRI AVN', 'MRI Gluteus medius pathology', 'MRI Hamstring tendon pathology', 'MRI Trochanteric Bursitis', 'MRI Generalized chondral damage', 'MRI Localized chondral defect (not degenerative)', 'MRI Subchondral cyst - Femur central compartment', 'MRI Subchondral cyst - Femur peripheral compartment', 'MRI Subchondral cyst - Acetabulum central compartment', 'MRI Perilabral cyst', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Tonnis Grade (Post-op)', 'Ischial Spine (Post-op)',  'Crossover (Post-op)', 'Lateral CEA  (Post-op)', 'Acetabular Inclination (Post-op)', 'Joint Space - Medial (Post-op)', 'Joint Space - Central (Post-op)', 'Joint Space - Lateral (Post-op)', 'Neck-Shaft Angle (Post-op)', 'Coxa Profunda (Post-op)', 'Anterior CEA (Post-op)', 'Alpha Angle (Post-op)', 'Femoral Offset (Post-op)', 'Gait']

#gait levels --> 0 = normal, 1 = right antalgic, 2 = right trendelenberg, 3 = left antalgic, 4 = left trendelenberg, 5 = other
"""