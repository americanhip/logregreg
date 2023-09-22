#read in data from csv
#use dict to map male to false, female to true
#use dict to map all the xray shit yes to true, no to false
import numpy as np
import pandas as pd

print("formatting data")
data = pd.read_csv('data_414.csv', low_memory=False)
#data = pd.DataFrame(data)
data = data.dropna()
print(data.head())

data.loc[data['Sex'] == 'Male', 'Sex'] = False
data.loc[data['Sex'] == 'Female', 'Sex'] = True

data.loc[data['MRI Labral tear'] == 'No', 'MRI Labral tear'] = False
data.loc[data['MRI Labral tear'] == 'Yes', 'MRI Labral tear'] = True

data.loc[data['MRI Ligamentum teres tear'] == 'No', 'MRI Ligamentum teres tear'] = False
data.loc[data['MRI Ligamentum teres tear'] == 'Yes', 'MRI Ligamentum teres tear'] = True

data.loc[data['MRI AVN'] == 'No', 'MRI AVN'] = False
data.loc[data['MRI AVN'] == 'Yes', 'MRI AVN'] = True

data.loc[data['MRI Gluteus medius pathology'] == 'No', 'MRI Gluteus medius pathology'] = False
data.loc[data['MRI Gluteus medius pathology'] == 'Yes', 'MRI Gluteus medius pathology'] = True

data.loc[data['MRI Generalized chondral damage'] == 'No', 'MRI Generalized chondral damage'] = False
data.loc[data['MRI Generalized chondral damage'] == 'Yes', 'MRI Generalized chondral damage'] = True

data.loc[data['MRI Localized chondral defect (not degenerative)'] == 'Male', 'Sex'] = 0
data.loc[data['MRI Localized chondral defect (not degenerative)'] == 'Female', 'Sex'] = 1

data.loc[data['MRI Subchondral cyst - Femur central compartment'] == 'Male', 'Sex'] = 0
data.loc[data['MRI Subchondral cyst - Femur central compartment'] == 'Female', 'Sex'] = 1

data.loc[data['MRI Subchondral cyst - Femur peripheral compartment'] == 'Male', 'Sex'] = 0
data.loc[data['MRI Subchondral cyst - Femur peripheral compartment'] == 'Female', 'Sex'] = 1

data.loc[data['MRI Subchondral cyst - Acetabulum central compartment'] == 'Male', 'Sex'] = 0
data.loc[data['MRI Subchondral cyst - Acetabulum central compartment'] == 'Female', 'Sex'] = 1

data.loc[data['MRI Perilabral cyst'] == 'Male', 'Sex'] = 0
data.loc[data['MRI Perilabral cyst'] == 'Female', 'Sex'] = 1

data.loc[data['MRI Capsular Laxity'] == 'Male', 'Sex'] = 0
data.loc[data['MRI Capsular Laxity'] == 'Female', 'Sex'] = 1

data.loc[data['Coxa Profunda (Pre-op)'] == 'Male', 'Sex'] = 0
data.loc[data['Coxa Profunda (Pre-op)'] == 'Female', 'Sex'] = 1

data.loc[data['Gait'] == 'Male', 'Sex'] = 0
data.loc[data['Gait'] == 'Female', 'Sex'] = 1

data.loc[data['Anterior Impinge'] == 'Male', 'Sex'] = 0
data.loc[data['Anterior Impinge'] == 'Female', 'Sex'] = 1

data.loc[data['Lateral Imping'] == 'Male', 'Sex'] = 0
data.loc[data['Lateral Imping'] == 'Female', 'Sex'] = 1

data.loc[data['Internal Snapping'] == 'Male', 'Sex'] = 0
data.loc[data['Internal Snapping'] == 'Female', 'Sex'] = 1

data.loc[data['External Snapp'] == 'Male', 'Sex'] = 0
data.loc[data['External Snapp'] == 'Female', 'Sex'] = 1
#d_sex = {'Female': True, 'Male': False}
#d_cond = {'Yes': True, 'No':False}
print(data.head())