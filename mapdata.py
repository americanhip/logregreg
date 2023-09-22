#read in data from csv
import numpy as np
import pandas as pd

print("running my stupid baka code")
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

data.loc[data['MRI Localized chondral defect (not degenerative)'] == 'No', 'MRI Localized chondral defect (not degenerative)'] = False
data.loc[data['MRI Localized chondral defect (not degenerative)'] == 'Yes', 'MRI Localized chondral defect (not degenerative)'] = True

data.loc[data['MRI Subchondral cyst - Femur central compartment'] == 'No', 'MRI Subchondral cyst - Femur central compartment'] = False
data.loc[data['MRI Subchondral cyst - Femur central compartment'] == 'Yes', 'MRI Subchondral cyst - Femur central compartment'] = True

data.loc[data['MRI Subchondral cyst - Femur peripheral compartment'] == 'No', 'MRI Subchondral cyst - Femur peripheral compartment'] = False
data.loc[data['MRI Subchondral cyst - Femur peripheral compartment'] == 'Yes', 'MRI Subchondral cyst - Femur peripheral compartment'] = True

data.loc[data['MRI Subchondral cyst - Acetabulum central compartment'] == 'No', 'MRI Subchondral cyst - Acetabulum central compartment'] = False
data.loc[data['MRI Subchondral cyst - Acetabulum central compartment'] == 'Yes', 'MRI Subchondral cyst - Acetabulum central compartment'] = True

data.loc[data['MRI Perilabral cyst'] == 'No', 'MRI Perilabral cyst'] = False
data.loc[data['MRI Perilabral cyst'] == 'Yes', 'MRI Perilabral cyst'] = True

data.loc[data['MRI Capsular Laxity'] == 'No', 'MRI Capsular Laxity'] = False
data.loc[data['MRI Capsular Laxity'] == 'Yes', 'MRI Capsular Laxity'] = True

data.loc[data['Coxa Profunda (Pre-op)'] == 'No', 'Coxa Profunda (Pre-op)'] = False
data.loc[data['Coxa Profunda (Pre-op)'] == 'Yes', 'Coxa Profunda (Pre-op)'] = True
"""
data.loc[data['Gait'] == 'No', 'Gait'] = False
data.loc[data['Gait'] == 'Yes', 'Gait'] = True
"""

data.loc[data['Anterior Impinge'] == 'Negative', 'Anterior Impinge'] = 0
data.loc[data['Anterior Impinge'] == 'Positive', 'Anterior Impinge'] = 1
data.loc[data['Anterior Impinge'] == 'Mildly Positive', 'Anterior Impinge'] = 2

data.loc[data['Lateral Imping'] == ' Negative', 'Lateral Imping'] = False
data.loc[data['Lateral Imping'] == 'Positive', 'Lateral Imping'] = True

data.loc[data['Internal Snapping'] == 'Negative', 'Internal Snapping'] = False
data.loc[data['Internal Snapping'] == 'Positive', 'Internal Snapping'] = True

data.loc[data['External Snapp'] == 'Negative', 'External Snapp'] = False
data.loc[data['External Snapp'] == 'Positive', 'External Snapp'] = True
#d_sex = {'Female': True, 'Male': False}
#d_cond = {'Yes': True, 'No':False}
print(data.head())
data.to_csv('dataformat.csv')