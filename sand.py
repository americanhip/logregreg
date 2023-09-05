print("running...")
## i use this function to trim down my data frame and print it out to use.
## for some reason it converts my str to bool too. yay!
import numpy as np
import pandas as pd

## man .

#bool conversion
# WC -- 1 if not, 2 if yes
# Sex -- 1 if female, 2 if male
# revision -- 1 if not, 2 if yes
data = pd.read_csv('datatrimmed.csv')
#print(data.shape)
data.to_csv('datatrimmed2.csv')
data2 = pd.read_csv('datatrimmed2.csv')
print("data types")
print(data2.dtypes)
#print(data2.head)
#data2['Sex'] = data2['Sex'].map({'TRUE': True, 'FALSE': False})
#print(data2.dtypes)
"""

headers =['WC', 'Age at Sx', 'Sex', 'Age at onset', 'BMI', 'Radiating Pain', 'Back Pain', 'Acute Injury', 'Pre mHHS', 'Pre NAHS', '1y NAHS', 'Pre HOS-SSS', '1y HOS-SSS', 'Pre VAS', '1y VAS', '1y Satisfaction', 'MRI Alpha angle value', 'MRI Femoral Version value', 'MRI Ligamentum teres tear', 'MRI AVN', 'MRI Gluteus medius pathology', 'MRI Hamstring tendon pathology', 'MRI Trochanteric Bursitis', 'MRI Generalized chondral damage', 'MRI Localized chondral defect (not degenerative)', 'MRI Subchondral cyst - Femur central compartment', 'MRI Subchondral cyst - Femur peripheral compartment', 'MRI Subchondral cyst - Acetabulum central compartment', 'MRI Perilabral cyst', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Protrusio Acetabuli (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Tonnis Grade (Post-op)', 'Ischial Spine (Post-op)', 'Crossover (Post-op)', 'Lateral CEA  (Post-op)', 'Acetabular Inclination (Post-op)', 'Joint Space - Medial (Post-op)', 'Joint Space - Central (Post-op)'  'Joint Space - Lateral (Post-op)', 'Neck-Shaft Angle (Post-op)', 'Coxa Profunda (Post-op)', 'Protrusio Acetabul (Post-op)', 'Anterior CEA (Post-op)', 'Alpha Angle (Post-op)', 'Femoral Offset (Post-op)', 'Gait']
data.rename({"Unnamed: 0":"ok"}, axis="columns",inplace=True)
data.drop(["ok"], axis = 1, inplace=True)
data.to_csv('trimtest2.csv')
print(data)



print(new_data.head())
#new_data = data.drop(index[264,1025769])
print(new_data.shape)

print(new_data)

from sksurv.datasets import load_veterans_lung_cancer
data_x, data_y = load_veterans_lung_cancer()
print(data_x)
print(data_y)
"""