import numpy as np
import pandas as pd
print("running my stupid baka code")
data = pd.read_csv('data1500.csv', low_memory=False)
pd.set_option('display.max_columns', 60)
print(data.head)

#I HAVE TO DO THIS STUPID SHIT MANUALLY BECAUSE THESE ARE NOT SCOPES AND IF I TRY TO DROP THEM WITH TO_NUMERIC IT WILL MAKE IT NULL AND MIX THEM WITH ACTUAL SCOPE DATA
data.drop(index=data[data['Lateral CEA (Pre-op)'] == 'THA'].index, inplace=True)
data.drop(index=data[data['Lateral CEA (Pre-op)'] == 'NO XRAY'].index, inplace=True)
data.drop(index=data[data['Alpha Angle (Pre-op)'] == 'NO XRAY'].index, inplace=True)
data.drop(index=data[data['Joint Space - Medial (Pre-op)'] == '0.29\t0.29\t0.5\t127'].index, inplace=True)
data.drop(index=data[data['Lateral CEA (Pre-op)'] == 'NO XRAYS'].index, inplace=True)
data.drop(index=data[data['Lateral CEA (Pre-op)'] == 'BHR'].index, inplace=True)
data.drop(index=data[data['Neck-Shaft Angle (Pre-op)'] == '132,4'].index, inplace=True)
data.drop(index=data[data['Alpha Angle (Pre-op)'] == 'N/a'].index, inplace=True)

data.loc[data['Sex'] == '0', 'Sex'] = None #<-- why the fuck is 0 read in as a string
data.dropna(subset=['Sex'], inplace = True)

data['GM Repair'] = data['GM Repair'].fillna(False)
data.loc[data['Anterior Impinge'] == ' Mildly Positive', 'Anterior Impinge'] = 'Mildly Positive'
data = pd.get_dummies(data, columns = ['Side', 'Anterior Impinge', 'Sex'], drop_first = True)

data.loc[data['GM Repair'] == 'No', 'GM Repair'] = False
data.loc[data['GM Repair'] == 'Yes', 'GM Repair'] = True

data.loc[data['Coxa Profunda (Pre-op)'] == 'No', 'Coxa Profunda (Pre-op)'] = False
data.loc[data['Coxa Profunda (Pre-op)'] == 'Yes', 'Coxa Profunda (Pre-op)'] = True

data.loc[data['WC'] == 'FALSE', 'WC'] = False
data.loc[data['WC'] == 'TRUE', 'WC'] = True

data.loc[data['Lateral Imping'] == ' Negative', 'Lateral Imping'] = False
data.loc[data['Lateral Imping'] == 'Positive', 'Lateral Imping'] = True

"""
isch = data.loc[:, 'Ischial Spine (Pre-op)']
cross = data.loc[:, 'Crossover (Pre-op)']
data['Isch true'] = (isch > 0)
data['Cross true'] = (cross > 20)
data['Retroversion'] = (isch > 0 and cross > 20)
data.drop['Ischial Spine (Pre-op)']
data.drop['Crossover (Pre-op)']
"""
print(data.head) #test to see whether data works
print(data.shape)

print(list(data))

data.to_csv('dataformat.csv')

headers_keep = ['WC', 'Age at Sx', 'BMI', 'GM Repair', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Lateral Imping', 'Side_R', 'Anterior Impinge_Negative', 'Anterior Impinge_Positive', 'Sex_Male']

def null_count(df_PRO, headers_keep): #input dataframe, headers_keep
    #sudo
    """
    array = []
    list_index = 0
    for i in headers_keep:
        x = df_PRO[i]
        #access column of df_PRO
        #count nulls and sum
        #append to array
        array.append() #<-- returns a bool telling us whether it's true or false
        list_index += 1
    return array
    """

def df_pro(PRO):
    prePRO = 'Pre ' + PRO
    twoyPRO = '2y ' + PRO
    deltapro = 'd' + PRO
    headers_keep.append(prePRO)
    headers_keep.append(twoyPRO)
    df_PRO = data.loc[:, headers_keep]
    # put an imputer here?
    df_PRO = df_PRO.dropna(subset=[twoyPRO, prePRO])
    x = (df_PRO[twoyPRO] - df_PRO[prePRO] > 0)
    
    df_PRO.drop(twoyPRO, axis=1)
    headers_keep.remove(prePRO)
    headers_keep.remove(twoyPRO)
    nacount = df_PRO.isna().sum()
    
    return df_PRO, nacount


dfmHHS, nullmHHS = df_pro('mHHS')
print(nullmHHS)
print(data.shape)
print("to")
print(dfmHHS.shape)
"""
dfNAHS, nullNAHS = df_pro('NAHS')
dfHOS, nullHOS = df_pro('HOS-SSS')
dfVAS, nullVAS = df_pro('VAS')
print(nullmHHS, nullNAHS, nullHOS, nullVAS)
"""
"""
headers = ['WC', 'Age at Sx', 'BMI', 'GM Repair', 'Pre mHHS', '2y mHHS', 'Pre NAHS', '2y NAHS', 'Pre HOS-SSS', '2y HOS-SSS', 'Pre VAS', '2y VAS', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Lateral Imping', 'Side_R', 'Anterior Impinge_Mildly Positive', 'Anterior Impinge_Negative', 'Anterior Impinge_Positive', 'Sex_Male']

"""
