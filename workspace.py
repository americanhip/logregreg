import pandas as pd
import numpy as np
import linreg as lg

print("wahoo!")





headers_keep = ['WC', 'Age at Sx', 'BMI', 'GM Repair', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Lateral Imping', 'Side_R', 'Anterior Impinge_Negative', 'Anterior Impinge_Positive', 'Sex_Male']

##### main function #################################################################################
def ml(PRO, dfPRO): #<-- PRO is a string
    factor = 'd' + PRO
    dfPROimp = lg.impute(dfPRO)
    x_col = dfPROimp.loc[:, headers_keep]
    score = lg.logregreg(x_col, dfPROimp[factor], PRO)
    return score
    #headers_keep.extend(heads)
    #
    #dfPRO[factor]
    #print(type(x_col))
    #lg.treereg(x_col, dfPRO[factor], PRO)
    #print(type(dfPRO[factor]))
    
    #logregreg(x_col, dfPRO[factor], PRO)
    #

##### workspace #####################################################################################
dfmHHS = lg.df_procat('mHHS')
dfNAHS = lg.df_procat('NAHS')
dfHOS = lg.df_procat('HOS-SSS')
dfVAS = lg.df_procat('VAS')

mHHSout = ml('mHHS', dfmHHS)
print(mHHSout)

mHHSout = ml('NAHS', dfNAHS)
print(mHHSout)

mHHSout = ml('HOS-SSS', dfHOS)
print(mHHSout)

mHHSout = ml('VAS', dfVAS)
print(mHHSout)