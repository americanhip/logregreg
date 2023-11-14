import pandas as pd
import numpy as np
import linreg as lg
import time

print("wahoo!")
start = time.time()

headers_keep = ['WC', 'Age at Sx', 'BMI', 'GM Repair', 'Tonnis Grade (Pre-op)', 'Ischial Spine (Pre-op)', 'Crossover (Pre-op)', 'Lateral CEA (Pre-op)', 'Acetabular Inclination (Pre-op)', 'Joint Space - Medial (Pre-op)', 'Joint Space - Central (Pre-op)', 'Joint Space - Lateral (Pre-op)', 'Neck-Shaft Angle (Pre-op)', 'Coxa Profunda (Pre-op)', 'Anterior CEA (Pre-op)', 'Alpha Angle (Pre-op)', 'Femoral Offset (Pre-op)', 'Lateral Imping', 'Side_R', 'Anterior Impinge_Negative', 'Anterior Impinge_Positive', 'Sex_Male']

##### main function #################################################################################
def ml(PRO, dfPRO): #<-- PRO is a string
    factor = 'd' + PRO
    dfPROimp = lg.impute(dfPRO)
    #headers_keep.append('PASS')
    x_col = dfPROimp.loc[:, headers_keep]
    print('fitting mcid')
    mcid = lg.randforestclass(x_col, dfPROimp[factor], PRO)
    print('fitting pass')
    passscore = lg.randforestclass(x_col, dfPROimp['PASS'], PRO)
    return mcid, passscore
    #headers_keep.extend(heads)
    #
    #dfPRO[factor]
    #print(type(x_col))
    #lg.treereg(x_col, dfPRO[factor], PRO)
    #print(type(dfPRO[factor]))
    """
    print(PRO)
    print(dfPROimp['PASS'].value_counts())
    print(dfPROimp[factor].value_counts())
    print(dfPROimp['change'].value_counts())
    """
    
    #logregreg(x_col, dfPRO[factor], PRO)

##### workspace #####################################################################################

dfmHHS = lg.df_procat('mHHS')
"""
dfNAHS = lg.df_procat('NAHS') #<-- SGD classifier+perceptron or mlp classifier is fine
dfVAS = lg.df_procat('VAS') #<-- randforestclassifier
dfHOS = lg.df_procat('HOS') #<-- randforestclassifier
"""

mHHSout, outtwo = ml('mHHS', dfmHHS)
print('MCID score mHHS', mHHSout, 'PASS score mHHS', outtwo)
print('this took', time.time()-start, 'seconds')
"""
mHHSout, outtwo = ml('NAHS', dfNAHS)
print('MCID score NAHS', mHHSout, 'PASS score NAHS', outtwo)

mHHSout, outtwo = ml('HOS-SSS', dfHOS)
print('MCID score HOS', mHHSout, 'PASS score HOS', outtwo)

mHHSout, outtwo = ml('VAS', dfVAS)
print('MCID score VAS', mHHSout, 'PASS score VAS', outtwo)



mcidmhhs = lg.MCID('Pre mHHS')
mcidnahs = lg.MCID('Pre NAHS')
mcidvas = lg.MCID('Pre VAS')
mcidhos = lg.MCID('Pre HOS-SSS')
"""