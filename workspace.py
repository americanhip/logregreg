import pandas as pd
import numpy as np
import 
from linreg.py import *

##### main function #################################################################################
def ml(PRO, dfPRO): #<-- PRO is a string
    factor = 'd' + PRO
    dfPROimp = impute(dfPRO)
    x_col = dfPROimp.loc[:, headers_keep]
    score = randforest(x_col, dfPROimp[factor])
    return score
    #headers_keep.extend(heads)
    #
    #dfPRO[factor]
    #print(type(x_col))
    #print(type(dfPRO[factor]))
    #logregreg(x_col, dfPRO[factor], PRO)
    #treereg(x_col, dfPRO[factor], PRO)

##### workspace #####################################################################################

mHHSout = ml('mHHS', dfmHHS)
print(mHHSout)
mHHSout = ml('NAHS', dfNAHS)
print(mHHSout)
mHHSout = ml('HOS-SSS', dfHOS)
print(mHHSout)
mHHSout = ml('VAS', dfVAS)
print(mHHSout)