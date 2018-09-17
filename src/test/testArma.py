#coding=utf-8
from __future__ import print_function
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm


from statsmodels.graphics.api import qqplot


### Sunpots Data

print(sm.datasets.sunspots.NOTE)


dta = sm.datasets.sunspots.load_pandas().data


dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
del dta["YEAR"]


dta.plot(figsize=(12,8));


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(dta.values.squeeze(), lags=40, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(dta, lags=40, ax=ax2)


arma_mod20 = sm.tsa.ARMA(dta, (2,0)).fit()
print(arma_mod20.params)


arma_mod30 = sm.tsa.ARMA(dta, (3,0)).fit()


print(arma_mod20.aic, arma_mod20.bic, arma_mod20.hqic)