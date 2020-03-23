from pandas import Series
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
#Data Loading & Dividing it for Modelling & Validation

xls = pd.ExcelFile('Train_dataset.xlsx')
df = pd.read_excel(xls, 'Diuresis_TS')

melt = df.melt(id_vars='people_ID', var_name='Date', value_name='Diuresis')
melt = melt.sort_values(['Date', 'people_ID'])

from datetime import datetime
con=melt['Date']
melt['Date']=pd.to_datetime(melt['Date'])
melt.set_index('Date', inplace=True)
#check datatype of index
print(melt.index)

ts = melt['Diuresis']
# print(ts.head(10))
# plt.plot(ts)
# plt.show()

from statsmodels.tsa.stattools import adfuller


def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean=timeseries.rolling(12).mean()
    rolstd=timeseries.rolling(12).std()

    # Plot rolling statistics:
    plt.plot(timeseries, color='blue', label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

ts_log=np.log(ts)
ts_log_diff=ts_log-ts_log.shift()
plt.plot(ts_log_diff)
# plt.show()
ts_log_diff.dropna(inplace=True)


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf,pacf

lag_acf=acf(ts_log_diff,nlags=20)
lag_pacf=pacf(ts_log_diff,nlags=20,method='ols')

model=ARIMA(ts_log,order=(2,1,0))
results_AR=model.fit(disp=-1)
plt.plot(ts_log_diff)
plt.plot(results_AR.fittedvalues,color='red')
plt.title('RSS: %.4f' % sum(results_AR.fittedvalues-ts_log_diff)**2)
# plt.show()

predictions_ARIMA=pd.Series(results_AR.fittedvalues,copy=True)
print(predictions_ARIMA.head())

predictions_ARIMA_cumsum=predictions_ARIMA.cumsum()
print(predictions_ARIMA_cumsum.head())

predictions_ARIMA_log=pd.Series(ts_log,index=ts_log.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_cumsum, fill_value=0)
predictions_ARIMA=np.exp(predictions_ARIMA_log)
print(predictions_ARIMA)
# test_stationarity(ts_log_diff)

