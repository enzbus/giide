import cvxportfolio as cvx
import matplotlib.pyplot as plt
import numpy as np

from giide import transform

rets = cvx.DownloadedMarketData(
    ['SPY', 'QQQ'], cash_key='cash', trading_frequency='monthly').returns.iloc[:,:-1]

rets = rets.dropna()
logrets = np.log(1 + rets)

logrets.cumsum().plot()
plt.title('CUMULATIVE LOGRETURNS, ORIGINAL DATA')

transformed_logrets = transform(logrets, target_loss=1e-2)

# plt.figure()
transformed_logrets.cumsum().plot()
plt.title('CUMSUM PLOT, TRANSFORMED DATA')


plt.show()
