import cvxportfolio as cvx
from giide.giide import GIIDE, RegularizedGIIDE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

md = cvx.DownloadedMarketData(
    ['AAPL',
    # 'AMZN', # duplicates
     'XOM', 'BLK', 'GOOG',
    'QQQ', 'SPY', 'GE', 'TLT',
    # 'GLD', # duplicates 
    'HD', 'WMT', 'SBUX'
    ]
, grace_period = pd.Timedelta('7d'), cash_key='cash')

rets = md.returns.dropna().iloc[:, :-1]

# for simplicity, remove duplicates
rets[rets==0] = np.nan
rets = rets.dropna()


rets.index = pd.to_datetime(rets.index.date)
vix = cvx.YahooFinance('^VIX').data['open']
vix.index = pd.to_datetime(vix.index.date)
rets['vix'] = vix + np.random.randn(len(vix)) * 1e-8 # has many many duplicates
rets['time'] = pd.Series((rets.index-pd.Timestamp('2020-01-01')).days, rets.index) * 1.



N_SHIFTS = 1

if N_SHIFTS <= 1:
    full_rets = rets
else:
    full_rets = pd.concat([
        rets.shift(i).rename(lambda x: x + f'_shift_{i}', axis=1)
        for i in range(N_SHIFTS)
    ], axis=1).dropna()

# full_rets.iloc[:,:] = np.random.randn(*full_rets.shape)

TESTMODEL = RegularizedGIIDE # GIIDE
# TESTMODEL = GIIDE

model = TESTMODEL(
    n_layers=1000,
    n_householders=1
)

transformed = pd.DataFrame(model.fit_transform(full_rets.values[:]), full_rets.index)

plt.imshow(transformed.corr()); plt.colorbar(); plt.show()

samples = np.random.randn(10000, full_rets.shape[1])
samples_back = pd.DataFrame(model.transform_back(samples), columns=full_rets.columns)

from scipy.stats import normaltest


print("Avg p-values < 5%", np.mean(normaltest(transformed, axis=0).pvalue < 0.05))
print("Avg p-values < 2.5%", np.mean(normaltest(transformed, axis=0).pvalue < 0.025))
print("Avg p-values < 1%", np.mean(normaltest(transformed, axis=0).pvalue < 0.01))

import scipy.stats as stats
for i in range(3):
    plt.figure()
    stats.probplot(transformed.iloc[:, i], dist="norm", plot=plt)
    plt.title(f"q-q plot of column {i}")

plt.show()

# # raise Exception
# plt.plot(
#     np.array([lay.s for lay in model._layers if isinstance(lay, SVD)])[1:])
# plt.show()


plt.scatter(full_rets['vix'], full_rets['SPY'] * full_rets['QQQ'], alpha=.33)
plt.scatter(samples_back['vix'], samples_back['SPY'] * samples_back['QQQ'], alpha=.1)
plt.title("real and gen r_spy * r_qqq over vix")
plt.show()

plt.scatter(full_rets['time'], full_rets['SPY'] * full_rets['QQQ'], alpha=.33)
plt.scatter(samples_back['time'], samples_back['SPY'] * samples_back['QQQ'], alpha=.1)
plt.title("real and gen r_spy * r_qqq over time")
plt.show()

plt.scatter(full_rets['time'], full_rets['vix'], alpha=.33)
plt.scatter(samples_back['time'], samples_back['vix'], alpha=.1)
plt.title("real and gen vix over time")
plt.show()



### TEST DERIVATIVE
from scipy import optimize as opt
INPUT = {"vix": 16.750000, "time": 2009.0}

def loss_func(rets):
    my_input = np.zeros((1, full_rets.shape[1]))
    my_input[0, :-2] = rets
    my_input[0, -2] = INPUT['vix']
    my_input[0, -1] = INPUT['time']
    return (model.transform(my_input)**2).sum()
results_test = []
for test in range(10):
    result = opt.fmin_l_bfgs_b(
        loss_func,
        x0=np.random.randn(full_rets.shape[1]-2)*0.01,
        # x0=np.zeros(full_rets.shape[1]-2),
        approx_grad=True)
    print(result)
    results_test.append(result[0])

pd.DataFrame(np.array(results_test).T, index=rets.columns[:-2]).plot()
full_rets.loc[full_rets.time==2009].iloc[0,:-2].plot(alpha=.33, label="REAL")
plt.legend()
plt.title("few attempts at max loglik inference, we should pick the one with lowest cost (hopefully there are a few almost equal)")
plt.show()

# backward (conditional distribution)
def loss_func(random_transf):
    t_back = model.transform_back(random_transf.reshape(1,-1))
    return ((t_back[0,-1] - INPUT['time'])/INPUT['time'])**2 + ((t_back[0,-2] - INPUT['vix'])/INPUT['vix'])**2  + ((t_back[0,0] - 0.015509)/0.015509)**2 + ((t_back[0,1] - 0.014517)/0.014517)**2 + ((t_back[0,2] +0.009879)/0.009879)**2 

# we draw a full cloud
draws_test = []
for test in range(100):
    random_seed = np.random.randn(full_rets.shape[1])
    result = opt.fmin_l_bfgs_b(loss_func, x0=random_seed, approx_grad=True)
    print(result)
    draw = pd.Series(model.transform_back(result[0].reshape(1,-1))[0], full_rets.columns)
    print("RANDOM DRAW OF SAMPLE WITH GIVEN VIX AND TIME")
    print(draw)
    draws_test.append(draw)
print("THE REAL OBSERVATION WITH THOSE PARAMETERS")
print(full_rets.loc[full_rets.time==2009].iloc[0])
print("OUT DRAWS")
print(pd.DataFrame(draws_test).T)

pd.DataFrame(draws_test).T.iloc[:-2].plot(alpha=.1)
full_rets.loc[full_rets.time==2009].iloc[0, :-2].plot()
plt.show()


raise Exception

for size in [0 , 1, 2,5, 10, 20, 50, 100]:#, 200, 500, 1000]:
    print(size)
    model = GIIDE(size, n_householders=1, svd=False)
    transformed = pd.DataFrame(model.fit_transform(rets.values))
    # print(transformed.describe())
    print(np.linalg.norm(transformed.corr() - np.eye(transformed.shape[1])))
# print(transformed.corr())
# plt.imshow(transformed.corr())
# plt.colorbar()
# plt.show()

# # features
# rets['time'] = pd.Series((rets.index-pd.Timestamp.utcnow()).days, rets.index) * 1.

# for size in [0 , 1, 2,5, 10, 20, 50, 100, 200, 500, 1000]:
#     print(size)
#     model = GIIDE(size, householder=1, svd=False)
#     transformed = pd.DataFrame(model.fit_transform(rets.values))
#     # print(transformed.describe())
#     print(np.linalg.norm(transformed.corr() - np.eye(transformed.shape[1])))
# # print(transformed.corr())


md1 = cvx.DownloadedMarketData(['SPY'], grace_period = pd.Timedelta('7d'), cash_key='cash')

rets = md1.returns.dropna().iloc[:, :-1]

data = pd.DataFrame({f"SPY_shift{i}":rets['SPY'].shift(i) for i in range(252)}).dropna()

for size in [0 , 1, 2,5, 10, 20, 50, 100]:#, 200, 500, 1000]:
    print(size)
    model = GIIDE(size, n_householders=1, svd=True)
    transformed = pd.DataFrame(model.fit_transform(data.values))
    # print(transformed.describe())
    print(np.linalg.norm(transformed.corr() - np.eye(transformed.shape[1])))