import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

with open(Path.home()/'cvxportfolio_data/BinanceFutures/BTCUSDT:1h.pickle', 'rb') as f:
    da = pickle.load(f)

open2high = np.log(da.high) - np.log(da['open'])

# sp.stats.probplot(open2high, plot=plt); plt.show()

open2low = np.log(da.low) - np.log(da['open'])
open2close = np.log(da.close) - np.log(da['open'])
open2high_corr = pd.Series(open2high, copy=True)
open2high_corr[open2high == 0.] = np.minimum(open2close[open2high == 0.], 0.)
open2low_corr = pd.Series(open2low, copy=True)
open2low_corr[open2low == 0.] = np.maximum(open2close[open2low == 0.], 0.)

newda = pd.DataFrame({'open2close':open2close, 'open2high_corr':open2high_corr, 'open2low_corr':open2low_corr})

# test with concat
newda = pd.concat([newda.shift(i) for i in range(1)], axis=1).dropna()
newda.columns = np.arange(newda.shape[1])

# # first impl takes a long time
# zio = sp.stats.percentileofscore(newda.open2close, newda.open2close)
# zia = sp.stats.norm.ppf(zio / 100. )#[0, 1])
# plt.scatter(zia, newda.open2close); plt.show()

def slow_transf(input):
    _ = sp.stats.percentileofscore(input, input)
    return sp.stats.norm.ppf(_  / 100. )


# second impl

class Transformer:
    """Transform to Gaussian."""
    def __init__(self, raw_data: pd.Series):
        ###############
        # the ECDF part is easy to extract;
        # it's a np.sort, np.unique, np.cumsum
        # the ppf is sp.special function ndtri
        ###############
        ecdf = sp.stats.ecdf(raw_data).cdf
        self.q, p = ecdf.quantiles, ecdf.probabilities
        # p_corr = (p - p[0]/2)
        p1 = np.concatenate([[0], p])
        p2 = (p1[1:] + p1[:-1]) / 2
        # probably same as
        # p2 = p - 0.5 / len(raw_data)
        self.q_norm = sp.stats.norm.ppf(p2)
        # plt.scatter(q_norm, q); plt.show()
    
    def transf(self, input_data: pd.Series):
        return pd.Series(
            np.interp(input_data, self.q, self.q_norm), index=input_data.index)
        # plt.scatter(transf, raw_data); plt.show()
    
    def transf_back(self, input_data: pd.Series):
        return pd.Series(
            np.interp(input_data, self.q_norm, self.q), index=input_data.index)

test = pd.Series(np.random.uniform(size=1000) > .2).astype(float)
test = open2high_corr # open2close
t = Transformer(test)
transf = t.transf(test)
back = t.transf_back(transf)
assert np.allclose(test, back)

def nonlin(input):
    transformers = []
    output = pd.DataFrame(dtype=float)
    for column in input.columns:
        t = Transformer(input[column])
        transformers.append(t)
        output[column] = t.transf(input[column])

    print(output.describe())
    return output

def lin(input):
    u,s,v = np.linalg.svd(input, full_matrices=False)
    print('SINGULAR VALUES', s)
    loss = 2 * (s[0] - s[-1]) / (s[0] + s[-1])
    print('LOSS', loss)
    return pd.DataFrame(input @ v.T, index=input.index), loss

result = pd.DataFrame(newda, copy=True)
losses = []
for i in range(10000):
    print('ITER', i)
    result, loss = lin(nonlin(result))
    losses.append(loss)
    if loss < 1e-3:
        break

plt.semilogy(losses)

# for i in range(10):
#     plt.figure(figsize=(20,20))
#     x, y = sp.stats.probplot(result.iloc[:,i])[0]
#     sp.stats.probplot(result.iloc[:,i], plot=plt)
#     cs = sp.interpolate.CubicSpline(x,y, bc_type='natural')
#     test = np.linspace(-6,6,1000); plt.plot(test, cs(test)); plt.show()


plt.figure()

sp.stats.probplot(result.iloc[:,1], plot=plt)
plt.figure()
sp.stats.probplot(result.iloc[:,2], plot=plt)

plt.show()

# u,s,v = np.linalg.svd(newda_transf, full_matrices=False)
# zio = newda_transf @ v.T
# sp.stats.probplot(zio.iloc[:,0], plot=plt)
# # sp.stats.probplot(zio.iloc[:,1], plot=plt)
# # sp.stats.probplot(zio.iloc[:,2], plot=plt)

# plt.show()

# test similarity
test = result.iloc[12345]
sim = np.linalg.norm(result - test, axis=1).argsort()
test.plot()
result.iloc[sim[1]].plot()

plt.figure()
newda.iloc[12345].plot()
sim1 = np.linalg.norm(newda - newda.iloc[12345], axis=1).argsort()

newda.iloc[sim[2]].plot()

plt.show()

plt.plot(np.linalg.norm(newda - newda.iloc[12345], axis=1)[sim]); plt.show()

pd.DataFrame([(result.shift(i) * result).mean() for i in range(1,24)]).plot(); plt.show()

Y = result
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

var = VAR(result)

fitted = var.fit(maxlags=6, trend = 'n')

print(fitted.summary())

# X = pd.concat([result.shift(i) for i in range(1, 24*7+1)], axis=1).dropna()
# X.columns = np.arange(X.shape[1])