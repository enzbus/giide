# Copyright (C) 2025 Enzo Busseti
#
# This file is part of GIIDE (Gaussian IID Embedding).
#
# GIIDE is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# GIIDE is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# GIIDE. If not, see <https://www.gnu.org/licenses/>.


"""Simple experiment with historical financial returns.

Transform returns and see if accuracy of toy models improve (very rough).
"""

import cvxportfolio as cvx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from giide import transform

UNIVERSE = ['SPY', 'QQQ', 'TLT', 'XLE', 'EFA']
returns = cvx.DownloadedMarketData(UNIVERSE, cash_key='USDOLLAR').returns
    
NSHIFTS = 10
NCV = 100

rets = returns.iloc[:,:-1]

vix = cvx.YahooFinance('^VIX').data['open']
vix.name = 'VIX'
tmp = pd.concat([returns, pd.DataFrame(vix)], axis=1)
tmp.VIX = tmp.VIX.ffill()
tmp.USDOLLAR = tmp.USDOLLAR.ffill()

for i in range(1,NSHIFTS+1):
    tmp[f'vix_shift_{i}'] = tmp.VIX.shift(i)
    tmp[f'USDOLLAR_shift_{i}'] = tmp['USDOLLAR'].shift(i)


tmp = tmp.dropna()
rets = tmp.iloc[:, :len(UNIVERSE)]
vix_and_rates = tmp.iloc[:, len(UNIVERSE):]

rets = rets.dropna()
targets = rets # **2 # np.log(1 + rets) # rets # rets**2

# logrets.cumsum().plot()
# plt.title('CUMULATIVE LOGRETURNS, ORIGINAL DATA')

transformed_targets = transform(targets, target_loss=1e-3)

targets.rolling(250).std().plot()

plt.figure()
transformed_targets.rolling(250).std().plot()
plt.show()

exit(0)

# # plt.figure()
# transformed_logrets.cumsum().plot()
# plt.title('CUMSUM PLOT, TRANSFORMED DATA')

# plt.figure()
plt.scatter(targets.iloc[:,0], targets.iloc[:,1], alpha=.3)
plt.title('SCATTER PLOT FIRST TWO COLUMNS ORIGINAL')
plt.figure()
plt.scatter(transformed_targets.iloc[:,0], transformed_targets.iloc[:,1], alpha=.3)
plt.title('SCATTER PLOT FIRST TWO COLUMNS TRANSFORMED')

# plt.show()

# import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, KFold

X = pd.DataFrame(vix_and_rates)
X['intercept'] = 1.
X['time'] = (X.index - pd.Timestamp.utcnow()).total_seconds()

# plt.figure()

print('\n'*3 + 'regression on VIX, rates, time, intercept:' + '\n')
cv_result = cross_validate(
    RandomForestRegressor(n_jobs=-1), X, targets, cv=KFold(NCV, shuffle=True), verbose=2, scoring='r2')
print('CROSS VALIDATED ACCURACY RANDOM FOREST ORIGINAL DATA')
print(cv_result['test_score'])
print('MEDIAN R2', np.median(cv_result['test_score']))

print('\n'*3 + f'transformed data regression on VIX, rates, time, intercept:' + '\n')
#     print(sm.OLS(transformed_logrets[col], X).fit_regularized().summary())
cv_result = cross_validate(
    RandomForestRegressor(n_jobs=-1), X, transformed_targets, cv=KFold(NCV, shuffle=True), verbose=2, scoring='r2')
print('CROSS VALIDATED ACCURACY RANDOM FOREST TRANSFORMED DATA')
print(cv_result['test_score'])
print('MEDIAN R2', np.median(cv_result['test_score']))


# for col in targets.columns:
#     print('\n'*3 + f'{col} regression on VIX, rates, time, intercept:' + '\n')
#     cv_result = cross_validate(
#         RandomForestRegressor(n_jobs=-1), X, targets[col], cv=KFold(NCV, shuffle=True), verbose=2, scoring='r2')
#     print('CROSS VALIDATED ACCURACY RANDOM FOREST ORIGINAL DATA')
#     print(cv_result['test_score'])
#     print('MEDIAN R2', np.median(cv_result['test_score']))
#     # plt.plot(cv_result['test_score'], label=col)

#     # print(sm.OLS(logrets[col], X).fit_regularized().summary())

# for col in transformed_targets.columns:
#     print('\n'*3 + f'transformed col {col} regression on VIX, rates, time, intercept:' + '\n')
# #     print(sm.OLS(transformed_logrets[col], X).fit_regularized().summary())
#     cv_result = cross_validate(
#         RandomForestRegressor(n_jobs=-1), X, transformed_targets[col], cv=KFold(NCV, shuffle=True), verbose=2, scoring='r2')
#     print('CROSS VALIDATED ACCURACY RANDOM FOREST TRANSFORMED DATA')
#     print(cv_result['test_score'])
#     print('MEDIAN R2', np.median(cv_result['test_score']))
#     # plt.plot(cv_result['test_score'], label=col)

# plt.legend()
plt.show()

#print(sm.OLS(transformed_logrets.iloc[:,0], vix).fit().summary())
#print(sm.OLS(transformed_logrets.iloc[:,1], vix).fit().summary())
