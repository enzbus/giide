from zipfile import ZipFile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from sklearn.datasets import load_iris


from giide import transform

# # Load Iris dataset
# with ZipFile('iris.zip', 'r') as iris:
#     with iris.open('iris.data') as iris_data:
#         data = pd.read_csv(iris_data, header=None)

# # features
# X = data.iloc[:, :-1]
# assert len(X.dtypes.unique()) == 1
# assert X.dtypes.unique()[0] == np.float64

# # targets
# y = data.iloc[:, -1]

X = pd.DataFrame(load_iris()['data'])
y = pd.Series(load_iris()['target'])

# transform features
X_transf = transform(X,  target_loss=1e-1)

# make a scatterplot like this
# https://en.wikipedia.org/wiki/Iris_flower_data_set#/media/File:Iris_dataset_scatterplot.svg
classes = y.unique()
fig, axes = plt.subplots(4, 4, figsize=(10,10))
fig.suptitle('SCATTER PLOT, ORIGINAL DATA')
for feature1 in range(4):
    for feature2 in range(4):
        if feature1 == feature2:
            continue
        for plot_class in classes:
            axes[feature1, feature2].scatter(
                X.iloc[:, feature1].loc[y==plot_class],
                X.iloc[:, feature2].loc[y==plot_class],
                alpha=.5)

fig, axes = plt.subplots(4, 4, figsize=(10,10))
fig.suptitle('SCATTER PLOT, TRANSFORMED DATA')
for feature1 in range(4):
    for feature2 in range(4):
        if feature1 == feature2:
            continue
        for plot_class in classes:
            axes[feature1, feature2].scatter(
                X_transf.iloc[:, feature1].loc[y==plot_class],
                X_transf.iloc[:, feature2].loc[y==plot_class],
                alpha=.5)

cv_result = cross_validate(
    RandomForestClassifier(), X, y, cv=5)
print('CROSS VALIDATED ACCURACY RANDOM FOREST ORIGINAL DATA')
print(cv_result['test_score'])

cv_result_transf = cross_validate(
    RandomForestClassifier(), X_transf, y, cv=5)
print('CROSS VALIDATED ACCURACY RANDOM FOREST TRANSFORMED DATA')
print(cv_result_transf['test_score'])

plt.show()