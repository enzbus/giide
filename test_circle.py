import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from giide import GaussianTransformer, nonlin
from sklearn.preprocessing import QuantileTransformer

N = 10000 # n input samples
NSAMP = N # n generate samples
M = 1000 # n layers model

TRIGON = False
ABS = False
HETEROSCHED = False
POWER = False
XOR = True

if TRIGON:
    # TRIGON TEST
    x = np.random.uniform(0, 2 * np.pi, N)
    data = pd.DataFrame({
        'cos': np.cos(x),
        'sin': np.sin(x),
        # 'x': x,
        # 'x2': x**2,
    })

if ABS:
    # ABS TEST
    x = np.random.uniform(-1,1, N)
    data = pd.DataFrame({
        'pos': np.maximum(x, 0.),
        'neg': np.minimum(x, 0.),
    })

if HETEROSCHED:
    # HETEROSCHED TEST
    y = np.random.uniform(-1, 1, N)
    x = np.random.randn(N) * y
    data = pd.DataFrame({
        'x': x,
        'y': y,
    })

if POWER:
    # POWER TEST
    x = np.random.randn(N)
    data = pd.DataFrame({
        'x': x,
        'x2': x**2,
        'x3': x**3,
        'x4': x**4,
    })

if ABS:
    # ABS TEST
    x = np.random.uniform(-1,1, N)
    data = pd.DataFrame({
        'x': x,
        'sign(x)': np.sign(x),
    })

if XOR:
    # XOR TEST
    x1 = np.sign(np.random.uniform(-1,1, N))
    x2 = np.sign(np.random.uniform(-1,1, N))
    data = pd.DataFrame({
        'x1': x1,
        'x2': x2,
        'XOR': (x1 != x2) * 1.0,
        # 'NOISE': np.random.randn(N),
    })


data_orig = pd.DataFrame(data, copy=True)

def scatter_plot(mydata):

    # Assume df is your DataFrame
    # plt.figure()
    pd.plotting.scatter_matrix(mydata, figsize=(10, 10), diagonal='kde')
    plt.tight_layout()

    # plt.scatter(
    #     mydata.iloc[:, 0], mydata.iloc[:, 1])
    # plt.show()

def random_rotation(size):
    a = np.random.randn(size,size),
    q,_ = np.linalg.qr(a)
    return q[0]

INTERACTIVE = False
if INTERACTIVE:
    scatter_plot(data)
    data_transf = nonlin(data)
    scatter_plot(data_transf)
    rot = random_rotation(2)
    data_transf_rot = data_transf @ rot
    scatter_plot(data_transf_rot)
    data_transf_rot_transf = nonlin(data_transf_rot)
    scatter_plot(data_transf_rot_transf)

def nonlin2(data):
    # for col in data.columns:
    qt = QuantileTransformer(
        output_distribution='normal', n_quantiles=100, random_state=0)
    data_transf = qt.fit_transform(data.values)
    return pd.DataFrame(data_transf)


TRANSFORMERS = []
ROTATIONS = []
def nonlin(input):
    layer = []
    output = pd.DataFrame(dtype=float)
    for column in input.columns:
        t = GaussianTransformer(input[column])
        layer.append(t)
        output[column] = t.transf(input[column])
    TRANSFORMERS.append(layer)
    print(output.describe())
    return output

scatter_plot(data)

for i in range(M):
    print(f"{100*i/M}%")
    data_transf = nonlin(data)
    # for i in range(data.shape[1]):
    #    plt.plot(np.sort(data.iloc[:,i]), np.sort(data_transf.iloc[:,i]))
    # plt.show()
    # data_transf,s,v = np.linalg.svd(data_transf, full_matrices=False)
    # print(s)
    # u /= np.sqrt(N)
    # data = pd.DataFrame(data_transf)
    rot = random_rotation(data.shape[1])
    ROTATIONS.append(rot)
    data = pd.DataFrame(data_transf @ rot)

scatter_plot(data)

samples = pd.DataFrame(np.random.randn(NSAMP, data.shape[1]))
# scatter_plot(samples)

for i, (rot, layer) in enumerate(zip(ROTATIONS[::-1], TRANSFORMERS[::-1])):
    print(f"{100*i/M}%")
    samples = samples @ rot.T
    for column in samples.columns:
        t = layer[column]
        samples[column] = t.transf_back(samples[column])

scatter_plot(samples)


plt.show()

if XOR:
    # XOR TEST
    samples.columns = data_orig.columns
    samples = np.round(samples)
    print('XOR TRUE')
    print(samples['XOR'][samples['x1'] != samples['x2']].describe())
    print('XOR FALSE')
    print(samples['XOR'][samples['x1'] == samples['x2']].describe())
