import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from giide.giide import GIIDE 
# from giide_old import GaussianTransformer, nonlin
from sklearn.preprocessing import QuantileTransformer

N = 10000 # n input samples
NSAMP = N # n generate samples
M = 1000 # n layers model


CHOICE = "TRIGON"
# CHOICE = "ABS"
# CHOICE = "HETEROSCHED"
# CHOICE = "POWER"
# CHOICE = "XOR"
# CHOICE = "POSNEG"
# CHOICE = "BALL"


PLOTS = False # False

if CHOICE == "TRIGON":
    # TRIGON TEST
    x = np.random.uniform(0, 2 * np.pi, N)
    data = pd.DataFrame({
        'cos': np.cos(x),
        'sin': np.sin(x),
        # 'x': x,
        # 'x2': x**2,
    })

if CHOICE == "BALL":
    dim = 5
    x = np.random.randn(N, dim)
    data = pd.DataFrame(x)
    data = data.divide(np.sqrt((data**2).sum(1)), axis=0)


if CHOICE == "POSNEG":
    # ABS TEST
    x = np.random.uniform(-1,1, N)
    data = pd.DataFrame({
        'pos': np.maximum(x, 0.),
        'neg': np.minimum(x, 0.),
    })

if CHOICE == "HETEROSCHED":
    # HETEROSCHED TEST
    y = np.random.uniform(-1, 1, N)
    x = np.random.randn(N) * y
    data = pd.DataFrame({
        'x': x,
        'y': y,
    })

if CHOICE == "POWER":
    # POWER TEST
    # x = np.random.randn(N)
    # if we limit to [0,1] it gets to zero error
    x = np.random.uniform(-1,1, N)
    data = pd.DataFrame({
        'x': x,
        'x2': x**2,
        'x3': x**3,
        'x4': x**4,
    })

if CHOICE == "ABS":
    # ABS TEST
    x = np.random.uniform(-1,1, N)
    data = pd.DataFrame({
        'x': x,
        'sign(x)': np.sign(x),
    })

if CHOICE == "XOR":
    # XOR TEST
    x1 = np.sign(np.random.uniform(-1,1, N)) * 1.0
    x2 = np.sign(np.random.uniform(-1,1, N)) * 1.0
    data = pd.DataFrame({
        'x1': x1 * 1.0,
        'x2': x2 * 1.0,
        'XOR': (x1 != x2) * 1.0,
        # 'OR': (x1.astype(bool) | x2.astype(bool)) * 1.0,
        # 'NOISE': np.random.randn(N),
    })


data_orig = pd.DataFrame(data, copy=True)

def scatter_plot(mydata):

    # Assume df is your DataFrame
    # plt.figure()
    pd.plotting.scatter_matrix(pd.DataFrame(mydata), figsize=(10, 10), diagonal='kde')
    plt.tight_layout()

    # plt.scatter(
    #     mydata.iloc[:, 0], mydata.iloc[:, 1])
    # plt.show()

# def random_rotation(size):
#     a = np.random.randn(size,size),
#     q,_ = np.linalg.qr(a)
#     return q[0]

# INTERACTIVE = False
# if INTERACTIVE:
#     scatter_plot(data)
#     data_transf = nonlin(data)
#     scatter_plot(data_transf)
#     rot = random_rotation(2)
#     data_transf_rot = data_transf @ rot
#     scatter_plot(data_transf_rot)
#     data_transf_rot_transf = nonlin(data_transf_rot)
#     scatter_plot(data_transf_rot_transf)

# def nonlin2(data):
#     # for col in data.columns:
#     qt = QuantileTransformer(
#         output_distribution='normal', n_quantiles=100, random_state=0)
#     data_transf = qt.fit_transform(data.values)
#     return pd.DataFrame(data_transf)


# TRANSFORMERS = []
# ROTATIONS = []
# def nonlin(input):
#     layer = []
#     output = pd.DataFrame(dtype=float)
#     for column in input.columns:
#         t = GaussianTransformer(input[column])
#         layer.append(t)
#         output[column] = t.transf(input[column])
#     TRANSFORMERS.append(layer)
#     print(output.describe())
#     return output

if PLOTS:
    scatter_plot(data)

model = GIIDE(M, robust=False)

data_transformed = model.fit_transform(data.values)

# for i in range(M):
#     print(f"{100*i/M}%")
#     data_transf = nonlin(data)
#     # for i in range(data.shape[1]):
#     #    plt.plot(np.sort(data.iloc[:,i]), np.sort(data_transf.iloc[:,i]))
#     # plt.show()
#     # data_transf,s,v = np.linalg.svd(data_transf, full_matrices=False)
#     # print(s)
#     # u /= np.sqrt(N)
#     # data = pd.DataFrame(data_transf)
#     rot = random_rotation(data.shape[1])
#     ROTATIONS.append(rot)
#     data = pd.DataFrame(data_transf @ rot)

if PLOTS:
    scatter_plot(data_transformed)

samples = pd.DataFrame(np.random.randn(NSAMP, data.shape[1]))
# scatter_plot(samples)

samples_back = pd.DataFrame(
    model.transform_back(samples.values), columns=data.columns)

# for i, (rot, layer) in enumerate(zip(ROTATIONS[::-1], TRANSFORMERS[::-1])):
#     print(f"{100*i/M}%")
#     samples = samples @ rot.T
#     for column in samples.columns:
#         t = layer[column]
#         samples[column] = t.transf_back(samples[column])

if PLOTS:
    scatter_plot(samples_back)


plt.show()

if CHOICE in ["TRIGON", "BALL"]:
    print('MEAN ERROR', np.sqrt(((1 - (samples_back**2).sum(1))**2).mean()))

if CHOICE == "ABS":
    print('MEAN ERROR', np.mean(np.abs(np.sign(samples_back['x']) - samples_back['sign(x)'])))

if CHOICE == "POWER":
    print('ERROR x2', np.linalg.norm(samples_back['x2'] - samples_back['x']**2) / np.sqrt(len(samples)))
    print('ERROR x3', np.linalg.norm(samples_back['x3'] - samples_back['x']**3) / np.sqrt(len(samples)))
    print('ERROR x4', np.linalg.norm(samples_back['x4'] - samples_back['x']**4) / np.sqrt(len(samples)))


if CHOICE == "XOR":
    # XOR TEST
    # samples.columns = data_orig.columns
    samples_back = np.round(samples_back)
    print('XOR MEAN ERROR', 
        (samples_back['XOR'] != (samples_back['x1'] != samples_back['x2'])).mean())
    # print('OR MEAN ERROR', 
    #     (samples_back['XOR'] != (samples_back['x1'] | samples_back['x2'])).mean())


# test derivative
if CHOICE == 'TRIGON':
    # the zero is a placeholder for unknown
    # x0 = np.array([[0, .5]])
    # loss = lambda x: (model.transform(x)**2).sum()
    # print(loss(x0))

    # forward (infer missing values), doesn't work very well
    my_y = .5
    loss_func = lambda unknown: (model.transform(np.array([[unknown[0], my_y]]))**2).sum()
    from scipy import optimize as opt
    result = opt.fmin_l_bfgs_b(loss_func, x0=[np.random.randn()], approx_grad=True)
    print(result)
    print(result[0][0]**2 + my_y**2)

    # backward (conditional distribution)
    my_y = .25
    random_seed = np.random.randn(2)
    loss_func = lambda transformed: (model.transform_back(transformed.reshape(1,2))[0, 1] - my_y)**2
    loss_func(random_seed)
    result = opt.fmin_l_bfgs_b(loss_func, x0=random_seed, approx_grad=True)
    print(result)
    print(model.transform_back(result[0].reshape(1,2)))
    print((model.transform_back(result[0].reshape(1,2))**2).sum())