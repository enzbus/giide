import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from giide.gaussian_marginals import gaussian_marginalizer

# sin
# samples = np.sin(np.random.uniform(0, 2*np.pi, size=1000))

# integers
# samples = np.linspace(0,1,100)

# bool
# samples = (np.random.uniform(size=5) > .33) * 1.
# samples = np.sort(samples)

# floored
samples = np.random.randn(100)
samples = np.maximum(samples, -.33)

samples -= samples.mean()
samples /= np.sqrt((samples**2).mean())

print('NORMALTEST')
print(sp.stats.normaltest(samples))

_ = sp.stats.probplot(samples, dist="norm", plot=plt)

nodes, node_values = gaussian_marginalizer(samples=samples, slopes_regularizer=1e-2)

plt.figure()
plt.plot(nodes, node_values)

slopes = np.diff(node_values) / np.diff(nodes)
print('SLOPES')
print(slopes)
plt.figure()
plt.plot(slopes)


transformed_samples = np.interp(samples, node_values, nodes)


print('NORMALTEST TRANSFORMED')
print(sp.stats.normaltest(transformed_samples))

plt.figure()
_ = sp.stats.probplot(transformed_samples, dist="norm", plot=plt)

plt.show()