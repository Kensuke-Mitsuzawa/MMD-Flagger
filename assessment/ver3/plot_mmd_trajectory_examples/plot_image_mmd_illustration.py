import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

"""A script of rendering Two Gaussian DIstribution.
The plot is used for the intuitive plotting of Maximum Mean Discrepacny (MMD)."""

# Set up figure
fig = plt.figure(figsize=(10, 5))

# --------------------
# Feature Space Plot
# --------------------
ax1 = fig.add_subplot(projection='3d')

# Meshgrid
x = y = np.linspace(-10, 10, 200)
x, y = np.meshgrid(x, y)
pos = np.dstack((x, y))

# Two Gaussian components
mu1 = [-5.0, 0]
mu2 = [5.0, 0]
cov = [[2.0, 0], [0, 2.0]]
rv1 = multivariate_normal(mean=mu1, cov=cov)
rv2 = multivariate_normal(mean=mu2, cov=cov)

# Evaluate each Gaussian individually
z1 = 0.5 * rv1.pdf(pos)
z2 = 0.5 * rv2.pdf(pos)

# Plot each Gaussian separately with custom colors
ax1.plot_surface(x, y, z1, cmap='seismic', alpha=0.9)
ax1.plot_surface(x, y, z2, cmap='Greens', alpha=0.6)


# Add labels
ax1.text(*mu1, np.max(z1) * 1.05, '$\\mathcal{P}$', fontsize=25, color='red')
ax1.text(*mu2, np.max(z2) * 1.05, '$\\mathcal{Q}$', fontsize=25, color='green')
ax1.axis('off')

plt.tight_layout()
plt.show()
