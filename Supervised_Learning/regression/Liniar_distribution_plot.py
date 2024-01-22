import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# Regresie liniară
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)  # Eroare gaussiană

plt.scatter(x, y, label='Date observate')
plt.plot(x, 4 + 3 * x, 'r', label='Regresie liniară')
plt.title('Regresie liniară')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
