import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Distribuție normală
mu = 0  # Medie
sigma = 1  # Deviație standard

x = np.linspace(-5, 5, 100)
pdf = norm.pdf(x, mu, sigma)  # Funcția de densitate de probabilitate

plt.plot(x, pdf, label='Distribuție normală')
plt.title('Distribuție normală')
plt.xlabel('x')
plt.ylabel('Densitatea de probabilitate')
plt.legend()
plt.grid(True)
plt.show()
