# Clasificator Bayes
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt

# Generăm setul de date cu caracteristici informative (n_informative=2) și redundante (n_redundant=0)
X, y = make_classification(n_samples=16000, n_features=2, n_informative=2, n_redundant=0, n_classes=3, n_clusters_per_class=1, random_state=91)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', marker='o', edgecolors='k')
plt.title('Exemplu pentru clasificatorul Bayes')
plt.xlabel('Caracteristică 1')
plt.ylabel('Caracteristică 2')

# Antrenăm și vizualizăm clasificatorul Bayes
clf = GaussianNB()
clf.fit(X, y)

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
plt.show()
