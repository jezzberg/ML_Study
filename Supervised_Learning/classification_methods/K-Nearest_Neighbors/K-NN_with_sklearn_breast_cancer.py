import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()

breast_cancer = load_breast_cancer() # extract data
x = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names) # keep only some data 
x = x[['mean area', 'mean compactness']]
y = pd.Categorical.from_codes(breast_cancer.target, breast_cancer.target_names)
y = pd.get_dummies(y, drop_first=True) # convert to indicator values

print(f"breast_cancer.feature_names = \n{breast_cancer.feature_names}")
print(f"breast_cancer.data = \n{breast_cancer.data}")
print(f"y = \n{y}")

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1) # train and test the model

knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(x_train, y_train)

sns.scatterplot(
    x='mean area',
    y='mean compactness',
    hue='benign',
    data=x_test.join(y_test, how='outer')
)
plt.show()

y_pred = knn.predict(x_test)
plt.scatter(
    x_test['mean area'],
    x_test['mean compactness'],
    c=y_pred,
    cmap='coolwarm',
    alpha=0.7
)
plt.show()

print(f"confusion matrix: \n{confusion_matrix(y_test, y_pred)}")