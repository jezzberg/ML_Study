import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 1, 1, 1, 1, 1, 1])

model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

# test the model
p_pred = model.predict_proba(x) # predict x
y_pred = model.predict(x) # predict y based on the logistical model fitted above
score_ = model.score(x, y) # gives more data on the calculations and predictions
conf_m = confusion_matrix(y, y_pred) # array explaining the comparison between the y and y_pred
report = classification_report(y, y_pred) # compare the prediction to the actual observations

print(f"x: \n{x}")
print(f"y: \n{y}")

print(f"Intercept: {model.intercept_}")
print(f"coefficient: {model.coef_}")

print(f"y_actual: {y} \ny_predic: {y_pred}")

print(f"conf_m: \n{conf_m}")


print(f"report: \n{report}") 