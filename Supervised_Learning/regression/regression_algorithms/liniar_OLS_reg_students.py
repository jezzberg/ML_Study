import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

students_data = pd.read_csv("students_data.csv")

# create class size and test scores arrays
class_size = students_data[["class_size"]]
test_scores = students_data[["test_score"]]

# fit the model to the data
reg = LinearRegression().fit(class_size, test_scores)

# print the coefficints
print(f"Intercept {reg.intercept_}") # beta 0
print(f"Coefficient {reg.coef_}") # beta 1