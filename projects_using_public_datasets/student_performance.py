from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import accuracy_score 
import numpy as np


# Step 1. Prepare Data
# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
with open('Data_visualization.txt', 'w') as file:
    file.write(f"\n\n1: \n{student_performance}")
  
# data (as pandas dataframes) + write all data to data_visualization.txt in order to analyze & understand the dataset
x = student_performance.data.features 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\n1 - Features: \n{x}")
y = student_performance.data.targets 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\n2 - Targets: \n{y}")  

# metadata 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\nmetadata: \n{student_performance.metadata}") 
  
# variable information 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\nvariable info: \n{student_performance.variables}") 


# Step 2: Encode and split the data into training and testing subsets
# Encode categorical variables
label_encoder = LabelEncoder()
for col in x.select_dtypes(include=['object']).columns:
    x.loc[:, col] = label_encoder.fit_transform(x[col])
print(f"x encoded: \n{x}")

# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


# Step 3: Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier()

# Step 4: Training the Model
rf_classifier.fit(x_train, y_train)

# Step 4: Making Predictions
y_pred = rf_classifier.predict(x_test)

# Step 5: Evaluating the Model

y_test_flat = np.ravel(y_test)
y_pred_flat = np.ravel(y_pred)
accuracy = accuracy_score(y_test_flat, y_pred_flat)
print("\nAccuracy:", accuracy)
