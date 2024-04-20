from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder 
from sklearn.metrics import  accuracy_score, classification_report 
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import category_encoders as label_encoder
from sklearn.metrics import accuracy_score

# Step 1. Prepare Data
# fetch dataset 
student_performance = fetch_ucirepo(id=320)
with open('Data_visualization.txt', 'w') as file:
    file.write(f"\n\n1: \n{student_performance}")
  
# get data as DataFrame  + write all data to data_visualization.txt in order to analyze & understand the dataset
x = pd.DataFrame(student_performance.data.features)
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\n1 - Features: \n{x}")
y = pd.DataFrame(student_performance.data.targets) 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\n2 - Targets: \n{y}")  

# metadata 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\nmetadata: \n{student_performance.metadata}") 
  
# variable information 
with open('data_visualization.txt', 'a') as file:
    file.write(f"\n\nvariable info: \n{student_performance.variables}") 


# split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)




# Step 2: Encode and split the data into training and testing subsets
# Encode categorical variables
print(f"cols types: \n{x_train.dtypes}")
# encoder = label_encoder.OrdinalEncoder(cols=['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic'])

# Step 6: build the model based on the most 5 important features
columns_to_drop = ['Fedu', 'Mjob', 'famrel', 'Walc', 'reason', 'Medu', 'studytime',
                   'Fjob', 'traveltime', 'Dalc', 'guardian', 'failures', 'famsup',
                   'activities', 'romantic', 'sex', 'famsize', 'school', 'address',
                   'internet', 'nursery', 'Pstatus', 'schoolsup', 'higher', 'paid']

x_train = x_train.drop(columns_to_drop, axis=1)
x_test = x_test.drop(columns_to_drop, axis=1)


print(f"x_train encoded: \n{x_train.head()}")


# Step 3: Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 4: Training the Model
rf_classifier.fit(x_train, y_train)

# Step 4: Making Predictions
y_pred = rf_classifier.predict(x_test)

# Step 5: Evaluating the Model
y_test = np.ravel(y_test)
y_pred = np.ravel(y_pred)
print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# view the feature scores
feature_scores = pd.Series(rf_classifier.feature_importances_, index=x_train.columns).sort_values(ascending=False)
print(f"feature_scores: \n{feature_scores}")

# Print the Confusion Matrix and slice it into four pieces
cm = confusion_matrix(y_test, y_pred)
print('Confusion matrix\n\n', cm)

print(classification_report(y_test, y_pred))
