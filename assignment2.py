import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3.csv"
df_train = pd.read_csv(train_url)
df_train.head

df_train.columns
len(df_train.columns)

# Load the testing data
test_url = "https://github.com/dustywhite7/Econ8310/raw/master/AssignmentData/assignment3test.csv"
df_test = pd.read_csv(test_url)
df_test.head

df_test.columns

# Define features and target variable
X_train = df_train.drop(columns=['id', 'DateTime','meal'])
y_train = df_train['meal']
X_test = df_test.drop(columns=['id', 'DateTime','meal'])

# Train the RandomForestClassifier model
model = DecisionTreeClassifier(max_depth=200, random_state=42)
modelFit = model.fit(X_train, y_train)

# Accuracy score on training set
pred_train = modelFit.predict(X_train)
accuracy_train = accuracy_score(y_train, pred_train)
print(f"Training Accuracy: {accuracy_train:.4f}")

# Predictions on test set
pred = modelFit.predict(X_test)
len(pred)
pred = pred.astype(int)
pred = pred.tolist()
pred
len(pred)