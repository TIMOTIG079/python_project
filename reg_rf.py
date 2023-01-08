import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import datetime
# Read in data as pandas dataframe and display first 5 rows
original_features1 = pd.read_csv('/home/ubuntu/.jenkins/workspace/python-project/crop_data - Copy.csv')
original_features = pd.get_dummies(original_features1)

# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
original_labels = np.array(original_features['pressure'])

# Remove the labels from the features
# axis 1 refers to the columns
original_features= original_features.drop('pressure', axis = 1)

# Saving feature names for later use
original_feature_list = list(original_features.columns)

# Convert to numpy array
original_features = np.array(original_features)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
original_train_features, original_test_features, original_train_labels, original_test_labels = train_test_split(original_features, original_labels, test_size = 0.25, random_state = 42)

# The baseline predictions are the historical averages
baseline_preds = original_test_features[:, original_feature_list.index('visibility')]

# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - original_test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Import the model we are using
from sklearn.ensemble import RandomForestRegressor

# Instantiate model 
rf = RandomForestRegressor(n_estimators= 1000, random_state=42)

# Train the model on training data
rf.fit(original_train_features, original_train_labels);

# Use the forest's predict method on the test data
predictions = rf.predict(original_test_features)

# Calculate the absolute errors
errors = abs(predictions - original_test_labels)

# Print out the mean absolute error (mae)
print('Average model error:', round(np.mean(errors), 2), 'degrees.')

# Compare to baseline
improvement_baseline = 100 * abs(np.mean(errors) - np.mean(baseline_errors)) / np.mean(baseline_errors)
print('Improvement over baseline:', round(improvement_baseline, 2), '%.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / original_test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

state = original_features1.iloc[:,1]
lis_state = original_features1['State'].unique()
print(lis_state)
print(predictions)
plt.scatter(original_test_labels,original_test_labels,color = 'green')
plt.scatter(original_test_labels,predictions, color = 'red')
plt.title('Random Forest Regression Results')
plt.xlabel('originalpress')
plt.ylabel('predpress')
plt.show()
"""

X = datasets.iloc[:, 10:11].values
Y = datasets.iloc[:, 9].values
labels = np.array(datasets['pressure'])
# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(datasets,labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('pressure')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Fitting the Regression model to the dataset
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 100)
regressor.fit(train_features,train_labels)
predictions = regressor.predict(test_features)
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Average model error:', round(np.mean(errors), 2), 'degrees.')

# Compare to baseline
improvement_baseline = 100 * abs(np.mean(errors) - np.mean(baseline_errors)) / np.mean(baseline_errors)
print('Improvement over baseline:', round(improvement_baseline, 2), '%.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# Visualising the Random Forest Regression results in higher resolution and smoother curve
X_Grid = np.arange(min(X), max(X), 0.01)
X_Grid = X_Grid.reshape((len(X_Grid), 1))
plt.scatter(X,Y, color = 'red')
Y_Grid = np.arange(min(Y), max(Y), 0.01)
Y_Grid1 = Y_Grid.reshape((len(Y_Grid), 1))
#print(Y_Grid)
#print(X_Grid)

plt.plot(X_Grid, regressor.predict(X_Grid), color = 'blue')
plt.title('Random Forest Regression Results')
plt.xlabel('temp&press')
plt.ylabel('temp')
plt.show()
plt.scatter(X,Y, color = 'green')
plt.plot(Y_Grid, regressor.predict(Y_Grid1),color = 'black')
plt.title('Random Forest Regression Results')
plt.xlabel('temp&press')
plt.ylabel('temp')
plt.show()
print(Y_Grid)
print((Y))
accuracy_score(Y,Y)
"""

