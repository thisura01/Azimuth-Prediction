# **Update the dataset**

### Import libraries

import pandas as pd


### Read the dataset

data = pd.read_csv('/content/Merged_all.csv')

### Create new columns for longitude and latitude in km

data['Longitude_km'] = data['ADJ_LONGITUDE'] * 111    # Assuming 1 degree of longitude is approximately 111 km
data['Latitude_km'] = data['ADJ_LATITUDE'] * 111      # Assuming 1 degree of latitude is approximately 111 km

### Drop first two columns

data = data.drop(['ADJ_LONGITUDE', 'ADJ_LATITUDE'], axis=1)

### Print the updated dataset

print(data)

# **Model building and training**

### Import libraries

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


### Prepare the input features (X) and target variable (y)

making cell key categorical

from sklearn.preprocessing import LabelEncoder


le = LabelEncoder()
data['CELL_KEY_H'] = data['CELL_KEY_H'].astype('category')
# Fit and transform the label encoder on the 'CELL_KEY_H' column
data['CELL_KEY_H'] = le.fit_transform(data['CELL_KEY_H'])

# Now 'CELL_KEY_H' is a categorical variable with numerical labels



# Now, you need to set your X and y based on the updated DataFrame
X = data.drop(['CELL_AZIMUTH'], axis=1)  # Drop the target variable
y = data['CELL_AZIMUTH']

### Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

### Create a random forest regression model

from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 150]}


rf_regressor = RandomForestRegressor(n_estimators=param_grid, random_state=42)
grid_search = GridSearchCV(estimator=rf_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

### Train the random forest model

# Extract the best number of estimators from the best_params dictionary
best_n_estimators = best_params['n_estimators']

# Create a new random forest regressor with the best number of estimators
rf_regressor = RandomForestRegressor(n_estimators=best_n_estimators, random_state=42)

# Fit the model to the training data
rf_regressor.fit(X_train, y_train)

# **Evaluate the training and testing sets**

### Import libraries

from sklearn.metrics import mean_squared_error, r2_score

### Predict on both training and testing data

y_train_pred = rf_regressor.predict(X_train)
y_test_pred = rf_regressor.predict(X_test)

### Calculate evaluation metrics on training and testing data


train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

### Print the evaluation metrics

print("Training Mean Squared Error:", train_mse)
print("Testing Mean Squared Error:", test_mse)
print("Training R-squared:", train_r2)
print("Testing R-squared:", test_r2)

# **Model accuracy**

### Calculate the R-squared score


r2_score_percentage = r2_score(y_test, y_test_pred) * 100

### Print the model's accuracy


print("Model Accuracy (R-squared): {:.2f}%".format(r2_score_percentage))

# **Overfitting/Underfitting?**

### Import the libraries

import numpy as np
import matplotlib.pyplot as plt

###  Initialize a list to store training and testing errors


train_errors = []
test_errors = []

### Define a range of training set sizes


training_sizes = np.linspace(0.1, 1.0, 10, endpoint=True)

### Initiate a loop that will iterate through each of the 10 values in 'training_sizes'

for size in training_sizes:
    # Calculate the number of samples based on the fraction of training data
    num_samples = int(len(X_train) * size)

    # Create a random forest regression model
    rf_regressor = RandomForestRegressor(n_estimators=50, random_state=42)

    # Fit the model to the training data
    rf_regressor.fit(X_train[:num_samples], y_train[:num_samples])

    # Predict on both training and testing data
    y_train_pred = rf_regressor.predict(X_train)
    y_test_pred = rf_regressor.predict(X_test)

    # Calculate Mean Squared Error for training and testing data
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Append errors to the lists
    train_errors.append(train_mse)
    test_errors.append(test_mse)

### Plot the learning curve


plt.figure(figsize=(10, 6))
plt.plot(training_sizes * len(X_train), train_errors, label='Training Error', color='blue')
plt.plot(training_sizes * len(X_train), test_errors, label='Testing Error', color='red')
plt.xlabel('Number of Training Samples')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve')
plt.legend()
plt.show()

# **Predict the azimuth of an antenna**

### Obtain inputs from the user

longitude_degrees = float(input("Enter the longitude of the new antenna in degrees: "))
latitude_degrees = float(input("Enter the latitude of the new antenna in degrees: "))


### Convert longitude and latitude from degrees to km

longitude_km = longitude_degrees * 111
latitude_km = latitude_degrees * 111

### Obtain other inputs from the user

cell_key = int(input("Enter the cell key of the new antenna: "))
mean_rsrp = float(input("Enter the mean_rsrp of the new antenna: "))
instance_count = int(input("Enter the instance count of the new antenna: "))

### Create a DataFrame with the user inputs

new_antenna = pd.DataFrame({
    'CELL_KEY_H': [cell_key],
    'MEAN_RSRP': [mean_rsrp],
    'INSTANCE_COUNT': [instance_count],
    'Longitude_km': [longitude_km],
    'Latitude_km': [latitude_km]
})

### Predict the actual azimuth using the model

predicted_azimuth = rf_regressor.predict(new_antenna)
print('Predicted Azimuth:', predicted_azimuth)


# **Identify antennas that has azimuth difference of more than 30 degrees**

### Calculate predicted_azimuth for all antennas in the dataset

predicted_azimuth_all = pd.DataFrame({
    'CELL_KEY_H': data['CELL_KEY_H'],
    'Predicted_Azimuth': rf_regressor.predict(data[['CELL_KEY_H', 'MEAN_RSRP', 'INSTANCE_COUNT', 'Longitude_km', 'Latitude_km']])
})

# Print or use the 'predicted_azimuth_all' DataFrame as needed
print(predicted_azimuth_all)




average_predicted_azimuths = predicted_azimuth_all.groupby('CELL_KEY_H')['Predicted_Azimuth'].mean()

# Print or use the 'average_predicted_azimuths' as needed
print(average_predicted_azimuths)

distinct_cell_key_azimuth = data[['CELL_KEY_H', 'CELL_AZIMUTH']].drop_duplicates()

# Print or use the 'distinct_cell_key_azimuth' DataFrame as needed
print(distinct_cell_key_azimuth)

### Calculate the difference between predicted_azimuth_all and CELL_AZIMUTH

merged_data = pd.merge(distinct_cell_key_azimuth, average_predicted_azimuths, on='CELL_KEY_H', how='left')

merged_data['Azimuth_Difference'] = merged_data['Predicted_Azimuth'] - merged_data['CELL_AZIMUTH']

print(merged_data)

# Create a new dataset with antennas having an azimuth difference greater than 30

azimuth_difference_threshold = 30


filtered_data = merged_data[merged_data['Azimuth_Difference'] > azimuth_difference_threshold]

print(filtered_data)

max_azimuth_difference = merged_data['Azimuth_Difference'].max()
print(max_azimuth_difference)

# **Save the dataset of the antennas list that has an azimuth difference more than 30 degrees to an Excel file**

filtered_data.to_excel('antennas_list.xlsx', index=False)

merged_data.to_excel('antennasListWithDifferences.xlsx', index=False)

# **Comparing Actual azimuths and Predicted azimuths**

### Create the scatter plot


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, color='red', label='Predicted', alpha=0.5)
plt.scatter(y_test, y_test, color='blue', label='Actual', alpha=0.5)
plt.xlabel('Actual Azimuth')
plt.ylabel('Predicted Azimuth')
plt.title('Actual vs. Predicted Azimuth')
plt.legend()
plt.grid(True)
plt.show()
