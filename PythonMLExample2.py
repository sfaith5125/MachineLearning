#  Code came from - https://towardsdatascience.com/how-to-build-your-first-machine-learning-model-in-python-e70fd1907cdd

import pandas as pd

# Code to run LinearRegression
def LinearRegressionModel (X_train, X_test, y_train, y_test):
    # us sklean to split the dataset into a training and testing set of data


    #start the traditional linear regression
    from sklearn.linear_model import LinearRegression
    # assign the variable to the model and then fit the model to the training data
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # apply the model to the training set
    y_lr_train_pred = lr.predict(X_train)
    y_lr_test_pred = lr.predict(X_test)

    # calculate the performance metrics
    from sklearn.metrics import mean_squared_error, r2_score
    lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
    lr_train_r2 = r2_score(y_train, y_lr_train_pred)
    lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
    lr_test_r2 = r2_score(y_test, y_lr_test_pred)

    #put the performance metrics into a dataframe so that it can printed out nicely
    lr_results = pd.DataFrame(['Linear regression',lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()
    lr_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

    print (lr_results)


def RandomForestModel (X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    rf = RandomForestRegressor(max_depth=2, random_state=42)
    rf.fit(X_train, y_train)

    y_rf_train_pred = rf.predict(X_train)
    y_rf_test_pred = rf.predict(X_test)

    from sklearn.metrics import mean_squared_error, r2_score
    rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
    rf_train_r2 = r2_score(y_train, y_rf_train_pred)
    rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
    rf_test_r2 = r2_score(y_test, y_rf_test_pred)

    rf_results = pd.DataFrame(['Random forest',rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()
    rf_results.columns = ['Method','Training MSE','Training R2','Test MSE','Test R2']

    print (rf_results)

# Download the csv, which is the sample data
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv')

# seperate the csv into 4 columns and 1 column
X = df.drop(['logS'], axis = 1)
y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

LinearRegressionModel(X_train, X_test, y_train, y_test)
print ('')
print ('____________________')
print ('')
RandomForestModel(X_train, X_test, y_train, y_test)
